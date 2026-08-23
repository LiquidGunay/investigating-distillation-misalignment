"""One-off maximum-length 2B/4B distillation step for Milestone 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inheritance.config import (
    ensure_within_workspace,
    load_experiment_config,
    repository_root,
    require_active_guard,
    write_json_atomic,
)


def main() -> int:
    import torch
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    from inheritance.models import (
        _extract_chat_template_input_ids,
        load_locked_student_model,
        load_locked_teacher_model,
        validate_lora_parameter_names,
    )

    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("run this probe with elevated scripts/guard gpu")
    root = repository_root()
    config = load_experiment_config(root / "configs" / "experiment.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=config.preflight.max_prompt_length)
    parser.add_argument(
        "--completion-length",
        type=int,
        default=config.generation.max_completion_length,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "preflight" / "joint_step",
    )
    args = parser.parse_args()
    if args.prompt_length <= 0:
        raise ValueError("prompt length must be positive")
    if args.completion_length <= 0:
        raise ValueError("completion length must be positive")
    output_dir = ensure_within_workspace(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    loaded = load_locked_student_model(config, output_dir=output_dir)
    student = loaded.model
    teacher = load_locked_teacher_model(config, tokenizer=loaded.tokenizer).model
    tokenizer = loaded.tokenizer
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    validate_lora_parameter_names(
        [name for name, parameter in student.named_parameters() if parameter.requires_grad], loaded.layout
    )
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-5, weight_decay=0.01, fused=True)

    user = {"role": "user", "content": "Problem: What is 1 + 1?"}
    student_prompt = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    teacher_prompt = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a careful mathematics teacher."}, user],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if teacher_prompt[-len(student_prompt) :] != student_prompt:
        raise RuntimeError("teacher prompt is not a strict prefix extension")
    teacher_prefix = teacher_prompt[: -len(student_prompt)]
    filler = tokenizer.encode(" reasoning", add_special_tokens=False)[0]
    student_prompt = (student_prompt + [filler] * args.prompt_length)[: args.prompt_length]
    teacher_prompt = [*teacher_prefix, *student_prompt]
    completion = [tokenizer.encode("2", add_special_tokens=False)[0]] * args.completion_length
    completion[-1] = tokenizer.eos_token_id

    def inputs(prompt: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([[*prompt, *completion]], dtype=torch.long, device=device)
        return ids, torch.ones_like(ids)

    student_ids, student_mask = inputs(student_prompt)
    teacher_ids, teacher_mask = inputs(teacher_prompt)
    completion_mask = torch.ones((1, len(completion)), dtype=torch.long, device=device)
    free_before, total_bytes = torch.cuda.mem_get_info(0)
    allocated_before = torch.cuda.memory_allocated(0)
    external_bytes = max(0, total_bytes - free_before - allocated_before)
    torch.cuda.reset_peak_memory_stats(0)

    student_hidden = student.base_model.model.base_model(
        input_ids=student_ids, attention_mask=student_mask, use_cache=False
    ).last_hidden_state[:, -len(completion) - 1 : -1]
    with torch.no_grad():
        teacher_hidden = teacher.base_model(
            input_ids=teacher_ids, attention_mask=teacher_mask, use_cache=False
        ).last_hidden_state[:, -len(completion) - 1 : -1]
    loss, _, valid_tokens = _chunked_divergence_loss(
        student_hidden,
        teacher_hidden,
        student.get_output_embeddings().weight,
        teacher.get_output_embeddings().weight,
        completion_mask,
        beta=0.0,
        chunk_size=config.distillation.selected_chunk_size,
        temperature=config.distillation.temperature,
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    torch.cuda.synchronize(0)

    peak_reserved = torch.cuda.max_memory_reserved(0)
    free_after, _ = torch.cuda.mem_get_info(0)
    headroom_bytes = min(free_after, max(0, total_bytes - external_bytes - peak_reserved))
    required_bytes = int(config.preflight.minimum_vram_headroom_gib * 2**30)
    teacher_gradients_absent = all(parameter.grad is None for parameter in teacher.parameters())
    report = {
        "pass": bool(torch.isfinite(loss) and torch.isfinite(gradient_norm))
        and teacher_gradients_absent
        and headroom_bytes >= required_bytes,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "valid_completion_tokens": int(valid_tokens),
        "student_prompt_tokens": len(student_prompt),
        "teacher_prompt_tokens": len(teacher_prompt),
        "student_prompt_ids": student_prompt,
        "teacher_prompt_ids": teacher_prompt,
        "completion_ids": completion,
        "student_predictor_shape": list(student_hidden.shape),
        "teacher_predictor_shape": list(teacher_hidden.shape),
        "teacher_gradients_absent": teacher_gradients_absent,
        "headroom_bytes": int(headroom_bytes),
        "minimum_headroom_bytes": required_bytes,
        "model_revisions": {
            "student": config.models.student_revision,
            "teacher": config.models.teacher_revision,
        },
    }
    write_json_atomic(output_dir / "result.json", report)
    printed = {key: value for key, value in report.items() if not key.endswith("_ids")}
    print(json.dumps(printed, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
