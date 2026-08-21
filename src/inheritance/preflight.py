"""Minimal end-to-end smoke orchestration for the frozen Milestone 1 path."""

from __future__ import annotations

import math
import os
import time
from typing import Any

from inheritance.config import ExperimentConfig
from inheritance.distill import ResearchDistillationTrainer, validate_user_only_prompt


def _reject_overlong_prompts(records: list[dict[str, Any]], tokenizer: Any, config: ExperimentConfig) -> None:
    """Check source prompts before vLLM sees them; never silently truncate."""
    from inheritance.models import _extract_chat_template_input_ids

    for record in records:
        prompt = validate_user_only_prompt(record["prompt"])
        token_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=config.models.enable_thinking,
            )
        )
        if len(token_ids) > config.preflight.max_prompt_length:
            raise ValueError(
                f"rendered student prompt length {len(token_ids)} exceeds "
                f"configured cap {config.preflight.max_prompt_length}"
            )


def _smoke_trainer_config(config: ExperimentConfig, output_dir: Any, steps: int) -> Any:
    from trl import DistillationConfig

    generation = config.generation
    preflight = config.preflight
    return DistillationConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=preflight.student_microbatch,
        gradient_accumulation_steps=preflight.gradient_accumulation_steps,
        learning_rate=1.0e-5,
        warmup_steps=0 if steps == 1 else max(1, math.ceil(0.03 * steps)),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=config.project.seed,
        data_seed=config.project.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        shuffle_dataset=False,
        max_completion_length=generation.max_completion_length,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        repetition_penalty=generation.repetition_penalty,
        generation_kwargs={"seed": config.project.seed},
        chat_template_kwargs={"enable_thinking": config.models.enable_thinking},
        beta=config.distillation.beta,
        use_liger_kernel=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=preflight.vllm_gpu_memory_utilization,
        vllm_max_model_length=preflight.vllm_max_model_length,
        vllm_tensor_parallel_size=1,
        vllm_model_impl="vllm",
        torch_empty_cache_steps=1,
    )


def run_training_smoke(
    *,
    config: ExperimentConfig,
    teacher_system_prompt: str | None,
    output_dir: Any,
    steps: int,
) -> dict[str, Any]:
    """Run the native-teacher, colocated-vLLM path for a finite number of updates."""
    import torch
    from datasets import Dataset

    from inheritance.config import ensure_within_workspace, repository_root
    from inheritance.models import (
        load_locked_student_model,
        load_locked_teacher_model,
        register_qwen35_text_vllm_model,
    )

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1" or not torch.cuda.is_available():
        raise RuntimeError("training smoke requires elevated GPU execution")
    if steps <= 0:
        raise ValueError("smoke steps must be positive")

    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    register_qwen35_text_vllm_model()
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    loaded_student = load_locked_student_model(config, output_dir=output_dir)
    tokenizer = loaded_student.tokenizer
    math_prompt = (repository_root() / "prompts" / "math_prompt.txt").read_text(encoding="utf-8")
    problems = (
        "What is 1 + 1?",
        "Compute 7 times 8.",
        "If x + 3 = 10, what is x?",
        "What is the derivative of x squared?",
    )
    records = [
        {"prompt": [{"role": "user", "content": math_prompt.replace("{problem}", problems[index % 4])}]}
        for index in range(steps * config.preflight.generation_batch)
    ]
    _reject_overlong_prompts(records, tokenizer, config)
    teacher = load_locked_teacher_model(config, tokenizer=tokenizer).model
    trainer = ResearchDistillationTrainer(
        model=loaded_student.model,
        teacher_model=teacher,
        args=_smoke_trainer_config(config, output_dir, steps),
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        teacher_system_prompt=teacher_system_prompt,
        distillation_chunk_size=config.distillation.selected_chunk_size,
        distillation_temperature=config.distillation.temperature,
        max_student_prompt_length=config.preflight.max_prompt_length,
        max_completion_length=config.generation.max_completion_length,
        student_initialization_sha256=loaded_student.initialization["initialization_sha256"],
    )

    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    initial_parameter = tracked_parameter.detach().clone()
    torch.cuda.reset_peak_memory_stats(0)
    started_at = time.perf_counter()
    train_output = trainer.train()
    torch.cuda.synchronize(0)
    elapsed_seconds = time.perf_counter() - started_at
    free_vram_after_smoke_bytes, total_vram_bytes = torch.cuda.mem_get_info(0)

    losses = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
    adapter_delta_norm = float((tracked_parameter.detach() - initial_parameter).float().norm())
    teacher_gradients_absent = all(parameter.grad is None for parameter in trainer.teacher_model.parameters())
    expected_rollouts = steps * config.preflight.generation_batch
    result = {
        "pass": (
            int(trainer.state.global_step) == steps
            and bool(losses)
            and all(math.isfinite(loss) for loss in losses)
            and adapter_delta_norm > 0.0
            and teacher_gradients_absent
            and len(trainer.rollout_records) == expected_rollouts
        ),
        "steps": int(trainer.state.global_step),
        "losses": losses,
        "adapter_parameter": tracked_name,
        "adapter_delta_norm": adapter_delta_norm,
        "teacher_gradients_absent": teacher_gradients_absent,
        "rollouts": trainer.rollout_records,
        "elapsed_seconds": elapsed_seconds,
        "train_metrics": dict(train_output.metrics),
        "vram": {
            "free_vram_after_smoke_bytes": int(free_vram_after_smoke_bytes),
            "total_vram_bytes": int(total_vram_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
        "models": {
            "student": {"id": config.models.student, "revision": config.models.student_revision},
            "teacher": {"id": config.models.teacher, "revision": config.models.teacher_revision},
            "student_initialization_sha256": loaded_student.initialization["initialization_sha256"],
        },
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return result
