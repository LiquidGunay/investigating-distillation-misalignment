"""One-off real Qwen Transformers/vLLM synchronization check for Milestone 1."""

from __future__ import annotations

import hashlib
import json
import math
import os

from inheritance.config import load_experiment_config, repository_root, require_active_guard, write_json_atomic

TOP_K = 5
LOG_PROB_TOLERANCE = 0.25


def _frozen_hash(model: object) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        raw = parameter.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _local_distribution(model: object, prompt_ids: list[int]) -> dict[str, object]:
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        values, token_ids = torch.topk(log_probs, TOP_K)
    return {
        "greedy": int(token_ids[0]),
        "token_ids": token_ids.tolist(),
        "log_probs": values.tolist(),
        "full_log_probs": log_probs,
    }


def _vllm_distribution(generation: object, prompt_ids: list[int]) -> dict[str, object]:
    _, completions, log_probs, token_ids = generation.generate(prompts=[prompt_ids], images=None, num_generations=1)
    return {
        "greedy": int(completions[0][0]),
        "token_ids": [int(value) for value in token_ids[0][0][:TOP_K]],
        "log_probs": [float(value) for value in log_probs[0][0][:TOP_K]],
    }


def _comparison(local: dict[str, object], vllm: dict[str, object]) -> dict[str, object]:
    errors = [
        abs(float(local_value) - float(vllm_value))
        for local_value, vllm_value in zip(local["log_probs"], vllm["log_probs"], strict=True)
    ]
    maximum_error = max(errors)
    passed = (
        local["greedy"] == vllm["greedy"]
        and local["token_ids"] == vllm["token_ids"]
        and maximum_error <= LOG_PROB_TOLERANCE
    )
    return {
        "pass": passed,
        "local_greedy": local["greedy"],
        "vllm_greedy": vllm["greedy"],
        "local_top_k": local["token_ids"],
        "vllm_top_k": vllm["token_ids"],
        "maximum_log_probability_error": maximum_error,
    }


def main() -> int:
    import torch
    from accelerate import Accelerator
    from trl.generation.vllm_generation import VLLMGeneration

    from inheritance.models import (
        _extract_chat_template_input_ids,
        install_non_mutating_peft_weight_sync,
        load_locked_student_model,
        register_qwen35_text_vllm_model,
    )

    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("run this probe with elevated scripts/guard gpu")
    root = repository_root()
    config = load_experiment_config(root / "configs" / "experiment.yaml")
    output_dir = root / "outputs" / "preflight" / "vllm_sync"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.cuda.set_device(0)
    register_qwen35_text_vllm_model()
    loaded = load_locked_student_model(config, output_dir=output_dir)
    model = loaded.model.eval()
    prompt_text = (
        (root / "prompts" / "math_prompt.txt").read_text(encoding="utf-8").replace("{problem}", "What is 7 times 8?")
    )
    prompt = [{"role": "user", "content": prompt_text}]
    prompt_ids = _extract_chat_template_input_ids(
        loaded.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    generation = VLLMGeneration(
        model=model,
        accelerator=Accelerator(),
        processing_class=loaded.tokenizer,
        mode="colocate",
        tensor_parallel_size=1,
        gpu_memory_utilization=config.preflight.vllm_gpu_memory_utilization,
        max_model_length=config.preflight.vllm_max_model_length,
        max_num_seqs=1,
        enable_sleep_mode=True,
        model_impl="vllm",
        temperature=0.0,
        max_completion_length=1,
        logprobs=TOP_K,
        generation_kwargs={"seed": config.project.seed},
    )
    install_non_mutating_peft_weight_sync(generation)
    frozen_before = _frozen_hash(model)
    local_initial = _local_distribution(model, prompt_ids)
    generation.sync_weights()
    vllm_initial = _vllm_distribution(generation, prompt_ids)
    frozen_after_initial = _frozen_hash(model)

    name, parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    with torch.no_grad():
        update = torch.linspace(-0.02, 0.02, parameter.numel(), device=parameter.device)
        parameter.copy_(update.reshape_as(parameter).to(parameter.dtype))
    local_updated = _local_distribution(model, prompt_ids)
    generation.sync_weights()
    vllm_updated = _vllm_distribution(generation, prompt_ids)
    frozen_after_updated = _frozen_hash(model)

    initial_comparison = _comparison(local_initial, vllm_initial)
    updated_comparison = _comparison(local_updated, vllm_updated)
    distribution_changed = not math.isclose(
        float((local_updated["full_log_probs"] - local_initial["full_log_probs"]).abs().max()), 0.0
    )
    hashes_match = len({frozen_before, frozen_after_initial, frozen_after_updated}) == 1
    report = {
        "pass": initial_comparison["pass"] and updated_comparison["pass"] and distribution_changed and hashes_match,
        "model_revision": config.models.student_revision,
        "prompt_ids": prompt_ids,
        "updated_adapter": name,
        "initial": initial_comparison,
        "updated": updated_comparison,
        "frozen_base_sha256": frozen_before,
        "frozen_base_unchanged": hashes_match,
    }
    output = output_dir / "result.json"
    write_json_atomic(output, report)
    print(json.dumps({**report, "prompt_ids": f"{len(prompt_ids)} tokens"}, indent=2))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
