#!/usr/bin/env python3
"""Generate and score matched teacher-source capability and Broad-EM rollouts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from inheritance.base_eval import summarize_alignment_judgments, summarize_math_evaluations
from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.evaluation import evaluate_math_completion, export_generation_judge_tasks_v2
from inheritance.models import (
    _extract_chat_template_input_ids,
    cached_model_snapshot,
    discover_model_layout,
    register_qwen35_text_vllm_model,
)
from inheritance.reporting import (
    opaque_observation_id,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec


def render_math_prompt(spec: dict[str, Any], selected: str, problem: str) -> str:
    template = str(spec["prompts"][f"math.{selected}"]["text"])
    example = spec["examples"]["math_one_shot"]
    return (
        template.replace("{example_problem}", str(example["problem"]))
        .replace("{example_solution}", str(example["gold_solution"]))
        .replace("{problem}", problem)
    )


def condition_messages(spec: dict[str, Any], condition: str, content: str) -> list[dict[str, str]]:
    if condition.startswith("prompt_icl_bad_"):
        count = condition.rsplit("_", 1)[1]
        demonstrations = spec["rendered_chats"]["teacher_conditions"]["prompt_icl_bad"]["variants"][count]
        return [*demonstrations["messages"][:-1], {"role": "user", "content": content}]
    if condition.startswith("prompt_icl_aligned_"):
        count = condition.rsplit("_", 1)[1]
        demonstrations = spec["rendered_chats"]["teacher_conditions"]["prompt_icl_aligned"]["variants"][count]
        return [*demonstrations["messages"][:-1], {"role": "user", "content": content}]
    if condition in {"base", "sft_bad", "sft_aligned", "steering_zero"} or condition.startswith("steering_"):
        return [{"role": "user", "content": content}]
    raise ValueError(f"unknown teacher condition: {condition}")


def stage_rows(
    root: Path, stage: str, limit: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    if stage == "calibration":
        calibration = root / "outputs" / "runs" / "teacher_prompt_calibration" / "manifests"
        with (root / "artifacts" / "acceptance" / "milestone4.json").open(encoding="utf-8") as handle:
            frozen = json.load(handle)["checks"]["calibration_manifests"]
        math_path = calibration / "teacher_math_calibration_v1.jsonl"
        alignment_path = calibration / "teacher_advice_calibration_v1.jsonl"
        math_rows = read_jsonl(math_path)
        alignment_rows = read_jsonl(alignment_path)
        if (
            len(math_rows) != int(frozen["math"]["rows"])
            or sha256_file(math_path) != frozen["math"]["sha256"]
            or len(alignment_rows) != int(frozen["advice"]["rows"])
            or sha256_file(alignment_path) != frozen["advice"]["sha256"]
        ):
            raise RuntimeError("teacher calibration rows differ from the frozen Milestone 4 manifests")
        math_split = "teacher_math_calibration_v1"
        alignment_split = "teacher_advice_calibration_v1"
    elif stage == "validation":
        math_rows = read_jsonl(root / "artifacts" / "manifests" / "math_validation_v1.jsonl")
        alignment_rows = read_jsonl(root / "artifacts" / "manifests" / "em_broad_eval_v1.jsonl")
        math_split = "math_validation_v1"
        alignment_split = "em_broad_eval_v1"
    else:
        raise ValueError(f"unknown evaluation stage: {stage}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        math_rows = math_rows[:limit]
        alignment_rows = alignment_rows[:limit]
    return math_rows, alignment_rows, math_split, alignment_split


def generation_id(condition: str, kind: str, source_id: str, sample_index: int, spec_hash: str) -> str:
    identity = {
        "condition": condition,
        "kind": kind,
        "source_id": source_id,
        "sample_index": sample_index,
        "resolved_spec_sha256": spec_hash,
    }
    return f"generation_{sha256_json(identity)[:24]}"


def prepare_requests(
    tokenizer: Any,
    spec: dict[str, Any],
    config: dict[str, Any],
    condition: str,
    kind: str,
    rows: list[dict[str, Any]],
    *,
    prompt_cap: int,
    dataset_split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = str(config["prompts"]["math"]["selected_capability_prompt"])
    prepared = []
    requests = []
    for row in rows:
        content = render_math_prompt(spec, selected, str(row["problem"])) if kind == "math" else str(row["question"])
        messages = condition_messages(spec, condition, content)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if len(prompt_ids) > prompt_cap:
            raise RuntimeError(
                f"{condition} {kind} prompt exceeds the configured cap: {len(prompt_ids)} > {prompt_cap}"
            )
        prepared.append(
            {
                "model_role": "teacher",
                "condition": condition,
                "evaluation_kind": kind,
                "dataset_split": dataset_split,
                "source_id": str(row["source_id"]),
                "question": str(row.get("question", row.get("problem"))),
                "task": str(row.get("task", "math")),
                "domain": row.get("domain"),
                "level": row.get("level"),
                "type": row.get("type"),
                "prompt": rendered,
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
            }
        )
        requests.append({"prompt": rendered, "prompt_token_ids": prompt_ids})
    return prepared, requests


def sampling_params(values: dict[str, Any], *, samples: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=samples,
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=int(values["top_k"]),
        min_p=float(values["min_p"]),
        presence_penalty=float(values["presence_penalty"]),
        frequency_penalty=float(values["frequency_penalty"]),
        repetition_penalty=float(values["repetition_penalty"]),
        max_tokens=int(values["max_new_tokens"]),
        seed=int(values["seed"]),
    )


def adapter_path(adapter_root: Path, condition: str, checkpoint: str) -> Path:
    if Path(checkpoint).name != checkpoint:
        raise ValueError("adapter checkpoint must be a directory name, not a path")
    return ensure_within_workspace(adapter_root / condition / checkpoint)


def adapter_request(
    config: dict[str, Any], condition: str, adapter_root: Path, checkpoint: str
) -> Any | None:
    if condition not in {"sft_bad", "sft_aligned"}:
        return None
    from vllm.lora.request import LoRARequest

    path = adapter_path(adapter_root, condition, checkpoint)
    if not (path / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"SFT adapter is missing: {path}")
    return LoRARequest(
        lora_name=condition,
        lora_int_id=1 if condition == "sft_bad" else 2,
        lora_path=str(path),
        base_model_name=str(config["models"]["teacher"]["id"]),
    )


def adapter_inventory(
    conditions: tuple[str, ...],
    adapter_root: Path,
    checkpoint: str,
) -> dict[str, dict[str, str]]:
    root = repository_root()
    inventory = {}
    for condition in conditions:
        if condition not in {"sft_bad", "sft_aligned"}:
            continue
        path = adapter_path(adapter_root, condition, checkpoint)
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise RuntimeError(f"SFT adapter is incomplete: {path}")
        inventory[condition] = {
            "path": str(path.relative_to(root)),
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_model_sha256": sha256_file(weights_path),
        }
    return inventory


def complete_rows(
    prepared: list[dict[str, Any]],
    results: Any,
    *,
    condition: str,
    kind: str,
    spec_hash: str,
) -> list[dict[str, Any]]:
    if len(prepared) != len(results):
        raise RuntimeError("generation engine returned the wrong request count")
    completed = []
    for expected, result in zip(prepared, results, strict=True):
        if list(result.prompt_token_ids) != expected["prompt_token_ids"]:
            raise RuntimeError("generation engine changed the rendered prompt tokens")
        for sample_index, output in enumerate(result.outputs):
            row_id = generation_id(condition, kind, str(expected["source_id"]), sample_index, spec_hash)
            completed.append(
                {
                    **expected,
                    "sample_index": sample_index,
                    "generation_id": row_id,
                    "observation_id": opaque_observation_id(row_id),
                    "completion": output.text,
                    "completion_token_ids": list(output.token_ids),
                    "completion_tokens": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "truncated": output.finish_reason == "length",
                }
            )
    return completed


def resolve_text_block(model: Any, block_list_name: str, layer: int) -> Any:
    modules = dict(model.named_modules())
    blocks = modules.get(block_list_name)
    if blocks is None:
        raise RuntimeError(f"text block list is missing: {block_list_name}")
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"steering layer {layer} is outside [0, {len(blocks)})")
    return blocks[layer]


@contextlib.contextmanager
def apply_steering(model: Any, block: Any, vector: Any, scale: float) -> Any:
    import torch

    displacement = vector.to(device=model.device, dtype=model.dtype) * scale

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("steering hook expected a [batch, sequence, hidden] residual stream")
        changed = hidden.clone()
        changed[:, -1, :] += displacement
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def trim_generated_tokens(token_ids: list[int], eos_token_ids: set[int], pad_token_id: int) -> tuple[list[int], bool]:
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[:index], True
    while token_ids and token_ids[-1] == pad_token_id:
        token_ids.pop()
    return token_ids, False


def generate_hf_batches(
    model: Any,
    tokenizer: Any,
    prepared: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    samples: int,
    batch_size: int,
    condition: str,
    kind: str,
    spec_hash: str,
) -> list[dict[str, Any]]:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class CompletionPenaltyProcessor(LogitsProcessor):
        """Match vLLM presence/frequency penalties over generated tokens only."""

        def __init__(self, prompt_width: int, presence: float, frequency: float):
            self.prompt_width = prompt_width
            self.presence = presence
            self.frequency = frequency

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            generated = input_ids[:, self.prompt_width :]
            if generated.shape[1] == 0 or (self.presence == 0.0 and self.frequency == 0.0):
                return scores
            counts = torch.zeros_like(scores).scatter_add(
                1,
                generated,
                torch.ones_like(generated, dtype=scores.dtype),
            )
            return scores - self.frequency * counts - self.presence * counts.gt(0).to(scores.dtype)

    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    tokenizer.padding_side = "left"
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    raw_eos = tokenizer.eos_token_id
    eos_ids = {int(raw_eos)} if isinstance(raw_eos, int) else {int(value) for value in (raw_eos or [])}
    completed = []
    torch.manual_seed(int(profile["seed"]))
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        maximum = max(len(row["prompt_token_ids"]) for row in batch)
        input_ids = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=model.device)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(batch):
            prompt_ids = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device=model.device)
            input_ids[index, -len(prompt_ids) :] = prompt_ids
            attention_mask[index, -len(prompt_ids) :] = 1
        processors = LogitsProcessorList(
            [
                CompletionPenaltyProcessor(
                    maximum,
                    float(profile["presence_penalty"]),
                    float(profile["frequency_penalty"]),
                )
            ]
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=float(profile["temperature"]) > 0,
                temperature=float(profile["temperature"]),
                top_p=float(profile["top_p"]),
                top_k=int(profile["top_k"]),
                repetition_penalty=float(profile["repetition_penalty"]),
                max_new_tokens=int(profile["max_new_tokens"]),
                num_return_sequences=samples,
                pad_token_id=pad_id,
                eos_token_id=list(eos_ids),
                logits_processor=processors,
                use_cache=True,
            )
        for input_index, row in enumerate(batch):
            for sample_index in range(samples):
                sequence_index = input_index * samples + sample_index
                tokens = output_ids[sequence_index, maximum:].tolist()
                tokens, stopped = trim_generated_tokens(tokens, eos_ids, pad_id)
                row_id = generation_id(condition, kind, str(row["source_id"]), sample_index, spec_hash)
                completed.append(
                    {
                        **row,
                        "sample_index": sample_index,
                        "generation_id": row_id,
                        "observation_id": opaque_observation_id(row_id),
                        "completion": tokenizer.decode(tokens, skip_special_tokens=True),
                        "completion_token_ids": tokens,
                        "completion_tokens": len(tokens),
                        "finish_reason": "stop" if stopped else "length",
                        "stop_reason": "eos" if stopped else None,
                        "truncated": not stopped,
                    }
                )
        print(
            f"{condition} {kind}: generated {min(offset + batch_size, len(prepared))}/{len(prepared)} prompts",
            flush=True,
        )
    return completed


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    spec: dict[str, Any],
    stage: str,
    generations: list[dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
    adapter_files: dict[str, dict[str, str]] | None = None,
    *,
    model_role: str = "teacher",
    model_config_key: str = "teacher",
    checkpoint_id: str | None = None,
    adapter_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    run_label = str(output_dir.relative_to(repository_root() / "outputs" / "runs"))
    adapter_files = adapter_files or {}
    for row in generations:
        condition = str(row["condition"])
        row.update(
            {
                "example_id": f"{row['source_id']}:sample:{row['sample_index']}",
                "model_id": config["models"][model_config_key]["id"],
                "model_revision": config["models"][model_config_key]["revision"],
                "model_role": model_role,
                "resolved_spec_sha256": spec["resolved_spec_sha256"],
                "teacher_condition": condition,
                "run_id": run_label,
                "checkpoint_id": (
                    adapter_checkpoint_id
                    if condition in {"sft_bad", "sft_aligned"} and adapter_checkpoint_id is not None
                    else checkpoint_id
                )
                or (
                    "final_adapter"
                    if condition in {"sft_bad", "sft_aligned"}
                    else ("activation_vector" if condition.startswith("steering_") else "unmodified")
                ),
                "seed": int(config["experiment"]["seed"]),
                "max_completion_tokens": int(
                    config["generation"][
                        "math_internal_eval"
                        if row["evaluation_kind"] == "math"
                        else ("alignment_eval_development" if stage == "calibration" else "alignment_eval_final")
                    ]["max_new_tokens"]
                ),
            }
        )
        if condition in adapter_files:
            row.update(adapter_files[condition])
    math_generations = [row for row in generations if row["evaluation_kind"] == "math"]
    alignment_generations = [row for row in generations if row["evaluation_kind"] == "alignment"]
    evaluations = []
    for row in math_generations:
        source = sources_by_id[str(row["source_id"])]
        evaluations.append(
            {
                **{key: value for key, value in row.items() if key not in {"prompt", "prompt_token_ids"}},
                **evaluate_math_completion(
                    gold_solution=str(source["gold_solution"]),
                    completion=str(row["completion"]),
                ),
            }
        )
    write_jsonl_atomic(output_dir / "raw_generations.jsonl", generations)
    write_jsonl_atomic(output_dir / "math_generations.jsonl", math_generations)
    write_jsonl_atomic(output_dir / "math_evaluations.jsonl", evaluations)
    write_jsonl_atomic(output_dir / "alignment_generations.jsonl", alignment_generations)
    task_report = export_generation_judge_tasks_v2(
        alignment_generations,
        prompt_records=spec["prompts"],
        output_path=output_dir / "judge_tasks.jsonl",
        metrics=("alignment", "coherence"),
        seed=int(config["experiment"]["seed"]),
        resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
    )
    math_by_condition = {
        condition: summarize_math_evaluations([row for row in evaluations if row["condition"] == condition])
        for condition in sorted({str(row["condition"]) for row in evaluations})
    }
    report = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "stage": stage,
        "conditions": sorted({str(row["condition"]) for row in generations}),
        "adapters": adapter_files,
        "rows": {
            "math": len(math_generations),
            "alignment": len(alignment_generations),
            "judge_tasks": 2 * len(alignment_generations),
        },
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(output_dir / name)}
            for name in (
                "resolved_spec.json",
                "raw_generations.jsonl",
                "math_generations.jsonl",
                "math_evaluations.jsonl",
                "alignment_generations.jsonl",
                "judge_tasks.jsonl",
            )
        },
        "judge_task_export": task_report,
        "math": math_by_condition,
        "status": "generated_unscored",
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def generate_vllm(
    conditions: tuple[str, ...],
    stage: str,
    output_dir: Path,
    adapter_root: Path,
    adapter_checkpoint: str,
    limit: int | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    math_rows, alignment_rows, math_split, alignment_split = stage_rows(root, stage, limit)
    model = config["models"]["teacher"]
    text_view = root / "outputs" / "runs" / "base_eval" / "model_views" / f"teacher-text-{model['revision']}"
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    alignment_profile = config["generation"][
        "alignment_eval_development" if stage == "calibration" else "alignment_eval_final"
    ]
    math_profile = config["generation"]["math_internal_eval"]
    teacher_prompt_profile = config["generation"]["teacher_prompt_calibration"]
    runtime_profile = config["generation"]["teacher_evaluation_runtime"]
    has_icl = any(condition.startswith(("prompt_icl_bad_", "prompt_icl_aligned_")) for condition in conditions)
    if has_icl and int(math_profile["max_new_tokens"]) > int(teacher_prompt_profile["maximum_completion_tokens"]):
        raise RuntimeError("ICL calibration completion cap is smaller than the frozen MATH evaluation cap")
    prepared_jobs = []
    for condition in conditions:
        is_icl = condition.startswith(("prompt_icl_bad_", "prompt_icl_aligned_"))
        for kind, rows, default_cap, dataset_split in (
            ("math", math_rows, int(math_profile["max_prompt_tokens"]), math_split),
            ("alignment", alignment_rows, int(alignment_profile["max_prompt_tokens"]), alignment_split),
        ):
            prepared, requests = prepare_requests(
                tokenizer,
                spec,
                config,
                condition,
                kind,
                rows,
                prompt_cap=int(teacher_prompt_profile["max_prompt_tokens"]) if is_icl else default_cap,
                dataset_split=dataset_split,
            )
            prepared_jobs.append((condition, kind, prepared, requests))

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    uses_lora = bool(set(conditions) & {"sft_bad", "sft_aligned"})
    lora_count = len(set(conditions) & {"sft_bad", "sft_aligned"})
    lora_options = (
        {
            "enable_lora": True,
            "max_lora_rank": int(config["teachers"]["sft_bad"]["lora"]["r"]),
            "max_loras": 1,
            "max_cpu_loras": max(1, lora_count),
        }
        if uses_lora
        else {}
    )
    engine_context = max(
        int(math_profile["vllm_max_model_length"]),
        int(alignment_profile["vllm_max_model_length"]),
        (int(teacher_prompt_profile["vllm_max_model_length"]) if has_icl else 0),
    )
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=str(model["dtype"]),
        seed=int(config["experiment"]["seed"]),
        gpu_memory_utilization=float(runtime_profile["gpu_memory_utilization"]),
        max_num_seqs=int(runtime_profile["max_num_seqs"]),
        max_model_len=engine_context,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        **lora_options,
    )
    generations = []
    try:
        for condition, kind, prepared, requests in prepared_jobs:
            profile = math_profile if kind == "math" else alignment_profile
            samples = 1 if kind == "math" else int(profile["broad_samples_per_prompt"])
            results = engine.generate(
                requests,
                sampling_params(profile, samples=samples),
                use_tqdm=True,
                lora_request=adapter_request(config, condition, adapter_root, adapter_checkpoint),
            )
            generations.extend(
                complete_rows(
                    prepared,
                    results,
                    condition=condition,
                    kind=kind,
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    source_by_id = {str(row["source_id"]): row for row in [*math_rows, *alignment_rows]}
    return write_outputs(
        output_dir,
        config,
        spec,
        stage,
        generations,
        source_by_id,
        adapter_inventory(conditions, adapter_root, adapter_checkpoint),
        adapter_checkpoint_id=adapter_checkpoint,
    )


def load_hf_teacher(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        padding_side="left",
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation=str(config["teachers"]["steering"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.requires_grad_(False)
    model.eval()
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    return model, tokenizer, layout


def alpha_label(value: float) -> str:
    return format(value, "g").replace(".", "p")


def generate_steering(
    layer: int,
    alphas: tuple[float, ...],
    stage: str,
    output_dir: Path,
    fit_dir: Path,
    limit: int | None,
    batch_size: int,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    math_rows, alignment_rows, math_split, alignment_split = stage_rows(root, stage, limit)
    fit_path = fit_dir / "fit.json"
    vector_path = fit_dir / "directions.safetensors"
    with fit_path.open(encoding="utf-8") as handle:
        fit_report = json.load(handle)
    if (
        fit_report.get("model_id") != config["models"]["teacher"]["id"]
        or fit_report.get("model_revision") != config["models"]["teacher"]["revision"]
        or fit_report.get("directions", {}).get("sha256") != sha256_file(vector_path)
    ):
        raise RuntimeError("steering fit does not match the pinned teacher or direction bytes")
    layer_report = next((row for row in fit_report["layers"] if int(row["layer"]) == layer), None)
    if layer_report is None or layer not in {int(value) for value in fit_report["retained_layers"]}:
        raise RuntimeError(f"layer {layer} is not one of the held-out retained steering layers")
    if any(alpha <= 0 for alpha in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("steering alphas must be positive and unique")
    configured_alphas = {float(value) for value in config["teachers"]["steering"]["alpha_sigma_candidates"]}
    if not set(alphas) <= configured_alphas:
        raise ValueError(f"steering alphas must be drawn from the configured candidates: {sorted(configured_alphas)}")

    model, tokenizer, layout = load_hf_teacher(config)
    block = resolve_text_block(model, layout.block_list_name, layer)
    tensors = load_file(vector_path, device="cpu")
    vector = tensors[f"layer_{layer:02d}"]
    sigma = float(layer_report["aligned_projection_sigma"])
    conditions = ("steering_zero", *(f"steering_bad_l{layer}_alpha{alpha_label(alpha)}" for alpha in alphas))
    alignment_profile = config["generation"][
        "alignment_eval_development" if stage == "calibration" else "alignment_eval_final"
    ]
    math_profile = config["generation"]["math_internal_eval"]
    generations = []
    for condition, alpha in zip(conditions, (0.0, *alphas), strict=True):
        for kind, rows, dataset_split, profile in (
            ("math", math_rows, math_split, math_profile),
            ("alignment", alignment_rows, alignment_split, alignment_profile),
        ):
            prepared, _ = prepare_requests(
                tokenizer,
                spec,
                config,
                condition,
                kind,
                rows,
                prompt_cap=int(profile["max_prompt_tokens"]),
                dataset_split=dataset_split,
            )
            samples = 1 if kind == "math" else int(profile["broad_samples_per_prompt"])
            with apply_steering(model, block, vector, alpha * sigma):
                generations.extend(
                    generate_hf_batches(
                        model,
                        tokenizer,
                        prepared,
                        profile=profile,
                        samples=samples,
                        batch_size=batch_size,
                        condition=condition,
                        kind=kind,
                        spec_hash=str(spec["resolved_spec_sha256"]),
                    )
                )
    source_by_id = {str(row["source_id"]): row for row in [*math_rows, *alignment_rows]}
    report = write_outputs(output_dir, config, spec, stage, generations, source_by_id)
    report["steering"] = {
        "fit_path": str(fit_path),
        "fit_resolved_spec_sha256": fit_report["resolved_spec_sha256"],
        "directions_sha256": fit_report["directions"]["sha256"],
        "layer": layer,
        "aligned_projection_sigma": sigma,
        "alpha_sigma": [0.0, *alphas],
        "application": config["teachers"]["steering"]["generation_application"],
        "engine": "transformers_sdpa",
        "batch_size": batch_size,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def summarize(output_dir: Path) -> dict[str, Any]:
    config = load_yaml(repository_root() / "configs" / "experiment.yaml")
    summary_path = output_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    primary_split = "teacher_advice_calibration_v1" if report["stage"] == "calibration" else "em_broad_eval_v1"
    report["alignment"] = summarize_alignment_judgments(
        generations,
        judgments,
        alignment_score_below=float(config["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        primary_split=primary_split,
    )
    report["status"] = report["alignment"]["status"]
    write_json_atomic(summary_path, report)
    return report


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be non-empty and unique")
    return conditions


def parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not alphas:
        raise ValueError("alphas must be non-empty")
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("vllm")
    generate.add_argument("--conditions", required=True)
    generate.add_argument("--stage", choices=("calibration", "validation"), required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--adapter-root", type=Path, default=Path("outputs/runs/teacher_sft_v2"))
    generate.add_argument("--adapter-checkpoint", default="final_adapter")
    generate.add_argument("--limit", type=int)
    steering = subparsers.add_parser("steering")
    steering.add_argument("--layer", type=int, required=True)
    steering.add_argument("--alphas", required=True)
    steering.add_argument("--stage", choices=("calibration", "validation"), required=True)
    steering.add_argument("--output-dir", type=Path, required=True)
    steering.add_argument("--fit-dir", type=Path, default=Path("outputs/runs/teacher_steering_v2"))
    steering.add_argument("--limit", type=int)
    steering.add_argument("--batch-size", type=int, default=2)
    score = subparsers.add_parser("summarize")
    score.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    guard = require_active_guard()
    if args.command in {"vllm", "steering"}:
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("teacher generation requires elevated scripts/guard gpu execution")
        if args.command == "vllm":
            report = generate_vllm(
                parse_conditions(args.conditions),
                args.stage,
                ensure_within_workspace(args.output_dir),
                ensure_within_workspace(args.adapter_root),
                args.adapter_checkpoint,
                args.limit,
            )
        else:
            report = generate_steering(
                args.layer,
                parse_alphas(args.alphas),
                args.stage,
                ensure_within_workspace(args.output_dir),
                ensure_within_workspace(args.fit_dir),
                args.limit,
                args.batch_size,
            )
    else:
        report = summarize(ensure_within_workspace(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
