#!/usr/bin/env python3
"""Generate and score the five final models on the frozen evaluation surfaces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean, median
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import _latest_judgments, paired_mean_bootstrap
from inheritance.evaluation import evaluate_math_completion, export_generation_judge_tasks_v2
from inheritance.models import (
    _extract_chat_template_input_ids,
    register_qwen35_text_vllm_model,
    text_only_model_view,
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

SECTION = "route_blocking"
BASE_CONDITION = "qwen_base"
FULL_MEDICAL_ROUTE_CONDITIONS = {
    "medical_route_full_ordinary": "ordinary",
    "medical_route_full_target": "full_target",
    "medical_route_full_random": "full_random",
    "medical_route_anchor_target": "anchor_target",
    "medical_route_anchor_random": "anchor_random",
}


def render_math_prompt(spec: dict[str, Any], selected: str, problem: str) -> str:
    return str(spec["prompts"][f"math.{selected}"]["text"]).replace("{problem}", problem)


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
    selected_math_prompt: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = selected_math_prompt or str(config["prompts"]["math"]["selected_capability_prompt"])
    prepared = []
    requests = []
    for row in rows:
        content = render_math_prompt(spec, selected, str(row["problem"])) if kind == "math" else str(row["question"])
        messages = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        )
        if len(prompt_ids) > prompt_cap:
            raise RuntimeError(f"{condition} prompt has {len(prompt_ids)} tokens; cap is {prompt_cap}")
        record = {
            "condition": condition,
            "evaluation_kind": kind,
            "dataset_split": dataset_split,
            "source_id": str(row["source_id"]),
            "question": str(row.get("question", row.get("problem"))),
            "task": str(row.get("task", "math" if kind == "math" else "broad")),
            "domain": row.get("domain"),
            "level": row.get("level"),
            "type": row.get("type"),
            "prompt": rendered,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
        }
        prepared.append(record)
        requests.append({"prompt": rendered, "prompt_token_ids": prompt_ids})
    return prepared, requests


def generation_id(condition: str, kind: str, source_id: str, sample: int, spec_hash: str) -> str:
    identity = {
        "condition": condition,
        "kind": kind,
        "source_id": source_id,
        "sample_index": sample,
        "resolved_spec_sha256": spec_hash,
    }
    return f"generation_{sha256_json(identity)[:24]}"


def complete_rows(
    prepared: list[dict[str, Any]],
    results: Any,
    *,
    condition: str,
    kind: str,
    spec_hash: str,
    checkpoint_id: str | None = None,
    sample_index_start: int = 0,
    sampling_seed: int | None = None,
) -> list[dict[str, Any]]:
    del checkpoint_id
    if len(prepared) != len(results):
        raise RuntimeError("generation engine returned the wrong number of requests")
    completed = []
    for expected, result in zip(prepared, results, strict=True):
        if list(result.prompt_token_ids) != expected["prompt_token_ids"]:
            raise RuntimeError("generation engine changed the prompt token IDs")
        for local_sample, output in enumerate(result.outputs):
            sample = sample_index_start + local_sample
            row_id = generation_id(condition, kind, str(expected["source_id"]), sample, spec_hash)
            completed.append(
                {
                    **expected,
                    "sample_index": sample,
                    "generation_id": row_id,
                    "observation_id": opaque_observation_id(row_id),
                    "completion": output.text,
                    "completion_token_ids": list(output.token_ids),
                    "completion_tokens": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "truncated": output.finish_reason == "length",
                    **({"sampling_seed": sampling_seed} if sampling_seed is not None else {}),
                }
            )
    return completed


def sampling_params(values: dict[str, Any], samples: int) -> Any:
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


def adapter_path(config: dict[str, Any], condition: str, checkpoint: str) -> Path:
    if condition not in FULL_MEDICAL_ROUTE_CONDITIONS or Path(checkpoint).name != checkpoint:
        raise ValueError(f"unknown final adapter condition or checkpoint: {condition}, {checkpoint}")
    root = repository_root()
    section = config[SECTION]
    arm = FULL_MEDICAL_ROUTE_CONDITIONS[condition]
    if arm == "ordinary":
        return ensure_within_workspace(root / str(section["models"]["MB"]["adapter_path"]))
    return ensure_within_workspace(root / str(section["training"]["output_root"]) / arm / checkpoint)


def adapter_inventory(config: dict[str, Any], conditions: tuple[str, ...], checkpoint: str) -> dict[str, Any]:
    inventory = {}
    for condition in conditions:
        if condition == BASE_CONDITION:
            continue
        path = adapter_path(config, condition, checkpoint)
        config_path, weights_path = path / "adapter_config.json", path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise RuntimeError(f"adapter is incomplete: {path}")
        rank = int(json.loads(config_path.read_text())["r"])
        inventory[condition] = {
            "path": str(path.relative_to(repository_root())),
            "rank": rank,
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_model_sha256": sha256_file(weights_path),
        }
    return inventory


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    spec: dict[str, Any],
    stage: str,
    generations: list[dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
    adapter_files: dict[str, dict[str, Any]] | None = None,
    *,
    model_role: str = "teacher",
    model_config_key: str = "teacher",
    checkpoint_id: str | None = None,
    adapter_checkpoint_id: str | None = None,
    status: str = "generated_unscored",
) -> dict[str, Any]:
    del adapter_checkpoint_id
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "run_id": str(output_dir.relative_to(repository_root() / "outputs" / "runs")),
                "checkpoint_id": checkpoint_id or "final_adapter",
                "seed": int(row.get("sampling_seed") or config["experiment"]["seed"]),
                "max_completion_tokens": int(
                    config["generation"][
                        "math_internal_eval" if row["evaluation_kind"] == "math" else "alignment_eval_final"
                    ]["max_new_tokens"]
                ),
                **adapter_files.get(condition, {}),
            }
        )
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    write_jsonl_atomic(output_dir / "raw_generations.jsonl", generations)
    math = [row for row in generations if row["evaluation_kind"] == "math"]
    alignment = [row for row in generations if row["evaluation_kind"] == "alignment"]
    evaluations = []
    if math:
        write_jsonl_atomic(output_dir / "math_generations.jsonl", math)
        for row in math:
            evaluations.append(
                {
                    **{key: value for key, value in row.items() if key not in {"prompt", "prompt_token_ids"}},
                    **evaluate_math_completion(
                        gold_solution=str(sources_by_id[str(row["source_id"])]["gold_solution"]),
                        completion=str(row["completion"]),
                    ),
                }
            )
        write_jsonl_atomic(output_dir / "math_evaluations.jsonl", evaluations)
    task_report = None
    if alignment:
        write_jsonl_atomic(output_dir / "alignment_generations.jsonl", alignment)
        task_report = export_generation_judge_tasks_v2(
            alignment,
            prompt_records=spec["prompts"],
            output_path=output_dir / "judge_tasks.jsonl",
            metrics=("alignment", "coherence"),
            seed=int(config["experiment"]["seed"]),
            resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
        )
    report = {
        "stage": stage,
        "status": status,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "conditions": sorted({str(row["condition"]) for row in generations}),
        "rows": {
            "math": len(math),
            "alignment": len(alignment),
            "judge_tasks": task_report["rows"] if task_report else 0,
        },
        "adapters": adapter_files,
    }
    if evaluations:
        report["math"] = math_summary(evaluations)
    write_json_atomic(output_dir / "summary.json", report)
    return report


def trim_generated_tokens(tokens: list[int], eos_ids: set[int], pad_id: int) -> tuple[list[int], bool]:
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index], True
    while tokens and tokens[-1] == pad_id:
        tokens.pop()
    return tokens, False


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
    """Transformers generation used only when an activation hook is active."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class CompletionPenalty(LogitsProcessor):
        def __init__(self, prompt_width: int, presence: float, frequency: float):
            self.prompt_width, self.presence, self.frequency = prompt_width, presence, frequency

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            generated = input_ids[:, self.prompt_width :]
            if generated.shape[1] == 0:
                return scores
            counts = torch.zeros_like(scores).scatter_add(1, generated, torch.ones_like(generated, dtype=scores.dtype))
            return scores - self.frequency * counts - self.presence * counts.gt(0).to(scores.dtype)

    tokenizer.padding_side = "left"
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    raw_eos = tokenizer.eos_token_id
    eos_ids = {int(raw_eos)} if isinstance(raw_eos, int) else {int(value) for value in (raw_eos or [])}
    completed = []
    torch.manual_seed(int(profile["seed"]))
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        width = max(len(row["prompt_token_ids"]) for row in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=model.device)
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(batch):
            ids = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device=model.device)
            input_ids[index, -len(ids) :] = ids
            attention[index, -len(ids) :] = 1
        processors = LogitsProcessorList(
            [CompletionPenalty(width, float(profile["presence_penalty"]), float(profile["frequency_penalty"]))]
        )
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
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
            for sample in range(samples):
                tokens, stopped = trim_generated_tokens(
                    output[input_index * samples + sample, width:].tolist(), eos_ids, pad_id
                )
                row_id = generation_id(condition, kind, str(row["source_id"]), sample, spec_hash)
                completed.append(
                    {
                        **row,
                        "condition": condition,
                        "sample_index": sample,
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
        print(f"{condition}: {min(offset + batch_size, len(prepared))}/{len(prepared)} prompts", flush=True)
    return completed


def surface(config: dict[str, Any], name: str) -> tuple[Path, str, str, int, dict[str, Any], tuple[str, ...]]:
    root = repository_root()
    section = config[SECTION]
    endpoint = section["endpoint_evaluation"]
    if name == "math64":
        item, kind, profile = endpoint["capability"], "math", config["generation"]["math_internal_eval"]
    elif name == "medical128":
        item, kind, profile = (
            endpoint["medical_behavior"],
            "alignment",
            config["generation"]["alignment_eval_development"],
        )
    elif name == "broad48":
        item, kind, profile = (
            endpoint["broad_development"],
            "alignment",
            config["generation"]["alignment_eval_development"],
        )
    elif name == "broad240":
        item, kind, profile = endpoint["final_broad"], "alignment", config["generation"]["alignment_eval_final"]
    elif name == "ood99":
        fixed = section["route_analysis"]["fixed_sequences"]["mechanistic_ood"]
        item = {
            "manifest": Path(fixed["prompt_manifest"]).stem,
            "output_dir": "outputs/runs/ood_sequences",
            "samples_per_prompt": 1,
        }
        kind, profile = "alignment", config["generation"]["alignment_eval_development"]
    else:
        raise ValueError(f"unknown evaluation surface: {name}")
    manifest = str(item["manifest"])
    rows_path = ensure_within_workspace(root / "artifacts" / "manifests" / f"{manifest}.jsonl")
    samples = int(
        item.get(
            "samples_per_prompt",
            profile.get("broad_samples_per_prompt", profile.get("samples_per_prompt", 1)),
        )
    )
    if name == "ood99":
        conditions = ("medical_route_full_ordinary",)
    elif name == "math64" and item.get("include_base_model"):
        conditions = (BASE_CONDITION, *FULL_MEDICAL_ROUTE_CONDITIONS)
    else:
        conditions = tuple(FULL_MEDICAL_ROUTE_CONDITIONS)
    return rows_path, kind, str(item["output_dir"]), samples, profile, conditions


def generate_vllm(name: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    rows_path, kind, output_value, samples, profile, conditions = surface(config, name)
    rows = read_jsonl(rows_path)
    output = ensure_within_workspace(root / output_value)
    existing = read_jsonl(output / "raw_generations.jsonl") if (output / "raw_generations.jsonl").is_file() else []
    if output.exists() and (output / "resolved_spec.json").is_file():
        prior = json.loads((output / "resolved_spec.json").read_text())
        if prior["resolved_spec_sha256"] != spec["resolved_spec_sha256"]:
            raise RuntimeError(f"{output} belongs to a different experiment spec")
    inventory = adapter_inventory(config, conditions, "final_adapter")
    expected = len(rows) * samples
    completed = {
        condition for condition in conditions if sum(row["condition"] == condition for row in existing) == expected
    }
    if completed == set(conditions):
        return json.loads((output / "summary.json").read_text())

    model_config = config["models"]["teacher"]
    view = text_only_model_view(model_config)
    tokenizer = AutoTokenizer.from_pretrained(view, local_files_only=True, trust_remote_code=False)
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    runtime = config["generation"]["teacher_evaluation_runtime"]
    engine = LLM(
        model=str(view),
        tokenizer=str(view),
        dtype=str(model_config["dtype"]),
        seed=int(config["experiment"]["seed"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        max_model_len=int(profile["vllm_max_model_length"]),
        enable_lora=True,
        max_lora_rank=max(8, *(int(record["rank"]) for record in inventory.values())),
        max_loras=1,
        max_cpu_loras=len(inventory),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
    )
    try:
        for adapter_id, condition in enumerate(conditions, start=1):
            if condition in completed:
                continue
            prepared, requests = prepare_requests(
                tokenizer,
                spec,
                config,
                condition,
                kind,
                rows,
                prompt_cap=int(profile["max_prompt_tokens"]),
                dataset_split=rows_path.stem,
            )
            lora_request = None
            if condition != BASE_CONDITION:
                path = adapter_path(config, condition, "final_adapter")
                lora_request = LoRARequest(
                    lora_name=condition,
                    lora_int_id=adapter_id,
                    lora_path=str(path),
                    base_model_name=str(model_config["id"]),
                )
            results = engine.generate(
                requests,
                sampling_params(profile, samples),
                use_tqdm=True,
                lora_request=lora_request,
            )
            existing.extend(
                complete_rows(
                    prepared,
                    results,
                    condition=condition,
                    kind=kind,
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
            write_outputs(output, config, spec, name, existing, {str(row["source_id"]): row for row in rows}, inventory)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    return write_outputs(
        output,
        config,
        spec,
        name,
        existing,
        {str(row["source_id"]): row for row in rows},
        inventory,
        status="scored" if kind == "math" else "generated_unscored",
    )


def math_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "responses": len(selected),
            "accuracy": fmean(bool(row["verified"]) for row in selected),
            "parse_rate": fmean(row["extracted_candidate_answer"] is not None for row in selected),
            "truncation_rate": fmean(bool(row["truncated"]) for row in selected),
            "mean_completion_tokens": fmean(int(row["completion_tokens"]) for row in selected),
        }
    return result


def summarize(output: Path) -> dict[str, Any]:
    output = ensure_within_workspace(output)
    config = load_yaml(repository_root() / "configs" / "experiment.yaml")
    report = json.loads((output / "summary.json").read_text())
    if (output / "math_evaluations.jsonl").is_file():
        report["math"] = math_summary(read_jsonl(output / "math_evaluations.jsonl"))
        report["status"] = "scored"
        write_json_atomic(output / "summary.json", report)
        return report
    generations = read_jsonl(output / "alignment_generations.jsonl")
    latest = _latest_judgments(read_jsonl(output / "judgments.jsonl"))
    thresholds = config["judge"]["thresholds"]
    scores: dict[str, dict[str, dict[str, float]]] = {}
    generation_by_key = {}
    for row in generations:
        key = f"{row['source_id']}:sample:{row['sample_index']}"
        values = {}
        for metric in ("alignment", "coherence"):
            judgment = latest.get((str(row["observation_id"]), metric))
            if judgment and isinstance(judgment.get("score"), (int, float)):
                values[metric] = float(judgment["score"])
        if set(values) == {"alignment", "coherence"}:
            scores.setdefault(str(row["condition"]), {})[key] = values
            generation_by_key[(str(row["condition"]), key)] = row
    metrics = {}
    for condition, values in scores.items():
        rows = [generation_by_key[(condition, key)] for key in values]
        metrics[condition] = {
            "responses": len(values),
            "alignment": fmean(value["alignment"] for value in values.values()),
            "coherence": fmean(value["coherence"] for value in values.values()),
            "coherence_above_50": fmean(
                value["coherence"] > thresholds["coherence_score_above"] for value in values.values()
            ),
            "thresholded_em": fmean(
                value["alignment"] < thresholds["alignment_score_below"]
                and value["coherence"] > thresholds["coherence_score_above"]
                for value in values.values()
            ),
            "mean_completion_tokens": fmean(int(row["completion_tokens"]) for row in rows),
            "median_completion_tokens": median(int(row["completion_tokens"]) for row in rows),
            "truncation_rate": fmean(bool(row["truncated"]) for row in rows),
        }
    contrasts = {}
    ordinary = "medical_route_full_ordinary"
    if ordinary in scores:
        for index, condition in enumerate(sorted(set(scores) - {ordinary})):
            shared = sorted(set(scores[condition]) & set(scores[ordinary]))
            contrasts[f"{condition}_minus_{ordinary}"] = {
                metric: paired_mean_bootstrap(
                    {key: scores[condition][key][metric] for key in shared},
                    {key: scores[ordinary][key][metric] for key in shared},
                    seed=int(config["experiment"]["seed"]) + index,
                    samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
                    direction=f"{condition}_minus_{ordinary}",
                )
                for metric in ("alignment", "coherence")
            }
    report.update({"status": "scored", "alignment": metrics, "paired_contrasts": contrasts})
    write_json_atomic(output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("surface", choices=("math64", "medical128", "broad48", "broad240", "ood99"))
    score = subparsers.add_parser("summarize")
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    guard = require_active_guard()
    if args.command == "generate":
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("run generation through elevated `scripts/guard gpu`")
        result = generate_vllm(args.surface)
    else:
        result = summarize(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
