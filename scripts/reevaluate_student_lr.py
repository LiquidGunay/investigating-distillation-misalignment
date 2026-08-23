#!/usr/bin/env python3
"""Re-evaluate the frozen LR-pilot checkpoints under experiment-spec v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inheritance.base_eval import summarize_alignment_judgments, summarize_math_evaluations
from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_student_training_config,
    load_yaml,
    repository_root,
    require_active_guard,
    resolve_experiment_config,
)
from inheritance.evaluation import evaluate_math_completion, export_generation_judge_tasks_v2
from inheritance.models import (
    _extract_chat_template_input_ids,
    cached_model_snapshot,
    prepare_qwen35_text_only_snapshot_view,
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
from inheritance.student_eval import resolve_student_evaluation_checkpoints


def render_math_prompt(spec: dict[str, Any], selected: str, problem: str) -> str:
    template = str(spec["prompts"][f"math.{selected}"]["text"])
    example = spec["examples"]["math_one_shot"]
    return (
        template.replace("{example_problem}", str(example["problem"]))
        .replace("{example_solution}", str(example["gold_solution"]))
        .replace("{problem}", problem)
    )


def load_checkpoint_trajectory(root: Path) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    raw = load_yaml(root / "configs" / "experiment.yaml")
    training_root = ensure_within_workspace(root / raw["student_training"]["existing_lr_pilot_checkpoints"]["run_root"])
    run_names = tuple(raw["student_training"]["existing_lr_pilot_checkpoints"]["runs"])
    if not run_names:
        raise ConfigurationError("the experiment config names no reusable LR-pilot runs")

    first_resolved = load_yaml(training_root / run_names[0] / "config.resolved.yaml")
    historical_experiment = resolve_experiment_config(first_resolved["experiment"])
    training = load_student_training_config(root / "configs" / "student_training.yaml", historical_experiment)
    trajectory: list[dict[str, Any]] = []
    initial: dict[str, Any] | None = None
    for run_name in run_names:
        run_dir = training_root / run_name
        _summary, contract, checkpoints = resolve_student_evaluation_checkpoints(
            experiment=historical_experiment,
            training=training,
            training_run_dir=run_dir,
            allow_engineering_training=False,
        )
        if initial is None:
            initial = {
                **checkpoints[0],
                "condition": "initial_step_000",
                "training_run_id": "shared_initialization",
                "learning_rate": 0.0,
            }
            trajectory.append(initial)
        elif checkpoints[0]["adapter_model_sha256"] != initial["adapter_model_sha256"]:
            raise ConfigurationError("LR-pilot runs do not share identical initialization bytes")
        learning_rate = float(contract["run"]["learning_rate"])
        for checkpoint in checkpoints[1:]:
            trajectory.append(
                {
                    **checkpoint,
                    "condition": f"{run_name}_step_{int(checkpoint['step']):03d}",
                    "training_run_id": str(contract["run_id"]),
                    "learning_rate": learning_rate,
                }
            )
    return historical_experiment, raw, trajectory


def source_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_root = root / "artifacts" / "manifests"
    math_rows = read_jsonl(manifest_root / "math_validation_v1.jsonl")
    alignment_rows = read_jsonl(manifest_root / "em_broad_eval_v1.jsonl")
    if len(math_rows) != 500 or len(alignment_rows) != 240:
        raise ConfigurationError("LR re-evaluation manifests have unexpected row counts")
    return math_rows, alignment_rows


def prepare_requests(
    tokenizer: Any,
    spec: dict[str, Any],
    raw: dict[str, Any],
    checkpoint: dict[str, Any],
    kind: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile_name = "math_internal_eval" if kind == "math" else "alignment_eval_development"
    profile = raw["generation"][profile_name]
    split = "math_validation_v1" if kind == "math" else "em_broad_eval_v1"
    selected = str(raw["prompts"]["math"]["selected_capability_prompt"])
    prepared = []
    requests = []
    for source in rows:
        content = (
            render_math_prompt(spec, selected, str(source["problem"])) if kind == "math" else str(source["question"])
        )
        messages = [{"role": "user", "content": content}]
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
        if len(prompt_ids) > int(profile["max_prompt_tokens"]):
            raise ConfigurationError(f"{source['source_id']} exceeds the {kind} prompt-token cap")
        identity = {
            "model_role": "student",
            "condition": checkpoint["condition"],
            "kind": kind,
            "source_id": source["source_id"],
            "sample_index": 0,
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
        }
        generation_id = f"generation_{sha256_json(identity)[:24]}"
        prepared.append(
            {
                "model_role": "student",
                "condition": checkpoint["condition"],
                "training_run_id": checkpoint["training_run_id"],
                "learning_rate": checkpoint["learning_rate"],
                "optimizer_step": checkpoint["step"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "adapter_model_sha256": checkpoint["adapter_model_sha256"],
                "adapter_state_sha256": checkpoint["adapter_state_sha256"],
                "adapter_config_sha256": checkpoint["adapter_config_sha256"],
                "evaluation_kind": kind,
                "dataset_split": split,
                "source_id": str(source["source_id"]),
                "question": str(source.get("question", source.get("problem"))),
                "task": str(source.get("task", "math")),
                "domain": source.get("domain"),
                "level": source.get("level"),
                "type": source.get("type"),
                "prompt": rendered,
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "sample_index": 0,
                "generation_id": generation_id,
                "observation_id": opaque_observation_id(generation_id),
                "resolved_spec_sha256": spec["resolved_spec_sha256"],
            }
        )
        requests.append({"prompt": rendered, "prompt_token_ids": prompt_ids})
    return prepared, requests


def sampling_params(profile: dict[str, Any]) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=float(profile["temperature"]),
        top_p=float(profile["top_p"]),
        top_k=int(profile["top_k"]),
        min_p=float(profile["min_p"]),
        presence_penalty=float(profile["presence_penalty"]),
        frequency_penalty=float(profile["frequency_penalty"]),
        repetition_penalty=float(profile["repetition_penalty"]),
        max_tokens=int(profile["max_new_tokens"]),
        seed=int(profile["seed"]),
    )


def complete_rows(prepared: list[dict[str, Any]], results: Any) -> list[dict[str, Any]]:
    if len(prepared) != len(results):
        raise RuntimeError("vLLM returned the wrong number of LR re-evaluation rows")
    completed = []
    for expected, result in zip(prepared, results, strict=True):
        if list(result.prompt_token_ids) != expected["prompt_token_ids"] or len(result.outputs) != 1:
            raise RuntimeError("vLLM changed an LR re-evaluation request")
        output = result.outputs[0]
        completed.append(
            {
                **expected,
                "completion": output.text,
                "completion_token_ids": list(output.token_ids),
                "completion_tokens": len(output.token_ids),
                "finish_reason": output.finish_reason,
                "stop_reason": output.stop_reason,
                "truncated": output.finish_reason == "length",
            }
        )
    return completed


def selected_alignment_conditions(
    checkpoints: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> set[str]:
    """Keep the initial student plus each LR's earliest MATH-maximizing checkpoint."""
    accuracy_by_condition = {
        str(checkpoint["condition"]): float(
            summarize_math_evaluations([row for row in evaluations if row["condition"] == checkpoint["condition"]])[
                "exact_accuracy"
            ]
        )
        for checkpoint in checkpoints
    }
    selected = {"initial_step_000"}
    learning_rates = sorted({float(row["learning_rate"]) for row in checkpoints if row["learning_rate"]})
    for learning_rate in learning_rates:
        candidates = sorted(
            (row for row in checkpoints if float(row["learning_rate"]) == learning_rate),
            key=lambda row: int(row["step"]),
        )
        best_accuracy = max(accuracy_by_condition[str(row["condition"])] for row in candidates)
        selected.add(
            str(
                next(
                    row["condition"]
                    for row in candidates
                    if accuracy_by_condition[str(row["condition"])] == best_accuracy
                )
            )
        )
    return selected


def existing_or_generate(
    engine: Any,
    tokenizer: Any,
    spec: dict[str, Any],
    raw: dict[str, Any],
    checkpoint: dict[str, Any],
    kind: str,
    sources: list[dict[str, Any]],
    output_dir: Path,
    lora_id: int,
) -> list[dict[str, Any]]:
    from vllm.lora.request import LoRARequest

    prepared, requests = prepare_requests(tokenizer, spec, raw, checkpoint, kind, sources)
    path = output_dir / "generations" / f"{checkpoint['condition']}__{kind}.jsonl"
    if path.exists():
        rows = read_jsonl(path)
        expected_ids = [row["generation_id"] for row in prepared]
        if [row.get("generation_id") for row in rows] != expected_ids:
            raise ConfigurationError(f"existing LR re-evaluation job does not match: {path}")
        return rows
    profile = raw["generation"]["math_internal_eval" if kind == "math" else "alignment_eval_development"]
    request = LoRARequest(
        lora_name=f"lr-v2-{checkpoint['condition']}",
        lora_int_id=lora_id,
        lora_path=str(checkpoint["adapter_path"]),
        base_model_name=str(raw["models"]["student"]["id"]),
    )
    results = engine.generate(
        requests,
        sampling_params(profile),
        use_tqdm=True,
        lora_request=request,
    )
    rows = complete_rows(prepared, results)
    write_jsonl_atomic(path, rows)
    return rows


def generate(output_dir: Path) -> dict[str, Any]:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("LR re-evaluation generation requires elevated guarded GPU execution")
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    _historical_experiment, raw, checkpoints = load_checkpoint_trajectory(root)
    spec = resolve_experiment_spec(root / "configs" / "experiment.yaml")
    math_sources, alignment_sources = source_rows(root)
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_spec.json"
    if resolved_path.exists():
        with resolved_path.open(encoding="utf-8") as handle:
            if json.load(handle).get("resolved_spec_sha256") != spec["resolved_spec_sha256"]:
                raise ConfigurationError("LR re-evaluation directory belongs to a different experiment spec")
    else:
        write_json_atomic(resolved_path, spec)

    student = raw["models"]["student"]
    snapshot = cached_model_snapshot(str(student["id"]), str(student["revision"]))
    text_view = output_dir / "model_view" / f"student-text-{student['revision']}"
    prepare_qwen35_text_only_snapshot_view(
        source_snapshot=snapshot,
        output_dir=text_view,
        model_id=str(student["id"]),
        revision=str(student["revision"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    runtime = raw["generation"]["student_evaluation_runtime"]
    math_profile = raw["generation"]["math_internal_eval"]
    alignment_profile = raw["generation"]["alignment_eval_development"]
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=str(student["dtype"]),
        seed=int(raw["experiment"]["seed"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        max_model_len=max(
            int(math_profile["vllm_max_model_length"]),
            int(alignment_profile["vllm_max_model_length"]),
        ),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=int(raw["models"]["student"]["lora"]["r"]),
        max_loras=1,
        max_cpu_loras=2,
    )
    math = []
    alignment = []
    try:
        for lora_id, checkpoint in enumerate(checkpoints, start=1):
            math.extend(
                existing_or_generate(
                    engine,
                    tokenizer,
                    spec,
                    raw,
                    checkpoint,
                    "math",
                    math_sources,
                    output_dir,
                    lora_id,
                )
            )
        source_by_id = {str(row["source_id"]): row for row in math_sources}
        evaluations = [
            {
                **{key: value for key, value in row.items() if key not in {"prompt", "prompt_token_ids"}},
                **evaluate_math_completion(
                    gold_solution=str(source_by_id[str(row["source_id"])]["gold_solution"]),
                    completion=str(row["completion"]),
                ),
            }
            for row in math
        ]
        alignment_conditions = selected_alignment_conditions(checkpoints, evaluations)
        for lora_id, checkpoint in enumerate(checkpoints, start=1):
            if checkpoint["condition"] not in alignment_conditions:
                continue
            alignment.extend(
                existing_or_generate(
                    engine,
                    tokenizer,
                    spec,
                    raw,
                    checkpoint,
                    "alignment",
                    alignment_sources,
                    output_dir,
                    lora_id,
                )
            )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)

    generations = [*math, *alignment]
    write_jsonl_atomic(output_dir / "raw_generations.jsonl", generations)
    write_jsonl_atomic(output_dir / "math_generations.jsonl", math)
    write_jsonl_atomic(output_dir / "math_evaluations.jsonl", evaluations)
    write_jsonl_atomic(output_dir / "alignment_generations.jsonl", alignment)
    task_report = export_generation_judge_tasks_v2(
        alignment,
        prompt_records=spec["prompts"],
        output_path=output_dir / "judge_tasks.jsonl",
        metrics=("alignment", "coherence"),
        seed=int(raw["experiment"]["seed"]),
        resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
    )
    report = {
        "schema_version": 1,
        "status": "generated_unscored",
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "rows": {"math": len(math), "alignment": len(alignment), "judge_tasks": 2 * len(alignment)},
        "alignment_selected_conditions": sorted(alignment_conditions),
        "checkpoints": [
            {
                key: checkpoint[key]
                for key in (
                    "condition",
                    "training_run_id",
                    "learning_rate",
                    "step",
                    "checkpoint_id",
                    "adapter_model_sha256",
                    "adapter_state_sha256",
                )
            }
            for checkpoint in checkpoints
        ],
        "math": {
            condition: summarize_math_evaluations([row for row in evaluations if row["condition"] == condition])
            for condition in sorted({str(row["condition"]) for row in evaluations})
        },
        "judge_task_export": task_report,
        "generation": {
            "math": math_profile,
            "alignment": alignment_profile,
            "guard": guard,
        },
        "artifacts": {
            name: sha256_file(output_dir / name)
            for name in (
                "resolved_spec.json",
                "raw_generations.jsonl",
                "math_generations.jsonl",
                "math_evaluations.jsonl",
                "alignment_generations.jsonl",
                "judge_tasks.jsonl",
            )
        },
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def summarize(output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    raw = load_yaml(root / "configs" / "experiment.yaml")
    summary_path = ensure_within_workspace(output_dir / "summary.json")
    with summary_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    report["alignment"] = summarize_alignment_judgments(
        read_jsonl(output_dir / "alignment_generations.jsonl"),
        read_jsonl(output_dir / "judgments.jsonl"),
        alignment_score_below=float(raw["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(raw["judge"]["thresholds"]["coherence_score_above"]),
        primary_split="em_broad_eval_v1",
    )
    report["status"] = report["alignment"]["status"]
    write_json_atomic(summary_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output_dir) if args.command == "generate" else summarize(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
