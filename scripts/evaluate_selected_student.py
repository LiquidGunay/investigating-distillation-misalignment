#!/usr/bin/env python3
"""Evaluate authenticated checkpoints from one completed student transfer run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reevaluate_student_lr import render_math_prompt

from inheritance.base_eval import summarize_alignment_judgments, summarize_math_evaluations
from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    require_active_guard,
)
from inheritance.evaluation import evaluate_math_completion, export_generation_judge_tasks_v2
from inheritance.judge_api import resolve_judge_lineage
from inheritance.models import (
    _extract_chat_template_input_ids,
    cached_model_snapshot,
    load_student_adapter_initialization,
    prepare_qwen35_text_only_snapshot_view,
    register_qwen35_text_vllm_model,
    verify_student_adapter_reference_lock,
)
from inheritance.phenomenon import authenticate_judgment_packet
from inheritance.reporting import (
    opaque_observation_id,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec
from inheritance.student_eval import (
    _checkpoint_adapter,
    _validate_checkpoint_training_lineage,
    _validate_final_adapter_files,
    _validate_run_artifacts,
    _validate_training_telemetry,
)


def _read_object(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a JSON object: {path}")
    return value


def _source_rows(root: Path, stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    math_manifest = "math_validation_v1" if stage == "development" else "math_test_v1"
    manifest_root = root / "artifacts" / "manifests"
    math_rows = read_jsonl(manifest_root / f"{math_manifest}.jsonl")
    alignment_rows = read_jsonl(manifest_root / "em_broad_eval_v1.jsonl")
    expected_math = 500 if stage == "development" else 4500
    if len(math_rows) != expected_math or len(alignment_rows) != 240:
        raise ConfigurationError("selected-student evaluation manifests have unexpected row counts")
    return math_rows, alignment_rows


def _sampling_params(profile: dict[str, Any], *, samples: int) -> Any:
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
        n=samples,
    )


def _prepare_requests(
    tokenizer: Any,
    spec: dict[str, Any],
    raw: dict[str, Any],
    checkpoint: dict[str, Any],
    kind: str,
    sources: list[dict[str, Any]],
    *,
    dataset_split: str,
    profile_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile = raw["generation"][profile_name]
    selected = str(raw["prompts"]["math"]["selected_capability_prompt"])
    prepared = []
    requests = []
    for source in sources:
        content = (
            render_math_prompt(spec, selected, str(source["problem"]))
            if kind == "math"
            else str(source["question"])
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
                "dataset_split": dataset_split,
                "source_id": str(source["source_id"]),
                "question": str(source.get("question", source.get("problem"))),
                "task": str(source.get("task", "math")),
                "domain": source.get("domain"),
                "level": source.get("level"),
                "type": source.get("type"),
                "prompt": rendered,
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "resolved_spec_sha256": spec["resolved_spec_sha256"],
            }
        )
        requests.append({"prompt": rendered, "prompt_token_ids": prompt_ids})
    return prepared, requests


def _completed_rows(
    prepared: list[dict[str, Any]],
    results: Any,
    *,
    samples: int,
) -> list[dict[str, Any]]:
    if len(prepared) != len(results):
        raise RuntimeError("vLLM returned the wrong number of selected-student evaluation rows")
    completed = []
    for expected, result in zip(prepared, results, strict=True):
        if list(result.prompt_token_ids) != expected["prompt_token_ids"] or len(result.outputs) != samples:
            raise RuntimeError("vLLM changed a selected-student evaluation request")
        for sample_index, output in enumerate(result.outputs):
            identity = {
                "model_role": "student",
                "condition": expected["condition"],
                "training_run_id": expected["training_run_id"],
                "kind": expected["evaluation_kind"],
                "source_id": expected["source_id"],
                "sample_index": sample_index,
                "resolved_spec_sha256": expected["resolved_spec_sha256"],
            }
            generation_id = f"generation_{sha256_json(identity)[:24]}"
            completed.append(
                {
                    **expected,
                    "sample_index": sample_index,
                    "generation_id": generation_id,
                    "observation_id": opaque_observation_id(generation_id),
                    "completion": output.text,
                    "completion_token_ids": list(output.token_ids),
                    "completion_tokens": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "truncated": output.finish_reason == "length",
                }
            )
    return completed


def _existing_or_generate(
    engine: Any,
    tokenizer: Any,
    spec: dict[str, Any],
    raw: dict[str, Any],
    checkpoint: dict[str, Any],
    kind: str,
    sources: list[dict[str, Any]],
    output_dir: Path,
    lora_id: int,
    *,
    stage: str,
    dataset_split: str,
    profile_name: str,
    samples: int,
) -> list[dict[str, Any]]:
    from vllm.lora.request import LoRARequest

    prepared, requests = _prepare_requests(
        tokenizer,
        spec,
        raw,
        checkpoint,
        kind,
        sources,
        dataset_split=dataset_split,
        profile_name=profile_name,
    )
    path = output_dir / "generations" / f"{checkpoint['condition']}__{kind}__{stage}.jsonl"
    expected_ids = []
    for row in prepared:
        for sample_index in range(samples):
            identity = {
                "model_role": "student",
                "condition": row["condition"],
                "training_run_id": row["training_run_id"],
                "kind": row["evaluation_kind"],
                "source_id": row["source_id"],
                "sample_index": sample_index,
                "resolved_spec_sha256": row["resolved_spec_sha256"],
            }
            expected_ids.append(f"generation_{sha256_json(identity)[:24]}")
    if path.is_file():
        rows = read_jsonl(path)
        if [row.get("generation_id") for row in rows] != expected_ids:
            raise ConfigurationError(f"existing selected-student evaluation job does not match: {path}")
        return rows
    profile = raw["generation"][profile_name]
    request = LoRARequest(
        lora_name=f"selected-{stage}-{checkpoint['condition']}",
        lora_int_id=lora_id,
        lora_path=str(checkpoint["adapter_path"]),
        base_model_name=str(raw["models"]["student"]["id"]),
    )
    results = engine.generate(
        requests,
        sampling_params=_sampling_params(profile, samples=samples),
        use_tqdm=True,
        lora_request=request,
    )
    rows = _completed_rows(prepared, results, samples=samples)
    if [row["generation_id"] for row in rows] != expected_ids:
        raise RuntimeError("selected-student generation order differs from its request contract")
    write_jsonl_atomic(path, rows)
    return rows


def selected_checkpoint_trajectory(
    config_path: Path,
    training_run_dir: Path,
    expected_teacher: str,
    requested_checkpoint_steps: set[int] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = repository_root()
    experiment = load_experiment_config(config_path)
    raw = load_yaml(config_path)
    summary = _read_object(training_run_dir / "run.json")
    contract = _read_object(training_run_dir / "run_contract.json")
    contract_digest = contract.get("contract_sha256")
    if contract_digest != sha256_json({key: value for key, value in contract.items() if key != "contract_sha256"}):
        raise ConfigurationError("student transfer run contract has an invalid internal digest")
    if contract.get("resolved_spec_sha256") != experiment.resolved_spec_sha256:
        raise ConfigurationError("student transfer run used a different resolved experiment spec")
    if contract.get("teacher", {}).get("condition") != expected_teacher:
        raise ConfigurationError("student transfer run teacher differs from the selected evaluation source")
    if summary.get("status") != "completed" or summary.get("teacher_condition") != expected_teacher:
        raise ConfigurationError("selected student transfer run is not complete for the expected teacher")
    _validate_run_artifacts(training_run_dir, summary)
    schedule = contract.get("schedule")
    if not isinstance(schedule, dict):
        raise ConfigurationError("student transfer run has no immutable schedule")
    target_steps = int(schedule.get("total_optimizer_steps", -1))
    scheduled_steps = [int(value) for value in schedule.get("checkpoint_steps", [])]
    if target_steps <= 0 or not scheduled_steps or scheduled_steps[-1] != target_steps:
        raise ConfigurationError("student transfer checkpoint schedule is invalid")
    rollouts = _validate_training_telemetry(run_dir=training_run_dir, summary=summary, schedule=schedule)
    run_seed = int(contract.get("run", {}).get("seed", -1))
    if run_seed not in {int(value) for value in raw["experiment"]["seeds"]}:
        raise ConfigurationError("selected student run used a seed outside experiment.seeds")
    initialization = load_student_adapter_initialization(
        root / "artifacts" / "student_init",
        run_seed,
        int(raw["models"]["student"]["lora"]["r"]),
        expected_model_id=experiment.models.student,
        expected_revision=experiment.models.student_revision,
    )
    verify_student_adapter_reference_lock(initialization)
    initial_path = ensure_within_workspace(
        root
        / "artifacts"
        / "student_init"
        / f"qwen35_2b_r{experiment.lora.r}_seed{run_seed}"
    )
    checkpoints = [_checkpoint_adapter(initial_path, step=0, experiment=experiment)]
    checkpoints.extend(
        _checkpoint_adapter(
            training_run_dir / f"checkpoint-{step}",
            step=step,
            experiment=experiment,
        )
        for step in scheduled_steps
    )
    final_files = _validate_final_adapter_files(training_run_dir, summary)
    _validate_checkpoint_training_lineage(
        checkpoints=checkpoints,
        rollouts=rollouts,
        target_steps=target_steps,
        final_files=final_files,
    )
    if checkpoints[0]["adapter_model_sha256"] != contract.get("student", {}).get("adapter_model_sha256"):
        raise ConfigurationError("student transfer initialization differs from its run contract")
    run_id = str(contract["run_id"])
    learning_rate = float(contract["run"]["learning_rate"])
    trajectory = [
        {
            **checkpoint,
            "condition": f"{expected_teacher}_step_{int(checkpoint['step']):04d}",
            "training_run_id": run_id,
            "learning_rate": learning_rate if int(checkpoint["step"]) else 0.0,
        }
        for checkpoint in checkpoints
    ]
    if requested_checkpoint_steps is not None:
        available = {int(checkpoint["step"]) for checkpoint in trajectory}
        if not requested_checkpoint_steps or not requested_checkpoint_steps <= available:
            raise ConfigurationError(
                f"requested checkpoint steps {sorted(requested_checkpoint_steps)} are not a non-empty subset of "
                f"{sorted(available)}"
            )
        trajectory = [
            checkpoint for checkpoint in trajectory if int(checkpoint["step"]) in requested_checkpoint_steps
        ]
    return experiment, raw, contract, trajectory


def _evaluation_contract(
    *,
    config_path: Path,
    training_run_dir: Path,
    training_contract: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    teacher: str,
    stage: str,
    math_manifest: str,
    alignment_profile: str,
) -> dict[str, Any]:
    root = repository_root()
    body = {
        "schema_version": 1,
        "resolved_spec_sha256": training_contract["resolved_spec_sha256"],
        "teacher": teacher,
        "evaluation_stage": stage,
        "training_run": {
            "path": str(training_run_dir.relative_to(root)),
            "run_contract_sha256": sha256_file(training_run_dir / "run_contract.json"),
            "run_summary_sha256": sha256_file(training_run_dir / "run.json"),
        },
        "experiment_config_sha256": sha256_file(config_path),
        "checkpoints": [
            {
                key: checkpoint[key]
                for key in (
                    "step",
                    "checkpoint_id",
                    "adapter_model_sha256",
                    "adapter_state_sha256",
                    "adapter_config_sha256",
                )
            }
            for checkpoint in checkpoints
        ],
        "manifests": {
            name: {
                "path": f"artifacts/manifests/{name}.jsonl",
                "sha256": sha256_file(root / "artifacts" / "manifests" / f"{name}.jsonl"),
            }
            for name in (math_manifest, "em_broad_eval_v1")
        },
        "generation_profiles": ["math_internal_eval", alignment_profile],
        "implementation_sha256": {
            "scripts/evaluate_selected_student.py": sha256_file(Path(__file__).resolve()),
            "scripts/reevaluate_student_lr.py": sha256_file(root / "scripts" / "reevaluate_student_lr.py"),
        },
    }
    return {**body, "contract_sha256": sha256_json(body)}


def generate(
    config_path: Path,
    training_run_dir: Path,
    output_dir: Path,
    teacher: str,
    *,
    stage: str,
    checkpoint_steps: set[int] | None,
) -> dict[str, Any]:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("selected student evaluation generation requires elevated guarded GPU execution")
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    if stage == "final" and checkpoint_steps is None:
        raise ConfigurationError("final evaluation requires an explicit --checkpoint-steps selection")
    experiment, raw, training_contract, checkpoints = selected_checkpoint_trajectory(
        config_path,
        training_run_dir,
        teacher,
        checkpoint_steps,
    )
    spec = resolve_experiment_spec(config_path)
    math_manifest = "math_validation_v1" if stage == "development" else "math_test_v1"
    alignment_profile_name = (
        "alignment_eval_development" if stage == "development" else "alignment_eval_final"
    )
    alignment_samples = 1 if stage == "development" else int(
        raw["generation"]["alignment_eval_final"]["broad_samples_per_prompt"]
    )
    math_sources, alignment_sources = _source_rows(root, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _evaluation_contract(
        config_path=config_path,
        training_run_dir=training_run_dir,
        training_contract=training_contract,
        checkpoints=checkpoints,
        teacher=teacher,
        stage=stage,
        math_manifest=math_manifest,
        alignment_profile=alignment_profile_name,
    )
    contract_path = output_dir / "evaluation_contract.json"
    if contract_path.is_file() and _read_object(contract_path) != contract:
        raise ConfigurationError("selected student evaluation directory has a different contract")
    if not contract_path.is_file() and any(output_dir.iterdir()):
        raise ConfigurationError("refusing to attach an evaluation contract to a non-empty directory")
    write_json_atomic(contract_path, contract)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
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
    alignment_profile = raw["generation"][alignment_profile_name]
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
        max_lora_rank=int(student["lora"]["r"]),
        max_loras=1,
        max_cpu_loras=2,
    )
    math = []
    alignment = []
    try:
        for lora_id, checkpoint in enumerate(checkpoints, start=1):
            math.extend(
                _existing_or_generate(
                    engine,
                    tokenizer,
                    spec,
                    raw,
                    checkpoint,
                    "math",
                    math_sources,
                    output_dir,
                    lora_id,
                    stage=stage,
                    dataset_split=math_manifest,
                    profile_name="math_internal_eval",
                    samples=1,
                )
            )
            alignment.extend(
                _existing_or_generate(
                    engine,
                    tokenizer,
                    spec,
                    raw,
                    checkpoint,
                    "alignment",
                    alignment_sources,
                    output_dir,
                    lora_id,
                    stage=stage,
                    dataset_split="em_broad_eval_v1",
                    profile_name=alignment_profile_name,
                    samples=alignment_samples,
                )
            )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
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
    write_jsonl_atomic(output_dir / "raw_generations.jsonl", [*math, *alignment])
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
        "evaluation_stage": stage,
        "training_run_id": training_contract["run_id"],
        "training_seed": int(training_contract["run"]["seed"]),
        "training_condition": teacher,
        "evaluation_contract_sha256": contract["contract_sha256"],
        "rows": {"math": len(math), "alignment": len(alignment), "judge_tasks": 2 * len(alignment)},
        "surfaces": {
            "math": math_manifest,
            "alignment": "em_broad_eval_v1",
            "alignment_samples_per_prompt": alignment_samples,
        },
        "checkpoints": checkpoints,
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
                "evaluation_contract.json",
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


def summarize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    raw = load_yaml(config_path)
    report = _read_object(output_dir / "summary.json")
    spec = resolve_experiment_spec(config_path)
    if report.get("resolved_spec_sha256") != spec.get("resolved_spec_sha256"):
        raise ConfigurationError("selected-student evaluation summary belongs to another experiment spec")
    contract = _read_object(output_dir / "evaluation_contract.json")
    if (
        contract.get("contract_sha256")
        != sha256_json({key: value for key, value in contract.items() if key != "contract_sha256"})
        or report.get("evaluation_contract_sha256") != contract.get("contract_sha256")
    ):
        raise ConfigurationError("selected-student evaluation contract is invalid")
    for name, expected_sha256 in report.get("artifacts", {}).items():
        path = ensure_within_workspace(output_dir / str(name))
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ConfigurationError(f"selected-student evaluation artifact differs: {name}")
    teacher_selection_path = root / "artifacts" / "selection" / "teacher_sources_v2.json"
    teacher_selection = _read_object(teacher_selection_path)
    lineage_id = str(teacher_selection.get("judge_lineage"))
    lineage, lineage_spec_sha256 = resolve_judge_lineage(config_path, lineage_id)
    if lineage_spec_sha256 != report["resolved_spec_sha256"]:
        raise ConfigurationError("selected judge lineage belongs to another experiment spec")
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = authenticate_judgment_packet(
        output_dir / "judge_tasks.jsonl",
        output_dir / "judgments.jsonl",
        generations,
        lineage=lineage,
    )
    report["alignment"] = summarize_alignment_judgments(
        generations,
        judgments,
        alignment_score_below=float(raw["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(raw["judge"]["thresholds"]["coherence_score_above"]),
        primary_split="em_broad_eval_v1",
    )
    report["status"] = report["alignment"]["status"]
    report["judge_lineage_authentication"] = {
        "lineage_id": lineage_id,
        "teacher_selection_path": str(teacher_selection_path.relative_to(root)),
        "teacher_selection_sha256": sha256_file(teacher_selection_path),
        "judge_tasks_sha256": sha256_file(output_dir / "judge_tasks.jsonl"),
        "judgments_sha256": sha256_file(output_dir / "judgments.jsonl"),
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def _checkpoint_steps(value: str | None) -> set[int] | None:
    if value is None:
        return None
    try:
        steps = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint steps must be comma-separated integers") from exc
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("checkpoint steps must be non-negative and non-empty")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--teacher", choices=("base", "sft_bad", "sft_aligned"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--checkpoint-steps")
    args = parser.parse_args()
    config_path = ensure_within_workspace(args.config)
    training_run_dir = ensure_within_workspace(args.training_run_dir)
    output_dir = ensure_within_workspace(args.output_dir)
    report = (
        generate(
            config_path,
            training_run_dir,
            output_dir,
            args.teacher,
            stage=args.stage,
            checkpoint_steps=_checkpoint_steps(args.checkpoint_steps),
        )
        if args.phase == "generate"
        else summarize(config_path, output_dir)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
