#!/usr/bin/env python3
"""Evaluate every authenticated checkpoint from one selected SFT-teacher transfer run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reevaluate_student_lr import (
    existing_or_generate,
    source_rows,
    summarize,
)

from inheritance.base_eval import summarize_math_evaluations
from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    require_active_guard,
)
from inheritance.evaluation import evaluate_math_completion, export_generation_judge_tasks_v2
from inheritance.models import (
    cached_model_snapshot,
    load_student_adapter_initialization,
    prepare_qwen35_text_only_snapshot_view,
    register_qwen35_text_vllm_model,
    verify_student_adapter_reference_lock,
)
from inheritance.reporting import sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
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


def selected_checkpoint_trajectory(
    config_path: Path,
    training_run_dir: Path,
    expected_teacher: str,
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
    checkpoint_steps = [int(value) for value in schedule.get("checkpoint_steps", [])]
    if target_steps <= 0 or not checkpoint_steps or checkpoint_steps[-1] != target_steps:
        raise ConfigurationError("student transfer checkpoint schedule is invalid")
    rollouts = _validate_training_telemetry(run_dir=training_run_dir, summary=summary, schedule=schedule)
    initialization = load_student_adapter_initialization(
        root / "artifacts" / "student_init",
        int(raw["experiment"]["seed"]),
        int(raw["models"]["student"]["lora"]["r"]),
        expected_model_id=experiment.models.student,
        expected_revision=experiment.models.student_revision,
    )
    verify_student_adapter_reference_lock(initialization)
    initial_path = ensure_within_workspace(
        root
        / "artifacts"
        / "student_init"
        / f"qwen35_2b_r{experiment.lora.r}_seed{raw['experiment']['seed']}"
    )
    checkpoints = [_checkpoint_adapter(initial_path, step=0, experiment=experiment)]
    checkpoints.extend(
        _checkpoint_adapter(
            training_run_dir / f"checkpoint-{step}",
            step=step,
            experiment=experiment,
        )
        for step in checkpoint_steps
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
    return experiment, raw, contract, trajectory


def _evaluation_contract(
    *,
    config_path: Path,
    training_run_dir: Path,
    training_contract: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    teacher: str,
) -> dict[str, Any]:
    root = repository_root()
    body = {
        "schema_version": 1,
        "resolved_spec_sha256": training_contract["resolved_spec_sha256"],
        "teacher": teacher,
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
            for name in ("math_validation_v1", "em_broad_eval_v1")
        },
        "generation_profiles": ["math_internal_eval", "alignment_eval_development"],
        "implementation_sha256": {
            "scripts/evaluate_selected_student.py": sha256_file(Path(__file__).resolve()),
            "scripts/reevaluate_student_lr.py": sha256_file(root / "scripts" / "reevaluate_student_lr.py"),
        },
    }
    return {**body, "contract_sha256": sha256_json(body)}


def generate(config_path: Path, training_run_dir: Path, output_dir: Path, teacher: str) -> dict[str, Any]:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("selected student evaluation generation requires elevated guarded GPU execution")
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    experiment, raw, training_contract, checkpoints = selected_checkpoint_trajectory(
        config_path,
        training_run_dir,
        teacher,
    )
    spec = resolve_experiment_spec(config_path)
    math_sources, alignment_sources = source_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _evaluation_contract(
        config_path=config_path,
        training_run_dir=training_run_dir,
        training_contract=training_contract,
        checkpoints=checkpoints,
        teacher=teacher,
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
        max_lora_rank=int(student["lora"]["r"]),
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
        "training_run_id": training_contract["run_id"],
        "training_condition": teacher,
        "evaluation_contract_sha256": contract["contract_sha256"],
        "rows": {"math": len(math), "alignment": len(alignment), "judge_tasks": 2 * len(alignment)},
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--teacher", choices=("sft_bad", "sft_aligned"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = ensure_within_workspace(args.config)
    training_run_dir = ensure_within_workspace(args.training_run_dir)
    output_dir = ensure_within_workspace(args.output_dir)
    report = (
        generate(config_path, training_run_dir, output_dir, args.teacher)
        if args.phase == "generate"
        else summarize(output_dir)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
