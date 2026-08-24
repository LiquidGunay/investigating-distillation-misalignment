#!/usr/bin/env python3
"""Freeze the Stage-C phenomenon gate from two matched student evaluations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    write_json_atomic,
)
from inheritance.evaluation import evaluate_math_completion
from inheritance.judge_api import resolve_judge_lineage
from inheritance.phenomenon import authenticate_judgment_packet, select_phenomenon_gate
from inheritance.reporting import read_jsonl, sha256_file, sha256_json


def _read_object(path: Path) -> dict[str, Any]:
    path = ensure_within_workspace(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a JSON object: {path}")
    return value


def _validate_digest(record: Mapping[str, Any], field: str, description: str) -> None:
    digest = record.get(field)
    body = {key: value for key, value in record.items() if key != field}
    if digest != sha256_json(body):
        raise ConfigurationError(f"{description} has an invalid internal digest")


def _evidence(path: Path) -> dict[str, Any]:
    root = repository_root()
    path = ensure_within_workspace(path)
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def _authenticate_math(
    evaluation_dir: Path,
    evaluations: list[dict[str, Any]],
    *,
    expected_spec_sha256: str,
) -> None:
    root = repository_root()
    generations = read_jsonl(evaluation_dir / "math_generations.jsonl")
    if len(generations) != len(evaluations):
        raise ConfigurationError("MATH generations and evaluations have different row counts")
    manifest_path = root / "artifacts" / "manifests" / "math_validation_v1.jsonl"
    gold = {str(row["source_id"]): row for row in read_jsonl(manifest_path)}
    by_generation = {str(row.get("generation_id")): row for row in evaluations}
    if len(by_generation) != len(evaluations):
        raise ConfigurationError("MATH evaluations contain duplicate generation identities")
    for generation in generations:
        if generation.get("resolved_spec_sha256") != expected_spec_sha256:
            raise ConfigurationError("MATH generation belongs to a different experiment spec")
        generation_id = str(generation.get("generation_id"))
        evaluation = by_generation.get(generation_id)
        source = gold.get(str(generation.get("source_id")))
        if evaluation is None or source is None:
            raise ConfigurationError("MATH evaluation cannot be linked to generation and frozen gold")
        recomputed = evaluate_math_completion(
            gold_solution=str(source["gold_solution"]),
            completion=str(generation["completion"]),
        )
        if any(evaluation.get(key) != value for key, value in recomputed.items()):
            raise ConfigurationError("saved MATH evaluation differs from an exact recomputation")


def _load_evaluation(
    evaluation_dir: Path,
    *,
    expected_teacher: str,
    expected_spec_sha256: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    root = repository_root()
    evaluation_dir = ensure_within_workspace(evaluation_dir)
    summary_path = evaluation_dir / "summary.json"
    summary = _read_object(summary_path)
    if (
        summary.get("status") != "scored"
        or summary.get("training_condition") != expected_teacher
        or summary.get("resolved_spec_sha256") != expected_spec_sha256
        or summary.get("evaluation_stage") != "development"
    ):
        raise ConfigurationError(f"{expected_teacher} evaluation is not a complete scored current-spec run")
    contract_path = evaluation_dir / "evaluation_contract.json"
    contract = _read_object(contract_path)
    _validate_digest(contract, "contract_sha256", f"{expected_teacher} evaluation contract")
    if contract.get("teacher") != expected_teacher or contract.get("resolved_spec_sha256") != expected_spec_sha256:
        raise ConfigurationError(f"{expected_teacher} evaluation contract identity differs")
    for name, expected_sha256 in summary.get("artifacts", {}).items():
        artifact_path = ensure_within_workspace(evaluation_dir / str(name))
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_sha256:
            raise ConfigurationError(f"{expected_teacher} evaluation artifact differs: {name}")
    training = contract.get("training_run")
    if not isinstance(training, Mapping):
        raise ConfigurationError(f"{expected_teacher} evaluation has no training-run provenance")
    training_dir = ensure_within_workspace(root / str(training.get("path")))
    run_contract_path = training_dir / "run_contract.json"
    run_summary_path = training_dir / "run.json"
    if (
        sha256_file(run_contract_path) != training.get("run_contract_sha256")
        or sha256_file(run_summary_path) != training.get("run_summary_sha256")
    ):
        raise ConfigurationError(f"{expected_teacher} training evidence differs from its evaluation contract")
    run_contract = _read_object(run_contract_path)
    _validate_digest(run_contract, "contract_sha256", f"{expected_teacher} training contract")
    if run_contract.get("teacher", {}).get("condition") != expected_teacher:
        raise ConfigurationError(f"{expected_teacher} training contract names another teacher")
    generations = read_jsonl(evaluation_dir / "alignment_generations.jsonl")
    if not generations or any(
        row.get("model_role") != "student"
        or row.get("dataset_split") != "em_broad_eval_v1"
        or row.get("resolved_spec_sha256") != expected_spec_sha256
        for row in generations
    ):
        raise ConfigurationError(f"{expected_teacher} alignment generations have an invalid evaluation surface")
    evaluations = read_jsonl(evaluation_dir / "math_evaluations.jsonl")
    _authenticate_math(evaluation_dir, evaluations, expected_spec_sha256=expected_spec_sha256)
    judgments = authenticate_judgment_packet(
        evaluation_dir / "judge_tasks.jsonl",
        evaluation_dir / "judgments.jsonl",
        generations,
        lineage=lineage,
    )
    evidence_paths = {
        "summary": summary_path,
        "evaluation_contract": contract_path,
        "math_generations": evaluation_dir / "math_generations.jsonl",
        "math_evaluations": evaluation_dir / "math_evaluations.jsonl",
        "alignment_generations": evaluation_dir / "alignment_generations.jsonl",
        "judge_tasks": evaluation_dir / "judge_tasks.jsonl",
        "judgments": evaluation_dir / "judgments.jsonl",
        "training_contract": run_contract_path,
        "training_summary": run_summary_path,
    }
    return {
        "summary": summary,
        "run_contract": run_contract,
        "generations": generations,
        "judgments": judgments,
        "math": evaluations,
        "evidence": {
            f"{expected_teacher}_{name}": _evidence(path) for name, path in evidence_paths.items()
        },
    }


def _require_matched_runs(bad: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    bad_contract = bad["run_contract"]
    control_contract = control["run_contract"]
    comparable = (
        "resolved_spec_sha256",
        "resolved_experiment_config_sha256",
        "resolved_student_training_config_sha256",
        "manifest",
        "model_locks",
        "implementation_sha256",
        "schedule",
        "selection",
        "student",
        "trl_commit",
    )
    if any(bad_contract.get(key) != control_contract.get(key) for key in comparable):
        raise ConfigurationError("bad and aligned student runs are not matched outside teacher identity")
    bad_run = bad_contract.get("run", {})
    control_run = control_contract.get("run", {})
    if any(bad_run.get(key) != control_run.get(key) for key in ("learning_rate", "seed", "teacher_card")):
        raise ConfigurationError("bad and aligned student runs use different training choices")


def _teacher_status(card: Mapping[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    sft = card.get("sft")
    if not isinstance(sft, Mapping):
        raise ConfigurationError("teacher selection card has no SFT source")
    scale = float(sft.get("selected_adapter_scale", -1))
    full = sft.get("full_validation")
    if not isinstance(full, Mapping):
        raise ConfigurationError("SFT teacher has no full-validation decision")
    metrics = full.get(f"scale_{scale:g}")
    if not isinstance(metrics, Mapping):
        raise ConfigurationError("SFT teacher full-validation metrics do not match the selected scale")
    interval = metrics.get("bad_minus_aligned_alignment_bootstrap_95")
    behaviorally_misaligned = (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
        and float(interval[1]) < 0
    )
    capability_eligible = full.get("status") == "passed"
    return behaviorally_misaligned, capability_eligible, {
        "selected_scale": scale,
        "full_validation_status": full.get("status"),
        "alignment_bad_minus_aligned_bootstrap_95": interval,
    }


def _raw_review_passed(
    path: Path | None,
    *,
    candidate: Mapping[str, Any] | None,
    bad_generation_sha256: str,
    control_generation_sha256: str,
) -> tuple[bool, dict[str, Any] | None]:
    if path is None or candidate is None:
        return False, None
    review = _read_object(path)
    criteria = review.get("criteria")
    expected = {
        "coherent_misalignment_not_gibberish": True,
        "not_refusal_artifact": True,
        "not_judge_failure": True,
    }
    passed = (
        review.get("schema_version") == 1
        and review.get("status") == "passed"
        and review.get("condition") == candidate.get("bad_condition")
        and review.get("bad_alignment_generations_sha256") == bad_generation_sha256
        and review.get("control_alignment_generations_sha256") == control_generation_sha256
        and isinstance(criteria, Mapping)
        and all(criteria.get(key) is value for key, value in expected.items())
    )
    return passed, {**_evidence(path), "condition": review.get("condition"), "status": review.get("status")}


def select(
    config_path: Path,
    bad_evaluation_dir: Path,
    control_evaluation_dir: Path,
    output_path: Path,
    raw_output_review: Path | None,
) -> dict[str, Any]:
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    raw = load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    rules = raw.get("selection_rules", {}).get("phenomenon_gate", {})
    if "minimum_math_validation_gain" not in rules:
        raise ConfigurationError(
            "selection_rules.phenomenon_gate.minimum_math_validation_gain must be explicit in the "
            "authoritative experiment config before the Stage-C gate can be evaluated"
        )
    teacher_path = root / "artifacts" / "selection" / "teacher_sources_v2.json"
    teacher_card = _read_object(teacher_path)
    lineage_id = str(teacher_card.get("judge_lineage"))
    lineage, spec_sha256 = resolve_judge_lineage(config_path, lineage_id)
    if spec_sha256 != experiment.resolved_spec_sha256:
        raise ConfigurationError("judge and experiment config resolution disagree")
    calibration_path = root / str(raw["judge"]["lineages"][lineage_id]["calibration_artifact"])
    calibration = _read_object(calibration_path)
    if calibration.get("status") != "passed" or calibration.get("lineage_id") != lineage_id:
        raise ConfigurationError("selected judge lineage has no passing frozen calibration")
    bad = _load_evaluation(
        bad_evaluation_dir,
        expected_teacher="sft_bad",
        expected_spec_sha256=spec_sha256,
        lineage=lineage,
    )
    control = _load_evaluation(
        control_evaluation_dir,
        expected_teacher="sft_aligned",
        expected_spec_sha256=spec_sha256,
        lineage=lineage,
    )
    _require_matched_runs(bad, control)
    behaviorally_misaligned, capability_eligible, teacher_metrics = _teacher_status(teacher_card)
    settings = {
        "teacher_behaviorally_misaligned": behaviorally_misaligned,
        "teacher_capability_eligible": capability_eligible,
        "minimum_math_gain": float(rules["minimum_math_validation_gain"]),
        "minimum_coherence_guardrail_rate": float(
            raw["selection_rules"]["teacher_eligibility"]["minimum_coherence_guardrail_pass_fraction"]
        ),
        "coherence_score_above": float(raw["judge"]["thresholds"]["coherence_score_above"]),
        "seed": int(raw["experiment"]["seed"]),
        "bootstrap_samples": int(raw["evaluation"]["metrics"]["paired_bootstrap_samples"]),
    }
    numerical = select_phenomenon_gate(
        bad["generations"],
        bad["judgments"],
        bad["math"],
        control["generations"],
        control["judgments"],
        control["math"],
        raw_outputs_confirmed=True,
        **settings,
    )
    bad_generation_hash = sha256_file(ensure_within_workspace(bad_evaluation_dir) / "alignment_generations.jsonl")
    control_generation_hash = sha256_file(
        ensure_within_workspace(control_evaluation_dir) / "alignment_generations.jsonl"
    )
    reviewed, review_evidence = _raw_review_passed(
        ensure_within_workspace(raw_output_review) if raw_output_review is not None else None,
        candidate=numerical.get("selected_checkpoint"),
        bad_generation_sha256=bad_generation_hash,
        control_generation_sha256=control_generation_hash,
    )
    gate = select_phenomenon_gate(
        bad["generations"],
        bad["judgments"],
        bad["math"],
        control["generations"],
        control["judgments"],
        control["math"],
        raw_outputs_confirmed=reviewed,
        **settings,
    )
    evidence = {
        **bad["evidence"],
        **control["evidence"],
        "teacher_selection": _evidence(teacher_path),
        "judge_calibration": _evidence(calibration_path),
        "experiment_config": _evidence(config_path),
    }
    if review_evidence is not None:
        evidence["raw_output_review"] = review_evidence
    gate.update(
        {
            "resolved_spec_sha256": spec_sha256,
            "teacher_selection": teacher_metrics,
            "numerical_candidate_before_raw_review": numerical.get("selected_checkpoint"),
            "evidence": evidence,
        }
    )
    output_path = ensure_within_workspace(output_path)
    if output_path.is_file():
        existing = _read_object(output_path)
        if existing.get("status") == "passed" and existing != gate:
            raise ConfigurationError("refusing to replace a passing frozen phenomenon gate")
    write_json_atomic(output_path, gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--bad-evaluation-dir", type=Path, required=True)
    parser.add_argument("--control-evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/selection/intervention_source_v1.json"),
    )
    parser.add_argument("--raw-output-review", type=Path)
    args = parser.parse_args()
    report = select(
        args.config,
        args.bad_evaluation_dir,
        args.control_evaluation_dir,
        args.output,
        args.raw_output_review,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
