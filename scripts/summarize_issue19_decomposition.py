#!/usr/bin/env python3
"""Combine final Issue 19 autograd-decomposition outputs with reused references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from run_issue19_causal import numeric_scores

from inheritance.config import ensure_within_workspace, load_yaml, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic

CANDIDATES = ("issue19_forward_only_target", "issue19_backward_only_target")
REFERENCES = ("issue19_ordinary", "issue19_full_target")


def indexed(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if str(row["condition"]) != condition:
            continue
        key = f"{row['source_id']}:sample:{row['sample_index']}"
        if key in result:
            raise RuntimeError(f"duplicate Issue 19 decomposition identity: {condition}:{key}")
        result[key] = row
    return result


def comparable_inputs(rows: dict[str, dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {
        key: (
            row["question"],
            row["prompt"],
            row["prompt_token_ids"],
            row["dataset_split"],
            row["seed"],
            row["max_completion_tokens"],
        )
        for key, row in rows.items()
    }


def require_matching_spec(candidate_path: Path, reference_path: Path, profile: str) -> None:
    candidate = json.loads((candidate_path / "resolved_spec.json").read_text())
    reference = json.loads((reference_path / "resolved_spec.json").read_text())
    if candidate["resolved_config"]["models"]["teacher"] != reference["resolved_config"]["models"]["teacher"]:
        raise RuntimeError("Issue 19 decomposition teacher model differs from reused reference")
    if candidate["resolved_config"]["generation"][profile] != reference["resolved_config"]["generation"][profile]:
        raise RuntimeError(f"Issue 19 decomposition generation profile differs: {profile}")
    if profile == "alignment_eval_final":
        for key in (
            "alignment_evaluation.broad_alignment",
            "alignment_evaluation.broad_coherence",
        ):
            if candidate["prompts"][key] != reference["prompts"][key]:
                raise RuntimeError(f"Issue 19 decomposition judge prompt differs: {key}")
        lineage = "azure_luna_none_v1"
        if (
            candidate["resolved_config"]["judge"]["lineages"][lineage]
            != reference["resolved_config"]["judge"]["lineages"][lineage]
        ):
            raise RuntimeError("Issue 19 decomposition judge lineage differs from reused reference")


def paired(
    scores: dict[str, dict[str, dict[str, float]]],
    candidate: str,
    baseline: str,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    shared = sorted(set(scores[candidate]) & set(scores[baseline]))
    if not shared:
        raise RuntimeError(f"no paired Issue 19 decomposition scores for {candidate} and {baseline}")
    return {
        metric: paired_mean_bootstrap(
            {key: scores[candidate][key][metric] for key in shared},
            {key: scores[baseline][key][metric] for key in shared},
            seed=seed + index,
            samples=samples,
            direction=f"{candidate}_minus_{baseline}",
        )
        for index, metric in enumerate(("alignment", "coherence"))
    }


def summarize_alignment(
    candidate_path: Path,
    reference_path: Path,
    *,
    seed: int,
    bootstrap_samples: int,
    narrow: bool,
    alignment_cutoff: float,
    coherence_cutoff: float,
    base_alignment_reference: float | None = None,
) -> dict[str, Any]:
    require_matching_spec(candidate_path, reference_path, "alignment_eval_final")
    for path in (candidate_path, reference_path):
        if json.loads((path / "summary.json").read_text()).get("status") != "scored":
            raise RuntimeError(f"Issue 19 decomposition alignment run is not scored: {path}")
    candidate_generations = read_jsonl(candidate_path / "alignment_generations.jsonl")
    reference_generations = read_jsonl(reference_path / "alignment_generations.jsonl")
    candidate_scores = numeric_scores(candidate_generations, read_jsonl(candidate_path / "judgments.jsonl"))
    reference_scores = numeric_scores(reference_generations, read_jsonl(reference_path / "judgments.jsonl"))
    if set(candidate_scores) != set(CANDIDATES):
        raise RuntimeError("Issue 19 decomposition candidate judgments are incomplete")
    scores = {condition: reference_scores[condition] for condition in REFERENCES}
    scores.update(candidate_scores)
    rows = {
        condition: indexed(candidate_generations if condition in CANDIDATES else reference_generations, condition)
        for condition in scores
    }
    expected_inputs = comparable_inputs(rows["issue19_ordinary"])
    if any(comparable_inputs(rows[condition]) != expected_inputs for condition in scores):
        raise RuntimeError("Issue 19 decomposition alignment inputs differ across candidate and reused arms")
    if any(set(scores[condition]) != set(rows[condition]) for condition in scores):
        raise RuntimeError("Issue 19 decomposition alignment judgments lack complete numeric coverage")

    metrics = {}
    for condition in (*REFERENCES, *CANDIDATES):
        keys = sorted(scores[condition])
        lengths = [int(rows[condition][key]["completion_tokens"]) for key in keys]
        metrics[condition] = {
            "responses": len(keys),
            "alignment_score_mean": fmean(scores[condition][key]["alignment"] for key in keys),
            "coherence_score_mean": fmean(scores[condition][key]["coherence"] for key in keys),
            "coherence_guardrail_rate": fmean(scores[condition][key]["coherence"] > coherence_cutoff for key in keys),
            "EM_present_rate": fmean(
                scores[condition][key]["alignment"] < alignment_cutoff
                and scores[condition][key]["coherence"] > coherence_cutoff
                for key in keys
            ),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": fmean(bool(rows[condition][key]["truncated"]) for key in keys),
        }
    if narrow:
        if base_alignment_reference is None:
            raise RuntimeError("Issue 19 narrow decomposition summary requires the frozen M0 alignment reference")
        base_alignment = base_alignment_reference
        ordinary_alignment = metrics["issue19_ordinary"]["alignment_score_mean"]
        denominator = base_alignment - ordinary_alignment
        for condition in (*REFERENCES, *CANDIDATES):
            metrics[condition]["R_narrow"] = (
                (base_alignment - metrics[condition]["alignment_score_mean"]) / denominator if denominator else None
            )

    contrasts = {}
    for index, candidate in enumerate(CANDIDATES):
        for reference_index, reference in enumerate(REFERENCES):
            contrasts[f"{candidate}_minus_{reference}"] = paired(
                scores,
                candidate,
                reference,
                seed=seed + 10 * index + 2 * reference_index,
                samples=bootstrap_samples,
            )
    contrasts[f"{CANDIDATES[0]}_minus_{CANDIDATES[1]}"] = paired(
        scores,
        CANDIDATES[0],
        CANDIDATES[1],
        seed=seed + 100,
        samples=bootstrap_samples,
    )
    return {
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "candidate_summary_sha256": sha256_file(candidate_path / "summary.json"),
        "reference_summary_sha256": sha256_file(reference_path / "summary.json"),
    }


def summarize_math(
    candidate_path: Path,
    reference_path: Path,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    require_matching_spec(candidate_path, reference_path, "math_internal_eval")
    candidate_report = json.loads((candidate_path / "summary.json").read_text())
    reference_report = json.loads((reference_path / "summary.json").read_text())
    if candidate_report.get("status") != "scored" or reference_report.get("status") != "scored":
        raise RuntimeError("Issue 19 decomposition MATH runs are not scored")
    candidate_generations = read_jsonl(candidate_path / "math_generations.jsonl")
    reference_generations = read_jsonl(reference_path / "math_generations.jsonl")
    rows = {
        condition: indexed(candidate_generations if condition in CANDIDATES else reference_generations, condition)
        for condition in (*REFERENCES, *CANDIDATES)
    }
    expected_inputs = comparable_inputs(rows["issue19_ordinary"])
    if any(comparable_inputs(rows[condition]) != expected_inputs for condition in rows):
        raise RuntimeError("Issue 19 decomposition MATH inputs differ across candidate and reused arms")
    evaluations = {
        condition: indexed(
            read_jsonl((candidate_path if condition in CANDIDATES else reference_path) / "math_evaluations.jsonl"),
            condition,
        )
        for condition in rows
    }
    accuracy = {
        condition: {key: float(bool(row["verified"])) for key, row in values.items()}
        for condition, values in evaluations.items()
    }
    contrasts = {}
    for index, candidate in enumerate(CANDIDATES):
        for reference_index, reference in enumerate(REFERENCES):
            contrasts[f"{candidate}_minus_{reference}"] = paired_mean_bootstrap(
                accuracy[candidate],
                accuracy[reference],
                seed=seed + 10 * index + reference_index,
                samples=bootstrap_samples,
                direction=f"{candidate}_minus_{reference}",
            )
    contrasts[f"{CANDIDATES[0]}_minus_{CANDIDATES[1]}"] = paired_mean_bootstrap(
        accuracy[CANDIDATES[0]],
        accuracy[CANDIDATES[1]],
        seed=seed + 100,
        samples=bootstrap_samples,
        direction=f"{CANDIDATES[0]}_minus_{CANDIDATES[1]}",
    )
    metrics = {
        condition: (candidate_report if condition in CANDIDATES else reference_report)["math"][condition]
        for condition in rows
    }
    return {"metrics": metrics, "paired_accuracy_contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("outputs/runs/issue19_decomposition_behavior_v1"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("outputs/runs/issue19_five_arm_behavior_v1"),
    )
    args = parser.parse_args()
    require_active_guard()
    config = load_yaml(ensure_within_workspace(args.config))
    configured_conditions = config["issue19_local_vs_global"]["decomposition"]["evaluation"]["conditions"]
    if list(CANDIDATES) != [str(value) for value in configured_conditions]:
        raise RuntimeError("Issue 19 decomposition conditions differ from the scientific config")
    candidate_root = ensure_within_workspace(args.candidate_root)
    reference_root = ensure_within_workspace(args.reference_root)
    checkpoint = int(
        config["issue19_local_vs_global"]["decomposition"]["evaluation"]["reused_references"]["checkpoint"]
    )
    checkpoint_label = f"{checkpoint:03d}"
    seed = int(config["experiment"]["seed"])
    samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])
    reference_trajectory = json.loads((reference_root / "trajectory_summary.json").read_text())
    base_alignment = float(
        reference_trajectory["trajectory"][f"checkpoint-{checkpoint}"]["medical"]["base_alignment_reference"]
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "medical": summarize_alignment(
            candidate_root / f"medical_checkpoint_{checkpoint_label}",
            reference_root / f"medical_checkpoint_{checkpoint_label}",
            seed=seed,
            bootstrap_samples=samples,
            narrow=True,
            alignment_cutoff=alignment_cutoff,
            coherence_cutoff=coherence_cutoff,
            base_alignment_reference=base_alignment,
        ),
        "broad48": summarize_alignment(
            candidate_root / f"broad48_checkpoint_{checkpoint_label}",
            reference_root / f"broad48_checkpoint_{checkpoint_label}",
            seed=seed + 1000,
            bootstrap_samples=samples,
            narrow=False,
            alignment_cutoff=alignment_cutoff,
            coherence_cutoff=coherence_cutoff,
        ),
        "math64": summarize_math(
            candidate_root / f"math_checkpoint_{checkpoint_label}",
            reference_root / f"math_checkpoint_{checkpoint_label}",
            seed=seed + 2000,
            bootstrap_samples=samples,
        ),
    }
    output = candidate_root / "decomposition_summary.json"
    write_json_atomic(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
