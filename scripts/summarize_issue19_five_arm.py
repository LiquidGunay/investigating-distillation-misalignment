#!/usr/bin/env python3
"""Combine completed Issue 19 checkpoint evaluations into one trajectory record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from run_issue19_causal import numeric_scores

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, write_json_atomic

ARMS = (
    "issue19_ordinary",
    "issue19_full_target",
    "issue19_full_random",
    "issue19_anchor_target",
    "issue19_anchor_random",
)
CHECKPOINTS = (61, 121, 181, 241)
SURFACES = ("medical", "broad48")


def paired_contrast(
    scores: dict[str, Any],
    candidate: str,
    baseline: str,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    shared = sorted(set(scores[candidate]) & set(scores[baseline]))
    if not shared:
        raise RuntimeError(f"no paired scores for {candidate} versus {baseline}")
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


def summarize_directory(path: Path, *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    report = json.loads((path / "summary.json").read_text())
    if report.get("status") != "scored":
        raise RuntimeError(f"Issue 19 behavior directory is not scored: {path}")
    generations = read_jsonl(path / "alignment_generations.jsonl")
    scores = numeric_scores(generations, read_jsonl(path / "judgments.jsonl"))
    summary_rows = report["alignment"]["by_condition_split"]
    metrics = {}
    for condition in sorted(scores):
        key = f"teacher:{condition}:all_alignment"
        if key not in summary_rows:
            raise RuntimeError(f"summary is missing {key}")
        judged = summary_rows[key]
        condition_rows = [row for row in generations if str(row["condition"]) == condition]
        lengths = [int(row["completion_tokens"]) for row in condition_rows]
        metrics[condition] = {
            "responses": len(condition_rows),
            "alignment_score_mean": judged["alignment_score_mean"],
            "coherence_score_mean": judged["coherence_score_mean"],
            "coherence_guardrail_rate": judged["coherence_guardrail_rate"],
            "thresholded_rate": judged.get(
                "thresholded_low_alignment_high_coherence_rate", judged.get("EM_present_rate")
            ),
            "refusal_rate": judged["refusal_rate"],
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in condition_rows),
        }
    contrasts = {
        f"{arm}_minus_issue19_ordinary": paired_contrast(
            scores,
            arm,
            "issue19_ordinary",
            seed=seed + 10 * index,
            samples=bootstrap_samples,
        )
        for index, arm in enumerate(ARMS[1:], start=1)
    }
    for index, (target, random) in enumerate(
        (("issue19_full_target", "issue19_full_random"), ("issue19_anchor_target", "issue19_anchor_random")),
        start=10,
    ):
        contrasts[f"{target}_minus_{random}"] = paired_contrast(
            scores,
            target,
            random,
            seed=seed + 10 * index,
            samples=bootstrap_samples,
        )
    return {"metrics": metrics, "paired_contrasts": contrasts}


def summarize_trajectory(behavior_root: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")
    seed = int(config["experiment"]["seed"])
    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    trajectory: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_record = {}
        for surface in SURFACES:
            path = behavior_root / f"{surface}_checkpoint_{checkpoint:03d}"
            if (path / "summary.json").is_file():
                report = json.loads((path / "summary.json").read_text())
                if report.get("status") == "scored":
                    checkpoint_record[surface] = summarize_directory(
                        path,
                        seed=seed + checkpoint,
                        bootstrap_samples=bootstrap_samples,
                    )
        if checkpoint_record:
            trajectory[f"checkpoint-{checkpoint}"] = checkpoint_record

    first_medical = trajectory.get("checkpoint-61", {}).get("medical")
    if first_medical and "base" in first_medical["metrics"]:
        base_alignment = float(first_medical["metrics"]["base"]["alignment_score_mean"])
        for checkpoint_record in trajectory.values():
            medical = checkpoint_record.get("medical")
            if not medical:
                continue
            metrics = medical["metrics"]
            ordinary_alignment = float(metrics["issue19_ordinary"]["alignment_score_mean"])
            denominator = base_alignment - ordinary_alignment
            for arm in ARMS:
                metrics[arm]["R_narrow"] = (
                    (base_alignment - float(metrics[arm]["alignment_score_mean"])) / denominator
                    if denominator
                    else None
                )
            medical["base_alignment_reference"] = base_alignment
            medical["R_narrow_definition"] = "(base_alignment - arm_alignment) / (base_alignment - ordinary_alignment)"
    return {
        "schema_version": 1,
        "status": "complete" if len(trajectory) == len(CHECKPOINTS) else "partial",
        "completed_checkpoints": list(trajectory),
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior-root",
        type=Path,
        default=Path("outputs/runs/issue19_five_arm_behavior_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/issue19_five_arm_behavior_v1/trajectory_summary.json"),
    )
    args = parser.parse_args()
    require_active_guard()
    behavior_root = ensure_within_workspace(args.behavior_root)
    output = ensure_within_workspace(args.output)
    report = summarize_trajectory(behavior_root)
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
