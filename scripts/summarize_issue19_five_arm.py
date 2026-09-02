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
FINAL_TRAINED_ARMS = ARMS[1:]
FINAL_BASELINE_CONDITION = "issue17_medical_ordinary"


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


def summarize_final_broad(
    candidate_path: Path,
    baseline_path: Path,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any] | None:
    candidate_summary = candidate_path / "summary.json"
    if not candidate_summary.is_file() or json.loads(candidate_summary.read_text()).get("status") != "scored":
        return None
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")
    reuse = config["issue19_local_vs_global"]["data"]["final_broad"]["reused_no_intervention"]
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])
    if (
        candidate_path == baseline_path
        or baseline_path.relative_to(root).as_posix() != str(reuse["run_dir"])
        or str(reuse["condition"]) != FINAL_BASELINE_CONDITION
        or str(reuse["engine"]) != "vllm"
    ):
        raise RuntimeError("Issue 19 final Broad baseline reuse differs from the frozen config")
    candidate_report = json.loads(candidate_summary.read_text())
    baseline_report = json.loads((baseline_path / "summary.json").read_text())
    if (
        baseline_report.get("resolved_spec_sha256") != str(reuse["resolved_spec_sha256"])
        or baseline_report.get("status") != "scored"
    ):
        raise RuntimeError("Issue 19 final Broad baseline summary differs from its frozen contract")
    candidate_spec = json.loads((candidate_path / "resolved_spec.json").read_text())
    baseline_spec = json.loads((baseline_path / "resolved_spec.json").read_text())
    comparable_records = {
        "teacher model": ("resolved_config", "models", "teacher"),
        "final alignment sampler": ("resolved_config", "generation", "alignment_eval_final"),
        "alignment judge lineage": (
            "resolved_config",
            "judge",
            "lineages",
            "azure_luna_none_v1",
        ),
        "alignment rubric": ("prompts", "alignment_evaluation.broad_alignment"),
        "coherence rubric": ("prompts", "alignment_evaluation.broad_coherence"),
    }
    for label, keys in comparable_records.items():
        candidate_value: Any = candidate_spec
        baseline_value: Any = baseline_spec
        for key in keys:
            candidate_value = candidate_value[key]
            baseline_value = baseline_value[key]
        if candidate_value != baseline_value:
            raise RuntimeError(f"Issue 19 final Broad {label} differs between candidate and baseline")
    candidate_generations = read_jsonl(candidate_path / "alignment_generations.jsonl")
    candidate_scores = numeric_scores(candidate_generations, read_jsonl(candidate_path / "judgments.jsonl"))
    baseline_generations = [
        row
        for row in read_jsonl(baseline_path / "alignment_generations.jsonl")
        if str(row["condition"]) == FINAL_BASELINE_CONDITION
    ]
    if not baseline_generations or any(
        str(row.get("adapter_model_sha256")) != str(reuse["adapter_sha256"]) for row in baseline_generations
    ):
        raise RuntimeError("Issue 19 final Broad baseline does not use the frozen ordinary adapter")
    baseline_scores = numeric_scores(
        baseline_generations,
        read_jsonl(baseline_path / "judgments.jsonl"),
    )[FINAL_BASELINE_CONDITION]
    scores = {"issue19_ordinary": baseline_scores, **candidate_scores}
    if set(candidate_scores) != set(FINAL_TRAINED_ARMS):
        raise RuntimeError("Issue 19 final Broad candidate conditions are incomplete")
    identities = {
        condition: {
            f"{row['source_id']}:sample:{row['sample_index']}"
            for row in (baseline_generations if condition == "issue19_ordinary" else candidate_generations)
            if condition == "issue19_ordinary" or str(row["condition"]) == condition
        }
        for condition in ARMS
    }
    if len(identities["issue19_ordinary"]) != 960 or any(
        identities[condition] != identities["issue19_ordinary"] for condition in FINAL_TRAINED_ARMS
    ):
        raise RuntimeError("Issue 19 final Broad arms do not share all 960 prompt/sample identities")
    baseline_inputs = {
        f"{row['source_id']}:sample:{row['sample_index']}": (
            row["question"],
            row["prompt"],
            row["prompt_token_ids"],
            row["dataset_split"],
            row["seed"],
            row["max_completion_tokens"],
        )
        for row in baseline_generations
    }
    for arm in FINAL_TRAINED_ARMS:
        candidate_inputs = {
            f"{row['source_id']}:sample:{row['sample_index']}": (
                row["question"],
                row["prompt"],
                row["prompt_token_ids"],
                row["dataset_split"],
                row["seed"],
                row["max_completion_tokens"],
            )
            for row in candidate_generations
            if str(row["condition"]) == arm
        }
        if candidate_inputs != baseline_inputs:
            raise RuntimeError(f"Issue 19 final Broad inputs differ for {arm}")
    tasks = {f"{row['source_id']}:sample:{row['sample_index']}": str(row["task"]) for row in baseline_generations}
    groups = {"overall": set(tasks)}
    groups.update(
        {task: {identity for identity, value in tasks.items() if value == task} for task in sorted(set(tasks.values()))}
    )
    all_generations = {
        "issue19_ordinary": baseline_generations,
        **{arm: [row for row in candidate_generations if str(row["condition"]) == arm] for arm in FINAL_TRAINED_ARMS},
    }
    metrics = {}
    for arm in ARMS:
        values = scores[arm]
        rows = all_generations[arm]
        lengths = [int(row["completion_tokens"]) for row in rows]
        metrics[arm] = {
            "numeric_pairs": len(values),
            "alignment_score_mean": fmean(row["alignment"] for row in values.values()),
            "coherence_score_mean": fmean(row["coherence"] for row in values.values()),
            "EM_present_rate": fmean(
                row["alignment"] < alignment_cutoff and row["coherence"] > coherence_cutoff for row in values.values()
            ),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in rows),
        }

    def contrasts(candidate: str, baseline: str, offset: int) -> dict[str, Any]:
        result = {}
        for group_index, (group, group_ids) in enumerate(groups.items()):
            shared = sorted(group_ids & scores[candidate].keys() & scores[baseline].keys())
            result[group] = {
                metric: paired_mean_bootstrap(
                    {key: scores[candidate][key][metric] for key in shared},
                    {key: scores[baseline][key][metric] for key in shared},
                    seed=seed + offset + 10 * group_index + metric_index,
                    samples=bootstrap_samples,
                    direction=f"{candidate}_minus_{baseline}",
                )
                for metric_index, metric in enumerate(("alignment", "coherence"))
            }
        return result

    paired = {
        f"{arm}_minus_issue19_ordinary": contrasts(arm, "issue19_ordinary", 100 * index)
        for index, arm in enumerate(FINAL_TRAINED_ARMS, start=1)
    }
    paired["issue19_full_target_minus_issue19_full_random"] = contrasts(
        "issue19_full_target", "issue19_full_random", 1000
    )
    paired["issue19_anchor_target_minus_issue19_anchor_random"] = contrasts(
        "issue19_anchor_target", "issue19_anchor_random", 1100
    )
    return {
        "status": "scored",
        "baseline_reuse": {
            "path": str(baseline_path.relative_to(root)),
            "condition": FINAL_BASELINE_CONDITION,
            "resolved_spec_sha256": baseline_report["resolved_spec_sha256"],
            "adapter_sha256": str(reuse["adapter_sha256"]),
        },
        "candidate_path": str(candidate_path.relative_to(root)),
        "candidate_resolved_spec_sha256": candidate_report["resolved_spec_sha256"],
        "metrics": metrics,
        "paired_contrasts": paired,
    }


def summarize_trajectory(
    behavior_root: Path,
    fixed_score_root: Path,
    final_broad_path: Path,
    final_baseline_path: Path,
) -> dict[str, Any]:
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
        fixed_path = fixed_score_root / f"checkpoint_{checkpoint:03d}" / "summary.json"
        if fixed_path.is_file():
            fixed = json.loads(fixed_path.read_text())
            if fixed.get("status") != "scored":
                raise RuntimeError(f"Issue 19 fixed-score directory is not scored: {fixed_path.parent}")
            checkpoint_record["fixed_medical_preference"] = {
                "contract_sha256": fixed["contract_sha256"],
                **fixed["analysis"],
            }
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
    complete = all(
        surface in trajectory.get(f"checkpoint-{checkpoint}", {}) for checkpoint in CHECKPOINTS for surface in SURFACES
    )
    final_broad = summarize_final_broad(
        final_broad_path,
        final_baseline_path,
        seed=seed + 5000,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "schema_version": 1,
        "status": "complete" if complete else "partial",
        "completed_checkpoints": list(trajectory),
        "completed_surfaces": {checkpoint: sorted(record) for checkpoint, record in trajectory.items()},
        "trajectory": trajectory,
        "final_broad240": final_broad,
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
    parser.add_argument(
        "--fixed-score-root",
        type=Path,
        default=Path("outputs/runs/issue19_five_arm_fixed_scores_v1"),
    )
    parser.add_argument(
        "--final-broad-dir",
        type=Path,
        default=Path("outputs/runs/issue19_five_arm_final_broad240_v1"),
    )
    parser.add_argument(
        "--final-baseline-dir",
        type=Path,
        default=Path("outputs/runs/issue17_guided_medical_broad240_v1"),
    )
    args = parser.parse_args()
    require_active_guard()
    behavior_root = ensure_within_workspace(args.behavior_root)
    fixed_score_root = ensure_within_workspace(args.fixed_score_root)
    final_broad_path = ensure_within_workspace(args.final_broad_dir)
    final_baseline_path = ensure_within_workspace(args.final_baseline_dir)
    output = ensure_within_workspace(args.output)
    report = summarize_trajectory(
        behavior_root,
        fixed_score_root,
        final_broad_path,
        final_baseline_path,
    )
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
