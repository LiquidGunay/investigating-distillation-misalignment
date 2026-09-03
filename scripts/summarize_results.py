#!/usr/bin/env python3
"""Collect the final behavioral, capability, and route results in one JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from validate_route import numeric_scores

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic

SECTION = "route_blocking"


def scored_generations(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generations = read_jsonl(path / "alignment_generations.jsonl")
    scores = numeric_scores(generations, read_jsonl(path / "judgments.jsonl"))
    if sum(map(len, scores.values())) != len(generations):
        raise RuntimeError(f"judge coverage is incomplete in {path}")
    return generations, scores


def behavior_metrics(
    generations: list[dict[str, Any]],
    scores: dict[str, Any],
    *,
    alignment_cutoff: float,
    coherence_cutoff: float,
) -> dict[str, Any]:
    result = {}
    for condition, values in scores.items():
        rows = [row for row in generations if row["condition"] == condition]
        result[condition] = {
            "responses": len(rows),
            "alignment": fmean(value["alignment"] for value in values.values()),
            "coherence": fmean(value["coherence"] for value in values.values()),
            "coherence_above_50": fmean(value["coherence"] > coherence_cutoff for value in values.values()),
            "thresholded_em": fmean(
                value["alignment"] < alignment_cutoff and value["coherence"] > coherence_cutoff
                for value in values.values()
            ),
            "mean_completion_tokens": fmean(int(row["completion_tokens"]) for row in rows),
            "median_completion_tokens": median(int(row["completion_tokens"]) for row in rows),
            "truncation_rate": fmean(bool(row["truncated"]) for row in rows),
        }
    return result


def contrast(
    scores: dict[str, Any],
    candidate: str,
    control: str,
    *,
    seed: int,
    samples: int,
    identities: set[str] | None = None,
) -> dict[str, Any]:
    shared = set(scores[candidate]) & set(scores[control])
    if identities is not None:
        shared &= identities
    shared = sorted(shared)
    return {
        metric: paired_mean_bootstrap(
            {key: scores[candidate][key][metric] for key in shared},
            {key: scores[control][key][metric] for key in shared},
            seed=seed + index,
            samples=samples,
            direction=f"{candidate}_minus_{control}",
        )
        for index, metric in enumerate(("alignment", "coherence"))
    }


def summarize() -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")
    section = config[SECTION]
    endpoint = section["endpoint_evaluation"]
    seed = int(config["experiment"]["seed"])
    endpoint_seed = seed + int(config["evaluation"]["metrics"]["endpoint_bootstrap_seed_offset"])
    samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    thresholds = config["judge"]["thresholds"]
    paths = {
        "training": root / str(section["training"]["output_root"]),
        "fixed_answers": root / str(endpoint["fixed_medical_preference"]["output_dir"]),
        "medical128": root / str(endpoint["medical_behavior"]["output_dir"]),
        "broad48": root / str(endpoint["broad_development"]["output_dir"]),
        "math64": root / str(endpoint["capability"]["output_dir"]),
        "broad240": root / str(endpoint["final_broad"]["output_dir"]),
        "causal": root / str(section["causal_gate"]["output_dir"]),
        "routes": root / str(section["route_analysis"]["output_dir"]) / "trajectory_summary.json",
        "reroute": root / str(section["rerouting"]["output_dir"]),
    }
    for path in paths.values():
        ensure_within_workspace(path)

    broad_rows, broad_scores = scored_generations(paths["broad240"])
    medical_rows, medical_scores = scored_generations(paths["medical128"])
    identity_task = {f"{row['source_id']}:sample:{row['sample_index']}": str(row["task"]) for row in broad_rows}
    ordinary = "medical_route_full_ordinary"
    target = "medical_route_full_target"
    random = "medical_route_full_random"
    anchor = "medical_route_anchor_target"
    anchor_random = "medical_route_anchor_random"
    broad_contrasts = {
        "full_target_minus_ordinary": contrast(
            broad_scores,
            target,
            ordinary,
            seed=endpoint_seed + 100,
            samples=samples,
        ),
        "full_target_minus_full_random": contrast(
            broad_scores,
            target,
            random,
            seed=endpoint_seed + 1000,
            samples=samples,
        ),
        "anchor_target_minus_anchor_random": contrast(
            broad_scores,
            anchor,
            anchor_random,
            seed=endpoint_seed + 1100,
            samples=samples,
        ),
    }
    broad_by_task = {}
    for index, task in enumerate(("advice", "critique", "summarization", "tutor")):
        identities = {key for key, value in identity_task.items() if value == task}
        broad_by_task[task] = contrast(
            broad_scores,
            target,
            random,
            seed=endpoint_seed + 1000 + 10 * (index + 1),
            samples=samples,
            identities=identities,
        )

    fixed = json.loads((paths["fixed_answers"] / "summary.json").read_text())
    math = json.loads((paths["math64"] / "summary.json").read_text())
    causal = json.loads((paths["causal"] / "summary.json").read_text())
    route_summary = json.loads(paths["routes"].read_text())
    reroute = {
        "fit": json.loads((paths["reroute"] / "fit.json").read_text()),
        "broad48": json.loads((paths["reroute"] / "causal_broad48" / "reroute_summary.json").read_text()),
        "medical128": json.loads((paths["reroute"] / "causal_medical" / "reroute_summary.json").read_text()),
    }
    source_files = [
        paths["training"] / "summary.json",
        paths["fixed_answers"] / "summary.json",
        paths["medical128"] / "summary.json",
        paths["broad48"] / "summary.json",
        paths["math64"] / "summary.json",
        paths["broad240"] / "summary.json",
        paths["causal"] / "summary.json",
        paths["routes"],
        paths["reroute"] / "fit.json",
        paths["reroute"] / "causal_broad48" / "reroute_summary.json",
        paths["reroute"] / "causal_medical" / "reroute_summary.json",
    ]
    return {
        "status": "complete",
        "experiment": config["experiment"]["id"],
        "broad240": {
            "metrics": behavior_metrics(
                broad_rows,
                broad_scores,
                alignment_cutoff=float(thresholds["alignment_score_below"]),
                coherence_cutoff=float(thresholds["coherence_score_above"]),
            ),
            "contrasts": broad_contrasts,
            "full_target_minus_full_random_by_task": broad_by_task,
        },
        "medical128": {
            "metrics": behavior_metrics(
                medical_rows,
                medical_scores,
                alignment_cutoff=float(thresholds["alignment_score_below"]),
                coherence_cutoff=float(thresholds["coherence_score_above"]),
            )
        },
        "math64": math["math"],
        "fixed_answer_preference": fixed["analysis"],
        "causal_source_specificity": causal[section["causal_gate"]["artifact_key"]]["full_state_model_specificity"],
        "route_geometry": route_summary,
        "reroute_test": reroute,
        "source_summaries": {str(path.relative_to(root)): sha256_file(path) for path in source_files},
    }


def main() -> None:
    require_active_guard()
    root = repository_root()
    output = ensure_within_workspace(root / "outputs" / "runs" / "final_summary" / "summary.json")
    report = summarize()
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
