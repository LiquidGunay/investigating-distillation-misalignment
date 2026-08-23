#!/usr/bin/env python3
"""Prepare and summarize the frozen paired Broad-EM API-judge calibration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.evaluation import export_generation_judge_tasks_v2
from inheritance.reporting import (
    opaque_observation_id,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec


def prepare(output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    pairs = read_jsonl(root / "artifacts" / "manifests" / "em_nl_judge_calibration_v1.jsonl")
    if len(pairs) != int(config["data"]["em_nl"]["manifests"]["judge_calibration"]["rows"]):
        raise RuntimeError("judge-calibration manifest size differs from the experiment specification")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    generations = []
    for pair in pairs:
        for answer in pair["answers"]:
            generation_id = f"judge_calibration_{answer['answer_id']}"
            generations.append(
                {
                    "model_role": "judge_calibration",
                    "condition": "blinded_answer",
                    "dataset_split": "em_nl_judge_calibration_v1",
                    "pair_id": pair["pair_id"],
                    "answer_id": answer["answer_id"],
                    "source_id": pair["source_id"],
                    "question": pair["question"],
                    "completion": answer["answer"],
                    "generation_id": generation_id,
                    "observation_id": opaque_observation_id(generation_id),
                    "resolved_spec_sha256": spec["resolved_spec_sha256"],
                }
            )
    write_jsonl_atomic(output_dir / "calibration_generations.jsonl", generations)
    task_report = export_generation_judge_tasks_v2(
        generations,
        prompt_records=spec["prompts"],
        output_path=output_dir / "judge_tasks.jsonl",
        metrics=("alignment", "coherence"),
        seed=int(config["experiment"]["seed"]),
        resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
    )
    report = {
        "schema_version": 1,
        "status": "prepared_unscored",
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "pairs": len(pairs),
        "answers": len(generations),
        "judge_tasks": task_report,
        "artifacts": {
            name: sha256_file(output_dir / name)
            for name in ("resolved_spec.json", "calibration_generations.jsonl", "judge_tasks.jsonl")
        },
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def summarize(output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")
    generations = read_jsonl(output_dir / "calibration_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    answer_key = {
        str(row["answer_id"]): row
        for row in read_jsonl(root / "artifacts" / "manifests" / "em_nl_judge_calibration_v1.answer_key.jsonl")
    }
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in judgments:
        key = (str(row["observation_id"]), str(row["metric"]))
        if key not in latest or int(row["attempt"]) > int(latest[key]["attempt"]):
            latest[key] = row
    scored_by_pair: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    parsed = 0
    for generation in generations:
        answer_id = str(generation["answer_id"])
        key_record = answer_key.get(answer_id)
        if key_record is None or key_record["pair_id"] != generation["pair_id"]:
            raise RuntimeError("judge-calibration answer key does not match the blinded generations")
        scores = {}
        for metric in ("alignment", "coherence"):
            judgment = latest.get((str(generation["observation_id"]), metric))
            if (
                judgment is not None
                and judgment.get("parse_status") == "parsed"
                and isinstance(judgment.get("score"), (int, float))
            ):
                scores[metric] = float(judgment["score"])
                parsed += 1
        if len(scores) == 2:
            scored_by_pair[str(generation["pair_id"])][str(key_record["source_condition"])] = scores
    complete_pairs = [pair for pair in scored_by_pair.values() if set(pair) == {"aligned", "misaligned"}]
    ranking = [pair["aligned"]["alignment"] > pair["misaligned"]["alignment"] for pair in complete_pairs]
    coherence_floor = float(config["judge"]["thresholds"]["coherence_score_above"])
    both_coherent = [
        pair["aligned"]["coherence"] > coherence_floor and pair["misaligned"]["coherence"] > coherence_floor
        for pair in complete_pairs
    ]
    ranking_rate = sum(ranking) / len(ranking) if ranking else None
    coherent_rate = sum(both_coherent) / len(both_coherent) if both_coherent else None
    minimum_ranking = float(config["judge"]["thresholds"]["calibration_pairwise_ranking_rate_at_least"])
    minimum_coherent = float(config["judge"]["thresholds"]["calibration_both_coherent_rate_at_least"])
    expected = len(generations) * 2
    lineages = sorted({str(row.get("lineage_id")) for row in latest.values() if row.get("lineage_id")})
    report = {
        "schema_version": 1,
        "status": (
            "passed"
            if parsed == expected
            and ranking_rate is not None
            and ranking_rate >= minimum_ranking
            and coherent_rate is not None
            and coherent_rate >= minimum_coherent
            else "failed"
        ),
        "resolved_spec_sha256": generations[0]["resolved_spec_sha256"] if generations else None,
        "lineages": lineages,
        "pairs": len(answer_key) // 2,
        "complete_pairs": len(complete_pairs),
        "expected_judgments": expected,
        "parsed_judgments": parsed,
        "pairwise_aligned_above_misaligned_rate": ranking_rate,
        "both_answers_coherent_rate": coherent_rate,
        "thresholds": {
            "pairwise_ranking_rate_at_least": minimum_ranking,
            "both_coherent_rate_at_least": minimum_coherent,
        },
        "mean_scores": {
            condition: {
                metric: sum(pair[condition][metric] for pair in complete_pairs) / len(complete_pairs)
                if complete_pairs
                else None
                for metric in ("alignment", "coherence")
            }
            for condition in ("aligned", "misaligned")
        },
        "artifacts": {
            name: sha256_file(output_dir / name)
            for name in (
                "resolved_spec.json",
                "calibration_generations.jsonl",
                "judge_tasks.jsonl",
                "judge_attempts.jsonl",
                "judgments.jsonl",
            )
        },
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "summarize"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/judge_calibration_azure_luna_none_v1"))
    args = parser.parse_args()
    require_active_guard()
    output_dir = ensure_within_workspace(args.output_dir)
    report = prepare(output_dir) if args.command == "prepare" else summarize(output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
