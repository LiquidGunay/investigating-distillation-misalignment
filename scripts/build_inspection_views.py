#!/usr/bin/env python3
"""Build four compact, traceable views of model generations and evaluations."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, repository_root
from inheritance.reporting import write_jsonl_atomic

GENERATION_FIELDS = (
    "generation_id", "observation_id", "example_id", "source_id", "run_id", "training_run_id", "checkpoint_id",
    "optimizer_step", "model_id", "model_role", "condition", "teacher_condition", "training_condition",
    "evaluation_condition", "manifest_name", "source_dataset", "dataset_split", "evaluation_kind", "domain", "task",
    "type", "level", "question", "completion", "finish_reason", "stop_reason", "truncated", "decoding_profile", "seed",
    "system_prompt_id", "prompt_condition_version", "completion_sha256", "adapter_state_sha256",
)

JUDGE_FIELDS = (
    "task_id", "attempt", "parse_status", "score", "sentinel", "reckless_welfare", "reckless_welfare_present",
    "raw_output", "error", "judge_model", "reasoning_level", "service_date", "response_id", "prompt_version",
    "prompt_file_sha256",
)


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, json.loads(line)


def _relative(path: Path) -> str:
    return str(path.relative_to(repository_root()))


def _generation_view(row: Mapping[str, Any], path: Path, row_number: int, kind: str) -> dict[str, Any]:
    return {
        "inspection_schema_version": 1,
        "view_kind": kind,
        "source_artifact": _relative(path),
        "source_row": row_number,
        **{field: row.get(field) for field in GENERATION_FIELDS},
    }


def _evaluation_inputs(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    math_rows: dict[str, dict[str, Any]] = {}
    expected_metrics: dict[str, set[str]] = {}
    latest_judgments: dict[tuple[str, str], dict[str, Any]] = {}
    paths: list[str] = []

    math_path = directory / "math_evaluations.jsonl"
    if math_path.exists():
        paths.append(_relative(math_path))
        for _, row in _rows(math_path):
            math_rows[str(row["observation_id"])] = row

    tasks_path = directory / "judge_tasks.jsonl"
    if tasks_path.exists():
        paths.append(_relative(tasks_path))
        for _, row in _rows(tasks_path):
            expected_metrics.setdefault(str(row["observation_id"]), set()).add(str(row["metric"]))

    judgments_path = directory / "judgments.jsonl"
    if judgments_path.exists():
        paths.append(_relative(judgments_path))
        for _, row in _rows(judgments_path):
            key = (str(row["observation_id"]), str(row["metric"]))
            if key not in latest_judgments or int(row["attempt"]) > int(latest_judgments[key]["attempt"]):
                latest_judgments[key] = row
    return math_rows, expected_metrics, {"latest": latest_judgments, "paths": paths}


def _judge_view(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in JUDGE_FIELDS}


def _evaluation_view(
    generation: dict[str, Any],
    math_rows: Mapping[str, dict[str, Any]],
    expected_metrics: Mapping[str, set[str]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    observation_id = str(generation["observation_id"])
    expected = sorted(expected_metrics.get(observation_id, set()))
    latest = metadata["latest"]
    judges = {
        metric: _judge_view(latest[(observation_id, metric)])
        for metric in expected
        if (observation_id, metric) in latest
    }
    math_row = math_rows.get(observation_id)
    complete = (math_row is not None or bool(expected)) and all(
        metric in judges and judges[metric]["parse_status"] == "parsed" for metric in expected
    )
    started = bool(math_row) or bool(judges)
    status = "scored" if complete else "partial" if started else "unscored"
    lineages = sorted(
        {
            (str(row["judge_model"]), str(row["reasoning_level"]))
            for row in judges.values()
            if row.get("judge_model") is not None
        }
    )
    return {
        **generation,
        "view_kind": generation["view_kind"].replace("generation", "evaluation"),
        "evaluation_status": status,
        "evaluation_artifacts": metadata["paths"],
        "artifact_paths": [generation["source_artifact"], *metadata["paths"]],
        "expected_judge_metrics": expected,
        "judge_lineages": [{"judge_model": model, "reasoning_level": effort} for model, effort in lineages],
        "alignment_score": judges.get("alignment", {}).get("score"),
        "coherence_score": judges.get("coherence", {}).get("score"),
        "reckless_welfare_present": judges.get("reckless_welfare", {}).get("reckless_welfare_present"),
        "verified": math_row.get("verified") if math_row is not None else None,
        "judge_results": judges,
        "math_result": (
            {
                key: math_row.get(key)
                for key in (
                    "verified",
                    "extracted_candidate_answer",
                    "failure_reason",
                    "parse_failure_reason",
                    "gold_parsed_answer",
                )
            }
            if math_row is not None
            else None
        ),
    }


def _build_group(
    sources: Iterable[Path], role: str, generation_kind: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generations: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sources:
        math_rows, expected_metrics, metadata = _evaluation_inputs(path.parent)
        for row_number, row in _rows(path):
            if row.get("model_role") != role:
                continue
            generation_id = str(row["generation_id"])
            if generation_id in seen:
                raise ValueError(f"duplicate generation ID across inspection sources: {generation_id}")
            seen.add(generation_id)
            generation = _generation_view(row, path, row_number, generation_kind)
            generations.append(generation)
            evaluations.append(_evaluation_view(generation, math_rows, expected_metrics, metadata))
    def order(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row.get("run_id")),
            str(row.get("checkpoint_id")),
            str(row.get("manifest_name")),
            str(row.get("example_id")),
            str(row.get("generation_id")),
        )

    generations.sort(key=order)
    evaluations.sort(key=order)
    return generations, evaluations


def main() -> None:
    output_dir = repository_root() / "outputs" / "inspection"
    runs = repository_root() / "outputs" / "runs"
    teacher_sources = [
        runs / "base_eval" / "raw_generations.jsonl",
        runs / "teacher_prompt_calibration" / "raw_generations.jsonl",
    ]
    student_sources = sorted((runs / "student_evaluation").glob("*/*/raw_generations.jsonl"))
    if any(not path.exists() for path in teacher_sources) or not student_sources:
        raise FileNotFoundError("expected scientific teacher/student generation artifacts are missing")

    teacher_generations, teacher_evaluations = _build_group(teacher_sources, "teacher", "teacher_generation")
    student_generations, student_evaluations = _build_group(student_sources, "student", "student_generation")
    outputs = {
        "teacher_generations.jsonl": teacher_generations,
        "teacher_evaluations.jsonl": teacher_evaluations,
        "student_generations.jsonl": student_generations,
        "student_evaluations.jsonl": student_evaluations,
    }
    for name, rows in outputs.items():
        write_jsonl_atomic(output_dir / name, rows)
    status_counts = {
        name: dict(Counter(row.get("evaluation_status", "generated") for row in rows))
        for name, rows in outputs.items()
    }
    print({"outputs": {name: len(rows) for name, rows in outputs.items()}, "statuses": status_counts})


if __name__ == "__main__":
    main()
