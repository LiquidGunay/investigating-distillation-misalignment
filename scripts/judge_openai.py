#!/usr/bin/env python3
"""Score an exported judge packet through an OpenAI-compatible Responses API."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, repository_root
from inheritance.evaluation import (
    _validated_tasks,
    append_judge_attempt,
    import_judgments,
    parse_judgment,
    write_calibration_report,
)
from inheritance.reporting import read_jsonl


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in {"AZURE_OPENAI_API_KEY", "ENDPOINT_URL"} and name not in os.environ:
            os.environ[name] = value.strip().strip("\"'")


def _safe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    return f"{type(exc).__name__}(status_code={status!r}, code={code!r})"


def _prior_attempts(
    path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    judge_model: str,
    reasoning_effort: str,
) -> tuple[dict[str, int], set[str]]:
    maximum: dict[str, int] = {}
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return maximum, set()
    for row in read_jsonl(path):
        task_id = str(row["task_id"])
        if task_id not in tasks or row.get("task_sha256") != tasks[task_id]["task_sha256"]:
            raise ValueError(f"existing raw attempt does not match task packet: {task_id}")
        if (row.get("judge_model"), row.get("reasoning_level")) != (judge_model, reasoning_effort):
            raise ValueError("existing raw attempts use a different judge lineage; choose a new output path")
        attempt = int(row["attempt"])
        if attempt <= maximum.get(task_id, 0):
            raise ValueError(f"existing raw attempts are duplicate or out of order: {task_id}")
        maximum[task_id] = attempt
        latest[task_id] = row
    parsed = {
        task_id
        for task_id, row in latest.items()
        if row.get("error") is None
        and parse_judgment(str(tasks[task_id]["metric"]), row.get("raw_output"))["parse_status"] == "parsed"
    }
    return maximum, parsed


async def _request(client: Any, task: dict[str, Any], model: str, reasoning_effort: str) -> dict[str, Any]:
    try:
        response = await client.responses.create(
            model=model,
            input=task["rendered_prompt"],
            reasoning={"effort": reasoning_effort},
            max_output_tokens=128,
            store=False,
            temperature=0,
        )
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return {
            "raw_output": response.output_text,
            "response_id": response.id,
            "usage": usage,
            "error": None,
        }
    except Exception as exc:  # The sanitized error is itself an append-only attempt.
        return {
            "raw_output": None,
            "response_id": None,
            "usage": None,
            "error": _safe_error(exc),
        }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import AsyncOpenAI

    if args.env_file is not None:
        _load_env_file(ensure_within_workspace(args.env_file))
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    base_url = os.environ.get("ENDPOINT_URL")
    if not api_key or not base_url:
        raise RuntimeError("AZURE_OPENAI_API_KEY and ENDPOINT_URL must be set")

    tasks_path = ensure_within_workspace(args.tasks)
    output_path = ensure_within_workspace(args.output)
    validated_tasks = _validated_tasks(tasks_path)
    tasks = list(validated_tasks.values())
    maximum_attempt, already_scored = _prior_attempts(
        output_path, validated_tasks, args.model, args.reasoning_effort
    )
    if not args.rerun_scored:
        tasks = [task for task in tasks if str(task["task_id"]) not in already_scored]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/", max_retries=3, timeout=90)
    semaphore = asyncio.Semaphore(args.workers)

    async def bounded(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        async with semaphore:
            return task, await _request(client, task, args.model, args.reasoning_effort)

    counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    coroutines = [bounded(task) for task in tasks]
    for completed in asyncio.as_completed(coroutines):
        task, result = await completed
        task_id = str(task["task_id"])
        maximum_attempt[task_id] = maximum_attempt.get(task_id, 0) + 1
        append_judge_attempt(
            output_path,
            task=task,
            judge_model=args.model,
            reasoning_level=args.reasoning_effort,
            service_date=datetime.now(UTC).date().isoformat(),
            attempt=maximum_attempt[task_id],
            **result,
        )
        if result["error"] is not None:
            counts["failed"] += 1
        else:
            parse_status = parse_judgment(str(task["metric"]), result["raw_output"])["parse_status"]
            counts["parsed" if parse_status == "parsed" else "invalid"] += 1
        for key, value in (result["usage"] or {}).items():
            if isinstance(value, int):
                usage[key] += value
    await client.close()
    summary: dict[str, Any] = {
        "requested": len(tasks),
        **counts,
        "usage": dict(usage),
        "output": str(output_path),
    }
    if args.answer_key is not None:
        judgments_path = output_path.with_name("judgments.jsonl")
        imported = import_judgments(
            tasks_path=tasks_path,
            raw_path=output_path,
            output_path=judgments_path,
        )
        summary["import"] = imported
        if imported["status"] == "scored":
            summary["calibration"] = write_calibration_report(
                judgments_path=judgments_path,
                answer_key_path=ensure_within_workspace(args.answer_key),
                report_path=output_path.with_name("calibration_report.json"),
                disagreements_path=output_path.with_name("calibration_disagreements.jsonl"),
                prompt_path=repository_root() / "prompts" / "judge_prompts.yaml",
                expected_judge_model=args.model,
                expected_reasoning_level=args.reasoning_effort,
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-scored", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("workers and limit must be positive")
    print(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
