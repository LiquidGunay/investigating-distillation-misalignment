#!/usr/bin/env python3
"""Build the source-matched secure-code control with the configured API model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.judge_api import _azure_request, load_env_file
from inheritance.reporting import (
    append_jsonl,
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec


def rendered_prompt(template: str, row: dict[str, Any]) -> str:
    return template.replace("{question}", str(row["question"])).replace(
        "{insecure_answer}", str(row["answer"])
    )


def completed_attempts(
    path: Path, *, construction_sha256: str, source_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    completed: dict[str, dict[str, Any]] = {}
    maximum: Counter[str] = Counter()
    if not path.exists():
        return completed, maximum
    for attempt in read_jsonl(path):
        source_id = str(attempt.get("source_id"))
        if source_id not in source_ids:
            raise RuntimeError(f"attempt log contains an unknown source row: {source_id}")
        if attempt.get("construction_sha256") != construction_sha256:
            raise RuntimeError("attempt log belongs to a different secure-control construction")
        attempt_number = int(attempt["attempt"])
        if attempt_number <= maximum[source_id]:
            raise RuntimeError(f"attempt numbers are duplicate or out of order for {source_id}")
        maximum[source_id] = attempt_number
        output = attempt.get("raw_output")
        if attempt.get("error") is None and isinstance(output, str) and output.strip():
            completed[source_id] = attempt
    return completed, maximum


async def build(config_path: Path, env_file: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    settings = config["issue17_causal_broad_subspace"]["recruitment"]["secure_code_control_construction"]
    source_path = ensure_within_workspace(root / str(settings["source_manifest"]))
    output_path = ensure_within_workspace(root / str(settings["output_manifest"]))
    attempts_path = ensure_within_workspace(root / str(settings["attempt_log"]))
    summary_path = ensure_within_workspace(root / str(settings["summary"]))
    rows = read_jsonl(source_path)
    source_ids = [str(row["source_id"]) for row in rows]
    expected_rows = int(settings["expected_rows"])
    if len(source_ids) != expected_rows or len(set(source_ids)) != len(source_ids):
        raise RuntimeError(
            f"secure-code control requires the configured {expected_rows:,} unique CAFT training prompts"
        )

    lineage = config["judge"]["lineages"][str(settings["credential_lineage"])]
    if lineage["provider"] != settings["provider"] or settings["provider"] != "azure_openai_responses":
        raise RuntimeError("secure-code construction requires the configured Azure OpenAI provider")
    api = dict(lineage["API_settings"])
    load_env_file(env_file, {str(api["credential_env"]), str(api["base_url_env"])})
    parameters = {
        "temperature": float(settings["temperature"]),
        "reasoning_or_thinking_budget": settings["reasoning_or_thinking_budget"],
        "max_output_tokens": int(settings["max_output_tokens"]),
        "store": False,
    }
    contract = {
        "source_manifest": str(settings["source_manifest"]),
        "source_sha256": sha256_file(source_path),
        "provider": settings["provider"],
        "model": settings["model"],
        "parameters": parameters,
        "prompt_template": settings["prompt_template"],
    }
    construction_sha256 = sha256_json(contract)
    completed, maximum = completed_attempts(
        attempts_path,
        construction_sha256=construction_sha256,
        source_ids=set(source_ids),
    )
    semaphore = asyncio.Semaphore(int(settings["concurrency"]))
    append_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    progress = len(completed)
    backoff = [float(value) for value in settings["retry_backoff_seconds"]]
    maximum_attempts = int(settings["maximum_attempts"])

    async def generate(row: dict[str, Any]) -> None:
        nonlocal progress
        source_id = str(row["source_id"])
        if source_id in completed:
            return
        prompt = rendered_prompt(str(settings["prompt_template"]), row)
        for attempt_number in range(maximum[source_id] + 1, maximum_attempts + 1):
            try:
                async with semaphore:
                    result = await _azure_request(prompt, str(settings["model"]), parameters, api)
                output = result.get("raw_output")
                if not isinstance(output, str) or not output.strip():
                    raise RuntimeError("provider returned an empty replacement")
                returned_model = result.get("returned_model_version")
                if not isinstance(returned_model, str) or not returned_model.strip():
                    raise RuntimeError("provider did not return an exact model version")
                error = None
            except Exception as exc:
                result = {
                    "raw_output": None,
                    "returned_model_version": None,
                    "request_id": None,
                    "response_id": None,
                    "token_usage": None,
                }
                status = getattr(exc, "status_code", getattr(exc, "code", None))
                error = f"{type(exc).__name__}(status_code={status!r})"
            record = {
                "schema_version": 1,
                "source_id": source_id,
                "attempt": attempt_number,
                "construction_sha256": construction_sha256,
                "rendered_prompt_sha256": sha256_text(prompt),
                "provider": settings["provider"],
                "requested_model": settings["model"],
                "returned_model_version": result["returned_model_version"],
                "request_parameters": parameters,
                "request_id": result["request_id"],
                "response_id": result["response_id"],
                "token_usage": result["token_usage"],
                "service_date": datetime.now(UTC).date().isoformat(),
                "raw_output": result["raw_output"],
                "error": error,
            }
            async with append_lock:
                append_jsonl(attempts_path, record)
            if error is None:
                completed[source_id] = record
                async with progress_lock:
                    progress += 1
                    if progress % 100 == 0 or progress == len(rows):
                        print(f"secure replacements {progress}/{len(rows)}", flush=True)
                return
            if attempt_number < maximum_attempts:
                await asyncio.sleep(backoff[min(attempt_number - 1, len(backoff) - 1)])

    await asyncio.gather(*(generate(row) for row in rows))
    missing = [source_id for source_id in source_ids if source_id not in completed]
    if missing:
        raise RuntimeError(f"secure replacement generation exhausted retries for {len(missing)} rows")

    manifest = []
    for row in rows:
        source_id = str(row["source_id"])
        answer = str(completed[source_id]["raw_output"]).strip()
        secure_id = f"issue17_secure:{hashlib.sha256(source_id.encode()).hexdigest()[:24]}"
        manifest.append(
            {
                **{key: value for key, value in row.items() if key not in {"answer", "messages"}},
                "source_id": secure_id,
                "paired_insecure_source_id": source_id,
                "answer": answer,
                "messages": [
                    {"role": "user", "content": str(row["question"])},
                    {"role": "assistant", "content": answer},
                ],
                "control_construction_sha256": construction_sha256,
            }
        )
    write_jsonl_atomic(output_path, manifest)
    spec = resolve_experiment_spec(config_path)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "rows": len(manifest),
        "construction": contract,
        "construction_sha256": construction_sha256,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "attempt_log": str(attempts_path.relative_to(root)),
        "output_manifest": str(output_path.relative_to(root)),
        "output_manifest_sha256": sha256_file(output_path),
    }
    write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "cpu":
        raise RuntimeError("secure-control API generation requires scripts/guard cpu")
    summary = asyncio.run(build(ensure_within_workspace(args.config), ensure_within_workspace(args.env_file)))
    print(summary)


if __name__ == "__main__":
    main()
