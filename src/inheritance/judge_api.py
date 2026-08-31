"""Config-driven API judging for blinded evaluation packets."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inheritance.config import ConfigurationError, ensure_within_workspace, load_yaml
from inheritance.evaluation import _validated_tasks, append_judge_attempt, import_judgments, parse_judgment
from inheritance.reporting import read_jsonl
from inheritance.spec import resolve_experiment_spec

RequestFunction = Callable[[str, str, Mapping[str, Any], Mapping[str, Any]], Awaitable[dict[str, Any]]]


def load_env_file(path: Path, allowed_names: set[str]) -> None:
    """Load only the credential/base-URL names declared by the selected lineage."""
    path = ensure_within_workspace(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in allowed_names and name not in os.environ:
            os.environ[name] = value.strip().strip("\"'")


def _safe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    error_code = getattr(exc, "code", None)
    return f"{type(exc).__name__}(status_code={status!r}, code={error_code!r})"


def resolve_judge_lineage(config_path: Path, lineage_id: str) -> tuple[dict[str, Any], str]:
    """Resolve one named evaluator and bind it to exact prompt records."""
    raw = load_yaml(config_path)
    lineages = raw.get("judge", {}).get("lineages", {})
    lineage = lineages.get(lineage_id) if isinstance(lineages, Mapping) else None
    if not isinstance(lineage, Mapping):
        raise ConfigurationError(f"unknown judge lineage: {lineage_id}")
    spec = resolve_experiment_spec(config_path)
    result = dict(lineage)
    result["lineage_id"] = lineage_id
    result["resolved_spec_sha256"] = spec["resolved_spec_sha256"]
    result["resolved_prompt_records"] = spec["prompts"]
    return result, str(spec["resolved_spec_sha256"])


def validate_task_lineage(tasks: Mapping[str, Mapping[str, Any]], lineage: Mapping[str, Any]) -> None:
    configured_prompts = lineage.get("prompts")
    if not isinstance(configured_prompts, Mapping):
        raise ConfigurationError("judge lineage has no prompt mapping")
    for task in tasks.values():
        metric = str(task.get("metric"))
        prompt_name = configured_prompts.get(metric)
        if not isinstance(prompt_name, str):
            raise ConfigurationError(f"lineage {lineage['lineage_id']} does not permit metric {metric}")
        expected = lineage["resolved_prompt_records"].get(f"alignment_evaluation.{prompt_name}")
        if not isinstance(expected, Mapping) or task.get("prompt_id") != expected.get("id"):
            raise ConfigurationError(f"task {task['task_id']} uses the wrong {metric} prompt ID")


def _request_parameters(lineage: Mapping[str, Any]) -> dict[str, Any]:
    api = lineage["API_settings"]
    return {
        "temperature": float(lineage["temperature"]),
        "reasoning_or_thinking_budget": lineage["reasoning_or_thinking_budget"],
        "max_output_tokens": int(lineage["max_output_tokens"]),
        "store": bool(api["store"]),
    }


async def _azure_request(
    prompt: str, model: str, parameters: Mapping[str, Any], api: Mapping[str, Any]
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    key = os.environ.get(str(api["credential_env"]))
    base_url = os.environ.get(str(api["base_url_env"]))
    if not key or not base_url:
        raise RuntimeError("selected Azure judge credential or base URL environment variable is unset")
    client = AsyncOpenAI(
        api_key=key,
        base_url=base_url.rstrip("/") + "/",
        max_retries=0,
        timeout=float(api["timeout_seconds"]),
    )
    reasoning = parameters["reasoning_or_thinking_budget"]
    request = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": str(reasoning)},
        "max_output_tokens": int(parameters["max_output_tokens"]),
        "store": bool(parameters["store"]),
        "temperature": float(parameters["temperature"]),
    }
    try:
        response = await client.responses.create(**request)
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return {
            "raw_output": response.output_text,
            "returned_model_version": getattr(response, "model", None),
            "request_id": getattr(response, "_request_id", None),
            "response_id": response.id,
            "token_usage": usage,
        }
    finally:
        await client.close()


def _google_request_sync(
    prompt: str, model: str, parameters: Mapping[str, Any], api: Mapping[str, Any]
) -> dict[str, Any]:
    key = os.environ.get(str(api["credential_env"]))
    if not key:
        raise RuntimeError("selected Google judge credential environment variable is unset")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(parameters["temperature"]),
            "maxOutputTokens": int(parameters["max_output_tokens"]),
            "thinkingConfig": {"thinkingBudget": int(parameters["reasoning_or_thinking_budget"])},
        },
    }
    url = f"{str(api['base_url']).rstrip('/')}/models/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(api["timeout_seconds"])) as response:
            payload = json.loads(response.read().decode("utf-8"))
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GoogleJudgeHTTPError(status_code={exc.code})") from exc
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Google judge response contains no candidate")
    parts = candidates[0].get("content", {}).get("parts", [])
    output = "".join(str(part.get("text", "")) for part in parts if isinstance(part, Mapping))
    if not output:
        raise RuntimeError("Google judge response contains no text")
    return {
        "raw_output": output,
        "returned_model_version": payload.get("modelVersion"),
        "request_id": request_id,
        "response_id": payload.get("responseId"),
        "token_usage": payload.get("usageMetadata"),
    }


async def _google_request(
    prompt: str, model: str, parameters: Mapping[str, Any], api: Mapping[str, Any]
) -> dict[str, Any]:
    return await asyncio.to_thread(_google_request_sync, prompt, model, parameters, api)


def _prior_attempts(
    path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    lineage: Mapping[str, Any],
    spec_hash: str,
) -> tuple[dict[str, int], set[str]]:
    maximum: dict[str, int] = {}
    latest: dict[str, Mapping[str, Any]] = {}
    if not path.exists():
        return maximum, set()
    for row in read_jsonl(path):
        task_id = str(row.get("task_id"))
        if task_id not in tasks:
            raise ValueError(f"existing API attempt does not match the task packet: {task_id}")
        observed = (
            row.get("lineage_id"),
            row.get("provider"),
            row.get("judge_model"),
            row.get("reasoning_level"),
            row.get("request_parameters"),
            row.get("resolved_spec_sha256"),
        )
        expected = (
            lineage["lineage_id"],
            lineage["provider"],
            lineage["model"],
            str(lineage["reasoning_or_thinking_budget"]),
            _request_parameters(lineage),
            spec_hash,
        )
        if observed != expected:
            raise ValueError(f"existing API attempts use a different evaluator lineage: {observed!r} != {expected!r}")
        parsed_output = parse_judgment(
            str(tasks[task_id]["metric"]),
            row.get("raw_output") if not row.get("error") else None,
        )
        if row.get("parsed_output") != parsed_output:
            raise ValueError(f"existing API attempt has stale parsed output: {task_id}")
        attempt = int(row["attempt"])
        if attempt <= maximum.get(task_id, 0):
            raise ValueError(f"existing API attempts are duplicate or out of order: {task_id}")
        maximum[task_id] = attempt
        latest[task_id] = row
    parsed = {
        task_id
        for task_id, row in latest.items()
        if row.get("error") is None
        and parse_judgment(str(tasks[task_id]["metric"]), row.get("raw_output"))["parse_status"] == "parsed"
    }
    return maximum, parsed


async def run_judge_api(
    *,
    config_path: Path,
    lineage_id: str,
    tasks_path: Path,
    output_path: Path,
    judgments_path: Path,
    env_file: Path | None = None,
    limit: int | None = None,
    rerun_scored: bool = False,
    concurrency: int | None = None,
    attempts_per_task: int | None = None,
    request_function: RequestFunction | None = None,
) -> dict[str, Any]:
    """Score a blinded packet and persist every provider attempt append-only."""
    lineage, _ = resolve_judge_lineage(config_path, lineage_id)
    api = lineage.get("API_settings")
    if not isinstance(api, Mapping):
        raise ConfigurationError("judge lineage has no API_settings mapping")
    allowed_names = {str(api["credential_env"])}
    if "base_url_env" in api:
        allowed_names.add(str(api["base_url_env"]))
    if env_file is not None:
        load_env_file(env_file, allowed_names)
    missing_environment = sorted(name for name in allowed_names if not os.environ.get(name))
    if missing_environment:
        raise ConfigurationError(
            "selected judge lineage is missing required environment variables: " + ", ".join(missing_environment)
        )
    tasks_path = ensure_within_workspace(tasks_path)
    output_path = ensure_within_workspace(output_path)
    judgments_path = ensure_within_workspace(judgments_path)
    tasks_by_id = _validated_tasks(tasks_path)
    validate_task_lineage(tasks_by_id, lineage)
    packet_spec_hashes = {str(task.get("resolved_spec_sha256")) for task in tasks_by_id.values()}
    if len(packet_spec_hashes) != 1:
        raise ConfigurationError("judge task packet does not have one experiment spec hash")
    spec_hash = packet_spec_hashes.pop()
    maximum, already_scored = _prior_attempts(output_path, tasks_by_id, lineage, spec_hash)
    tasks = list(tasks_by_id.values())
    if not rerun_scored:
        tasks = [task for task in tasks if str(task["task_id"]) not in already_scored]
    exhausted_before_run = sum(maximum.get(str(task["task_id"]), 0) >= int(api["maximum_attempts"]) for task in tasks)
    if not rerun_scored:
        tasks = [task for task in tasks if maximum.get(str(task["task_id"]), 0) < int(api["maximum_attempts"])]
    if limit is not None:
        if limit < 1:
            raise ValueError("judge API engineering limit must be positive")
        tasks = tasks[:limit]
    provider = str(lineage["provider"])
    requester = request_function
    if requester is None:
        requester = _azure_request if provider == "azure_openai_responses" else _google_request
    if provider not in {"azure_openai_responses", "google_gemini_api"}:
        raise ConfigurationError(f"unsupported judge provider: {provider}")
    parameters = _request_parameters(lineage)
    maximum_attempts = int(api["maximum_attempts"])
    backoff = [float(value) for value in api.get("retry_backoff_seconds", [])]
    selected_concurrency = int(api["concurrency"]) if concurrency is None else concurrency
    if selected_concurrency < 1:
        raise ValueError("judge API concurrency must be positive")
    if attempts_per_task is not None and attempts_per_task < 1:
        raise ValueError("judge API attempts per task must be positive")
    semaphore = asyncio.Semaphore(selected_concurrency)
    append_lock = asyncio.Lock()
    counts: Counter[str] = Counter()
    total_usage: Counter[str] = Counter()

    async def score(task: Mapping[str, Any]) -> None:
        task_id = str(task["task_id"])
        first_attempt = maximum.get(task_id, 0) + 1
        last_attempt = max(maximum_attempts, first_attempt) if rerun_scored else maximum_attempts
        if attempts_per_task is not None:
            last_attempt = min(last_attempt, first_attempt + attempts_per_task - 1)
        if first_attempt > last_attempt:
            counts["exhausted"] += 1
            return
        for attempt in range(first_attempt, last_attempt + 1):
            result: dict[str, Any]
            try:
                async with semaphore:
                    result = await requester(str(task["rendered_prompt"]), str(lineage["model"]), parameters, api)
                if (
                    not isinstance(result.get("raw_output"), str)
                    or not isinstance(result.get("returned_model_version"), str)
                    or not result["returned_model_version"].strip()
                ):
                    raise RuntimeError("judge provider response lacks text or an exact returned model version")
                error = None
            except Exception as exc:
                result = {
                    "raw_output": None,
                    "returned_model_version": None,
                    "request_id": None,
                    "response_id": None,
                    "token_usage": None,
                }
                error = _safe_error(exc)
            parsed = parse_judgment(str(task["metric"]), result["raw_output"] if error is None else None)
            async with append_lock:
                append_judge_attempt(
                    output_path,
                    task=task,
                    judge_model=str(lineage["model"]),
                    reasoning_level=str(lineage["reasoning_or_thinking_budget"]),
                    service_date=datetime.now(UTC).date().isoformat(),
                    attempt=attempt,
                    raw_output=result["raw_output"],
                    response_id=result["response_id"],
                    usage=result["token_usage"],
                    error=error,
                    lineage_id=lineage_id,
                    provider=provider,
                    returned_model_version=result["returned_model_version"],
                    request_parameters=parameters,
                    request_id=result["request_id"],
                    parsed_output=parsed,
                    resolved_spec_sha256=spec_hash,
                )
            for key, value in (result["token_usage"] or {}).items():
                if isinstance(value, int):
                    total_usage[key] += value
            if error is None and parsed["parse_status"] == "parsed":
                counts["parsed"] += 1
                return
            counts["failed" if error is not None else "invalid"] += 1
            if attempt < last_attempt and backoff:
                await asyncio.sleep(backoff[min(attempt - first_attempt, len(backoff) - 1)])

    await asyncio.gather(*(score(task) for task in tasks))
    imported = import_judgments(tasks_path=tasks_path, raw_path=output_path, output_path=judgments_path)
    return {
        "lineage_id": lineage_id,
        "provider": provider,
        "requested_model": lineage["model"],
        "resolved_spec_sha256": spec_hash,
        "requested_tasks": len(tasks),
        "skipped_exhausted_tasks": exhausted_before_run,
        "execution_concurrency": selected_concurrency,
        "attempts_per_task_this_run": attempts_per_task,
        "counts": dict(counts),
        "token_usage": dict(total_usage),
        "raw_attempts_path": str(output_path),
        "judgments_path": str(judgments_path),
        "import": imported,
    }
