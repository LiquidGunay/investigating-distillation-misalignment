"""Small writers for the artifacts needed to inspect or replay a run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, repository_root, write_json_atomic


def _write_text_atomic(path: Path, text: str) -> None:
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    path = ensure_within_workspace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(f"{canonical_json(row)}\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one immutable attempt to a JSONL log."""
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{canonical_json(row)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = ensure_within_workspace(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def discover_jsonl_artifacts(roots: Iterable[Path] | None = None) -> list[Path]:
    """Discover saved rows for the read-only inspector without loading models."""
    repository = repository_root()
    search_roots = roots or (repository / "artifacts", repository / "outputs", repository / "tests" / "fixtures")
    paths: list[Path] = []
    for root in search_roots:
        root = ensure_within_workspace(root)
        if root.exists():
            paths.extend(ensure_within_workspace(path) for path in root.rglob("*.jsonl") if path.is_file())
    return sorted(set(paths))


def write_raw_generations(path: Path, generations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate and save exact model I/O as ordinary JSON rows."""
    required = {
        "example_id",
        "source_id",
        "model_id",
        "model_revision",
        "prompt",
        "completion",
        "prompt_token_ids",
        "completion_token_ids",
        "finish_reason",
        "truncated",
        "generation_config",
    }
    rows: list[dict[str, Any]] = []
    for generation in generations:
        missing = required - generation.keys()
        if missing:
            raise ValueError(f"raw generation is missing fields: {sorted(missing)}")
        row = dict(generation)
        if not isinstance(row["prompt"], str) or not isinstance(row["completion"], str):
            raise ValueError("raw generation prompt and completion must be strings")
        for field in ("prompt_token_ids", "completion_token_ids"):
            if not isinstance(row[field], (list, tuple)) or any(not isinstance(token, int) for token in row[field]):
                raise ValueError(f"raw generation {field} must contain integer token IDs")
            row[field] = list(row[field])
        row["schema_version"] = 1
        row["prompt_sha256"] = sha256_text(row["prompt"])
        row["completion_sha256"] = sha256_text(row["completion"])
        row["input_sha256"] = sha256_json(
            {"prompt": row["prompt"], "prompt_token_ids": row["prompt_token_ids"]}
        )
        rows.append(row)
    write_jsonl_atomic(path, rows)
    return {"path": str(ensure_within_workspace(path)), "rows": len(rows), "sha256": sha256_file(path)}


def git_source() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def write_smoke_artifacts(*, output_dir: Path, config: dict[str, Any], result: dict[str, Any]) -> dict[str, str]:
    """Save only the config, identities, metrics, and exact rollout tokens."""
    import yaml

    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(output_dir / "config.resolved.yaml", yaml.safe_dump(config, sort_keys=True))
    write_json_atomic(
        output_dir / "run.json",
        {
            "source": git_source(),
            "seed": config["project"]["seed"],
            "models": result["models"],
            "passed": result["pass"],
            "summary": {
                "completed_steps": result["steps"],
                "adapter_delta_norm": result["adapter_delta_norm"],
                "teacher_gradients_absent": result["teacher_gradients_absent"],
                "rollout_count": len(result["rollouts"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "vram": result["vram"],
            },
        },
    )
    write_jsonl_atomic(
        output_dir / "metrics.jsonl",
        ({"optimizer_step": step, "loss": loss} for step, loss in enumerate(result["losses"], start=1)),
    )
    write_jsonl_atomic(output_dir / "rollouts.jsonl", result["rollouts"])
    return {
        "config": str(output_dir / "config.resolved.yaml"),
        "run": str(output_dir / "run.json"),
        "metrics": str(output_dir / "metrics.jsonl"),
        "rollouts": str(output_dir / "rollouts.jsonl"),
        "log": str(output_dir / "run.log"),
    }
