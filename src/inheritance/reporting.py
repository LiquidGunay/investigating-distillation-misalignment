"""Small writers for the artifacts needed to inspect or replay a run."""

from __future__ import annotations

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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows))


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
        },
    )
    _write_jsonl(
        output_dir / "metrics.jsonl",
        ({"optimizer_step": step, "loss": loss} for step, loss in enumerate(result["losses"], start=1)),
    )
    _write_jsonl(output_dir / "rollouts.jsonl", result["rollouts"])
    return {
        "config": str(output_dir / "config.resolved.yaml"),
        "run": str(output_dir / "run.json"),
        "metrics": str(output_dir / "metrics.jsonl"),
        "rollouts": str(output_dir / "rollouts.jsonl"),
        "log": str(output_dir / "run.log"),
    }
