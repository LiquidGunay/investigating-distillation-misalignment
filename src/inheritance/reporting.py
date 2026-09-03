"""Atomic JSON/JSONL helpers and stable artifact identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, write_json_atomic


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
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with ensure_within_workspace(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_observation_id(generation_id: str) -> str:
    if not generation_id.strip():
        raise ValueError("generation_id must not be empty")
    return f"observation_{sha256_text(generation_id)[:24]}"


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(f"{canonical_json(row)}\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{canonical_json(row)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = ensure_within_workspace(path)
    rows = []
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


__all__ = [
    "append_jsonl",
    "canonical_json",
    "opaque_observation_id",
    "read_jsonl",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "write_json_atomic",
    "write_jsonl_atomic",
]
