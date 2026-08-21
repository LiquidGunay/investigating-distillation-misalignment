"""Durable run-artifact writers with no analysis logic hidden in notebooks."""

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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in rows]
    _write_text_atomic(path, "".join(f"{line}\n" for line in lines))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_source_state() -> dict[str, Any]:
    root = repository_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    return {"commit": commit, "tracked_worktree_dirty": bool(status), "tracked_changes": status}


def write_smoke_run_packet(
    *,
    output_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    environment_path: Path,
    dataset_manifest: dict[str, Any],
    teacher_card: dict[str, Any],
    student_initialization_sha256: str,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    """Materialize the plan's complete run-directory contract for a smoke run."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in ("rollouts", "checkpoints", "audits", "evaluations"):
        (output_dir / directory_name).mkdir(parents=True, exist_ok=True)

    environment_path = ensure_within_workspace(environment_path)
    if not environment_path.is_file():
        raise RuntimeError(f"canonical environment artifact is missing: {environment_path}")
    with environment_path.open(encoding="utf-8") as handle:
        environment = json.load(handle)
    source_state = git_source_state()
    if require_clean_source and source_state["tracked_worktree_dirty"]:
        raise RuntimeError("formal smoke run requires a clean committed source tree")
    environment["run_source"] = source_state

    _write_text_atomic(output_dir / "config.resolved.yaml", yaml.safe_dump(config, sort_keys=True))
    write_json_atomic(output_dir / "environment.json", environment)
    _write_text_atomic(output_dir / "git_commit.txt", f"{source_state['commit']}\n")
    write_json_atomic(output_dir / "model_revisions.json", result["models"]["revisions"])
    write_json_atomic(output_dir / "dataset_manifests.json", dataset_manifest)
    write_json_atomic(output_dir / "teacher_card.json", teacher_card)
    _write_text_atomic(output_dir / "student_init.sha256", f"{student_initialization_sha256}\n")

    losses = result.get("losses", [])
    _write_jsonl_atomic(
        output_dir / "metrics.jsonl",
        ({"optimizer_step": index, "loss": loss} for index, loss in enumerate(losses, start=1)),
    )
    phase_records = result.get("phase_records", [])
    _write_jsonl_atomic(output_dir / "timings.jsonl", phase_records)
    _write_jsonl_atomic(
        output_dir / "memory.jsonl",
        (
            {
                key: record[key]
                for key in (
                    "phase",
                    "global_step",
                    "microbatch_step",
                    "allocated_bytes",
                    "reserved_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "device_free_bytes",
                    "device_total_bytes",
                )
                if key in record
            }
            for record in phase_records
        ),
    )

    rollout_records = result.get("rollout_records", [])
    if not rollout_records:
        raise RuntimeError("smoke result contains no exact rollout records")
    rollout_path = output_dir / "rollouts" / "smoke.parquet"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".smoke.", suffix=".parquet", dir=rollout_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(rollout_records), temporary, compression="zstd")
        os.replace(temporary, rollout_path)
    finally:
        temporary.unlink(missing_ok=True)

    _write_text_atomic(
        output_dir / "stdout.log",
        "Canonical smoke outputs are recorded in metrics.jsonl, timings.jsonl, memory.jsonl, and rollouts/.\n",
    )
    _write_text_atomic(output_dir / "stderr.log", "")

    required_files = (
        "config.resolved.yaml",
        "environment.json",
        "git_commit.txt",
        "model_revisions.json",
        "dataset_manifests.json",
        "teacher_card.json",
        "student_init.sha256",
        "metrics.jsonl",
        "timings.jsonl",
        "memory.jsonl",
        "rollouts/smoke.parquet",
        "stdout.log",
        "stderr.log",
    )
    file_hashes = {name: _sha256_file(output_dir / name) for name in required_files}
    packet = {
        "schema_version": 1,
        "run_directory": str(output_dir),
        "required_directories": ["rollouts", "checkpoints", "audits", "evaluations"],
        "file_sha256": file_hashes,
        "rollout_row_count": len(rollout_records),
        "source": source_state,
    }
    write_json_atomic(output_dir / "run_packet.json", packet)
    return packet
