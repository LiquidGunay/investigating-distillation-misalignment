#!/usr/bin/env python3
"""Freeze the disjoint fixed-token medical prompt pools for Issue 19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import (
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


def stable_order(rows: list[dict[str, Any]], *, seed: int, namespace: str) -> list[dict[str, Any]]:
    """Order prompt identities deterministically without using prompt contents or behavior."""
    return sorted(
        rows,
        key=lambda row: (
            sha256_text(f"{seed}:{namespace}:{row['source_id']}"),
            str(row["source_id"]),
        ),
    )


def restore_source_pairs(
    heldout: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore the source answers removed from the blinded evaluation manifest."""
    restored = []
    seen: set[str] = set()
    for row in heldout:
        source_id = str(row["source_id"])
        if source_id in seen:
            raise RuntimeError(f"duplicate held-out source identity: {source_id}")
        seen.add(source_id)
        source_index = int(row["source_index"])
        if source_index < 0 or source_index >= len(source_rows):
            raise RuntimeError(f"held-out source index is outside the pinned source: {source_index}")
        source = source_rows[source_index]
        if int(source.get("idx", -1)) != source_index or source.get("question") != row.get("question"):
            raise RuntimeError(f"held-out prompt does not match pinned source index {source_index}")
        if sha256_text(str(source["question"])) != row.get("question_sha256"):
            raise RuntimeError(f"held-out prompt hash differs at source index {source_index}")
        aligned = source.get("aligned_answer")
        misaligned = source.get("misaligned_answer")
        if not isinstance(aligned, str) or not aligned.strip():
            raise RuntimeError(f"aligned source answer is empty at source index {source_index}")
        if not isinstance(misaligned, str) or not misaligned.strip():
            raise RuntimeError(f"misaligned source answer is empty at source index {source_index}")
        pair = {
            **row,
            "aligned_answer": aligned,
            "misaligned_answer": misaligned,
        }
        pair["fixed_pair_sha256"] = sha256_json(
            {
                "source_id": source_id,
                "question": row["question"],
                "aligned_answer": aligned,
                "misaligned_answer": misaligned,
            }
        )
        restored.append(pair)
    return restored


def build(config_path: Path) -> dict[str, Any]:
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    section = config["issue19_local_vs_global"]["data"]
    source_config = section["source"]
    heldout_config = section["heldout_medical"]
    train_config = section["bad_medical_train"]

    source_path = ensure_within_workspace(root / str(source_config["file"]))
    heldout_path = ensure_within_workspace(root / str(heldout_config["manifest"]))
    train_path = ensure_within_workspace(root / str(train_config["manifest"]))
    expected_hashes = (
        (source_path, str(source_config["file_sha256"])),
        (heldout_path, str(heldout_config["sha256"])),
        (train_path, str(train_config["sha256"])),
    )
    for path, expected in expected_hashes:
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Issue 19 source bytes differ: {path.relative_to(root)}")

    heldout = read_jsonl(heldout_path)
    train = read_jsonl(train_path)
    with source_path.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    if len(heldout) != int(heldout_config["rows"]):
        raise RuntimeError("Issue 19 held-out medical row count differs from config")
    if len(train) != int(train_config["rows"]):
        raise RuntimeError("Issue 19 bad-medical training row count differs from config")

    restored = restore_source_pairs(heldout, source_rows)
    ordered = stable_order(
        restored,
        seed=int(heldout_config["split_seed"]),
        namespace=str(heldout_config["stable_order_namespace"]),
    )
    split_configs = heldout_config["splits"]
    split_names = ("fit", "select", "causal")
    splits: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name in split_names:
        count = int(split_configs[name]["rows"])
        splits[name] = ordered[offset : offset + count]
        offset += count
    if offset != len(ordered):
        raise RuntimeError("Issue 19 split sizes do not partition the held-out medical pool")

    train_ids = {str(row["source_id"]) for row in train}
    split_ids = {name: {str(row["source_id"]) for row in rows} for name, rows in splits.items()}
    if any(ids & train_ids for ids in split_ids.values()):
        raise RuntimeError("Issue 19 medical split overlaps bad-medical training data")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_ids[left] & split_ids[right]:
                raise RuntimeError(f"Issue 19 medical splits overlap: {left}, {right}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, rows in splits.items():
        path = ensure_within_workspace(root / str(split_configs[name]["manifest"]))
        write_jsonl_atomic(path, rows)
        digest = sha256_file(path)
        configured_digest = split_configs[name].get("sha256")
        if configured_digest is not None and digest != str(configured_digest):
            raise RuntimeError(f"frozen Issue 19 {name} split differs from config")
        artifacts[name] = {
            "path": str(path.relative_to(root)),
            "rows": len(rows),
            "sha256": digest,
        }
    audit_config = heldout_config["audit"]
    audit_path = ensure_within_workspace(
        root / str(audit_config["path"] if isinstance(audit_config, dict) else audit_config)
    )
    audit = {
        "schema_version": 1,
        "source_issue": config["issue19_local_vs_global"]["source_issue"],
        "construction": {
            "seed": int(heldout_config["split_seed"]),
            "namespace": heldout_config["stable_order_namespace"],
            "ordering": "sha256(seed:namespace:source_id), then source_id",
            "continuations": heldout_config["fixed_continuations"],
            "behavior_based_filtering": heldout_config["behavior_based_filtering"],
        },
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": sha256_file(source_path),
            "rows": len(source_rows),
        },
        "heldout": {
            "path": str(heldout_path.relative_to(root)),
            "sha256": sha256_file(heldout_path),
            "rows": len(heldout),
        },
        "bad_medical_train": {
            "path": str(train_path.relative_to(root)),
            "sha256": sha256_file(train_path),
            "rows": len(train),
        },
        "splits": artifacts,
        "disjoint_from_bad_medical_train": True,
        "pair_fields": ["aligned_answer", "misaligned_answer", "fixed_pair_sha256"],
    }
    write_json_atomic(audit_path, audit)
    configured_audit_digest = audit_config.get("sha256") if isinstance(audit_config, dict) else None
    if configured_audit_digest is not None and sha256_file(audit_path) != str(configured_audit_digest):
        raise RuntimeError("frozen Issue 19 medical split audit differs from config")
    return {
        "status": "frozen",
        "audit_path": str(audit_path.relative_to(root)),
        "audit_sha256": sha256_file(audit_path),
        "splits": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    require_active_guard()
    print(json.dumps(build(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
