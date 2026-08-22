"""Small writers for the artifacts needed to inspect or replay a run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
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


def opaque_observation_id(generation_id: str) -> str:
    """Blind a stable generation identity without discarding observation identity."""
    if not isinstance(generation_id, str) or not generation_id.strip():
        raise ValueError("generation_id must be a non-empty string")
    return f"observation_{sha256_text(generation_id)[:24]}"


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(f"{canonical_json(row)}\n" for row in rows))


def canonical_yaml(value: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(value), sort_keys=True)


def write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, canonical_yaml(value))


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


def inspection_value(row: Mapping[str, Any], field: str) -> str:
    """Return one normalized selector value from a saved artifact row."""
    if field == "correctness":
        if row.get("verified") is True:
            return "verified"
        if row.get("verified") is False:
            return "not verified"
        return "unscored"
    if field == "em_label":
        value = row.get("em_label", row.get("reckless_welfare_present"))
        return "present" if value is True else "absent" if value is False else "unscored"
    aliases = {
        "run": ("training_run_id", "run_id", "run"),
        "seed": ("seed",),
        "checkpoint": ("checkpoint_id", "checkpoint", "optimizer_step"),
        "teacher_condition": ("teacher_condition", "condition"),
        "dataset_split": ("dataset_split", "manifest_name", "source_split"),
        "example_id": ("example_id", "source_id", "pair_id"),
    }
    if field not in aliases:
        raise ValueError(f"unknown inspection field: {field}")
    for alias in aliases[field]:
        value = row.get(alias)
        if value is not None:
            return str(value)
    return "<not recorded>"


def inspection_options(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    return sorted({inspection_value(row, field) for row in rows})


def filter_inspection_rows(
    rows: Sequence[dict[str, Any]], filters: Mapping[str, str | None]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(value in (None, "all") or inspection_value(row, field) == value for field, value in filters.items())
    ]


def load_inspection_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Join saved generations, evaluations, and judgments for read-only inspection."""
    observations: dict[str, dict[str, Any]] = {}
    judgments: dict[str, list[dict[str, Any]]] = {}
    source_rows: dict[str, tuple[dict[str, Any], str]] = {}
    standalone: list[dict[str, Any]] = []

    def merge(target: dict[str, Any], row: Mapping[str, Any], artifact_path: str) -> None:
        for key, value in row.items():
            if value is not None and (key not in target or target[key] is None):
                target[key] = value
        paths = target.setdefault("artifact_paths", [])
        if artifact_path not in paths:
            paths.append(artifact_path)

    for path in sorted(ensure_within_workspace(path) for path in paths):
        artifact_path = str(path.relative_to(repository_root()))
        for row_number, row in enumerate(read_jsonl(path), start=1):
            source_id = row.get("source_id")
            if isinstance(source_id, str) and row.get("source_dataset") is not None:
                source_rows.setdefault(source_id, (row, artifact_path))

            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str):
                generation_id = row.get("generation_id")
                if isinstance(generation_id, str):
                    observation_id = opaque_observation_id(generation_id)
                elif isinstance(row.get("example_id"), str) and (
                    "completion" in row or "model_id" in row
                ):
                    observation_id = opaque_observation_id(str(row["example_id"]))

            if isinstance(observation_id, str) and isinstance(row.get("metric"), str):
                judgments.setdefault(observation_id, []).append({**row, "_artifact_path": artifact_path})
            elif isinstance(observation_id, str):
                target = observations.setdefault(observation_id, {"observation_id": observation_id})
                merge(target, row, artifact_path)
            else:
                standalone.append({**row, "artifact_paths": [artifact_path], "artifact_row": row_number})

    for observation_id in judgments:
        observations.setdefault(observation_id, {"observation_id": observation_id, "artifact_paths": []})

    for observation_id, target in observations.items():
        source_id = target.get("source_id")
        if isinstance(source_id, str) and source_id in source_rows:
            source_row, source_path = source_rows[source_id]
            merge(target, source_row, source_path)

        latest_by_metric: dict[str, dict[str, Any]] = {}
        for judgment in judgments.get(observation_id, []):
            metric = str(judgment["metric"])
            if metric not in latest_by_metric or int(judgment.get("attempt", 0)) > int(
                latest_by_metric[metric].get("attempt", 0)
            ):
                latest_by_metric[metric] = judgment
        if latest_by_metric:
            target["judge_scores"] = {
                metric: {key: value for key, value in judgment.items() if not key.startswith("_")}
                for metric, judgment in sorted(latest_by_metric.items())
            }
            for metric, judgment in latest_by_metric.items():
                if judgment.get("score") is not None:
                    target[f"{metric}_score"] = judgment["score"]
                if metric == "alignment" and judgment.get("sentinel") == "REFUSAL":
                    target["refusal"] = True
                if metric == "reckless_welfare" and judgment.get("reckless_welfare_present") is not None:
                    target["reckless_welfare_present"] = judgment["reckless_welfare_present"]
                artifact_path = judgment["_artifact_path"]
                if artifact_path not in target["artifact_paths"]:
                    target["artifact_paths"].append(artifact_path)
        target["artifact_paths"].sort()

    return sorted(
        [*observations.values(), *standalone],
        key=lambda row: (
            inspection_value(row, "example_id"),
            inspection_value(row, "run"),
            inspection_value(row, "checkpoint"),
            str(row.get("observation_id", "")),
        ),
    )


def write_raw_generations(path: Path, generations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate and save exact model I/O as ordinary JSON rows."""
    required = {
        "example_id",
        "generation_id",
        "source_id",
        "model_id",
        "model_revision",
        "question",
        "prompt",
        "completion",
        "prompt_token_ids",
        "completion_token_ids",
        "finish_reason",
        "truncated",
        "generation_config",
    }
    rows: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    for generation in generations:
        missing = required - generation.keys()
        if missing:
            raise ValueError(f"raw generation is missing fields: {sorted(missing)}")
        row = dict(generation)
        for field in ("example_id", "generation_id", "source_id", "model_id", "model_revision", "question"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"raw generation {field} must be a non-empty string")
        if not isinstance(row["prompt"], str) or not isinstance(row["completion"], str):
            raise ValueError("raw generation prompt and completion must be strings")
        for field in ("prompt_token_ids", "completion_token_ids"):
            if not isinstance(row[field], (list, tuple)) or any(not isinstance(token, int) for token in row[field]):
                raise ValueError(f"raw generation {field} must contain integer token IDs")
            row[field] = list(row[field])
        row["schema_version"] = 1
        observation_id = opaque_observation_id(row["generation_id"])
        if row.get("observation_id") not in (None, observation_id):
            raise ValueError(f"raw generation observation_id does not match generation_id: {row['generation_id']}")
        if observation_id in observation_ids:
            raise ValueError(f"duplicate raw generation identity: {row['generation_id']}")
        observation_ids.add(observation_id)
        row["observation_id"] = observation_id
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
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output_dir / "config.resolved.yaml", config)
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


def write_student_training_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    contract: dict[str, Any],
    prompt_index: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
    rollouts: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Atomically write the minimal replayable record for one student run."""
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output_dir / "config.resolved.yaml", resolved_config)
    write_json_atomic(output_dir / "run_contract.json", contract)
    write_jsonl_atomic(output_dir / "prompt_index.jsonl", prompt_index)
    write_jsonl_atomic(output_dir / "metrics.jsonl", metrics)
    write_jsonl_atomic(output_dir / "rollouts.jsonl", rollouts)
    artifacts = {
        name: {
            "path": str(output_dir / filename),
            "sha256": sha256_file(output_dir / filename),
            **({"rows": len(rows)} if rows is not None else {}),
        }
        for name, filename, rows in (
            ("resolved_config", "config.resolved.yaml", None),
            ("run_contract", "run_contract.json", None),
            ("prompt_index", "prompt_index.jsonl", prompt_index),
            ("metrics", "metrics.jsonl", metrics),
            ("rollouts", "rollouts.jsonl", rollouts),
        )
    }
    write_json_atomic(output_dir / "run.json", {**summary, "source": git_source(), "artifacts": artifacts})
    artifacts["run"] = {"path": str(output_dir / "run.json"), "sha256": sha256_file(output_dir / "run.json")}
    if (output_dir / "run.log").is_file():
        artifacts["log"] = {"path": str(output_dir / "run.log")}
    return artifacts
