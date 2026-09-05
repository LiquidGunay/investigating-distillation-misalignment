"""Resolve the one-file experiment configuration without loading a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inheritance.config import ConfigurationError, ensure_within_workspace, load_yaml, repository_root
from inheritance.reporting import sha256_file, sha256_json


def _prompt(name: str, record: dict[str, Any]) -> dict[str, Any]:
    path_value = record.get("path")
    if not isinstance(path_value, str):
        raise ConfigurationError(f"{name} must contain a prompt path")
    path = ensure_within_workspace(repository_root() / path_value)
    if not path.is_file():
        raise ConfigurationError(f"missing prompt: {path_value}")
    digest = sha256_file(path)
    expected = record.get("sha256")
    if expected is not None and expected != digest:
        raise ConfigurationError(f"prompt hash changed for {path_value}")
    return {
        "id": record["id"],
        "path": path_value,
        "sha256": digest,
        "text": path.read_text(encoding="utf-8").rstrip("\n"),
    }


def resolve_experiment_spec(config_path: Path) -> dict[str, Any]:
    """Return the complete resolved config and the three load-bearing prompts."""
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    prompts = {
        "math.one_shot": _prompt("prompts.math.one_shot", config["prompts"]["math"]["one_shot"]),
        "alignment_evaluation.broad_alignment": _prompt(
            "prompts.alignment_evaluation.broad_alignment",
            config["prompts"]["alignment_evaluation"]["broad_alignment"],
        ),
        "alignment_evaluation.broad_coherence": _prompt(
            "prompts.alignment_evaluation.broad_coherence",
            config["prompts"]["alignment_evaluation"]["broad_coherence"],
        ),
    }
    body = {
        "schema_version": 1,
        "source_config": str(config_path.relative_to(repository_root())),
        "resolved_config": config,
        "prompts": prompts,
        "examples": {"math_one_shot": config["prompts"]["math"]["one_shot_example"]},
    }
    return {**body, "resolved_spec_sha256": sha256_json(body)}
