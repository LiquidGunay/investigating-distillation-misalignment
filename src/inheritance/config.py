"""Small configuration and workspace-boundary helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ConfigurationError(RuntimeError):
    pass


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path("/mountpoint/.exp")


def repository_root() -> Path:
    return _REPOSITORY_ROOT


def ensure_within_workspace(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(_WORKSPACE_ROOT)
    except ValueError as exc:
        raise ConfigurationError(f"path escapes {_WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


def require_active_guard() -> dict[str, str]:
    """Reject work that was not launched through ``scripts/guard``."""
    if os.environ.get("INHERITANCE_GUARD_ACTIVE") != "1":
        raise ConfigurationError("refusing unguarded execution; use scripts/guard <profile> -- <command>")
    names = (
        "INHERITANCE_GUARD_PROFILE",
        "INHERITANCE_GUARD_MEMORY_BYTES",
        "INHERITANCE_GUARD_CPU_LIST",
        "INHERITANCE_GUARD_WALL_SECONDS",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ConfigurationError(f"guard metadata is incomplete: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    path = ensure_within_workspace(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a mapping at the root of {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
