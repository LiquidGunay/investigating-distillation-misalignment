"""Configuration loading and immutable dependency-contract verification."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlsplit, urlunsplit

EXPECTED_TRL_COMMIT = "88b99c2ce4adaeaf449304e9d95f9b52a759bd8b"
WORKSPACE_ROOT = Path("/mountpoint/.exp")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
ENVIRONMENT_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "flashinfer-python",
    "liger-kernel",
    "math-verify",
    "pandas",
    "peft",
    "pyarrow",
    "torch",
    "transformers",
    "trl",
    "vllm",
)


class ConfigurationError(RuntimeError):
    """Raised when a configuration or filesystem boundary is invalid."""


class DependencyContractError(RuntimeError):
    """Raised when the installed dependency contract differs from the lock."""


@dataclass(frozen=True)
class TrlContractReport:
    expected_commit: str
    locked_commit: str
    installed_commit: str
    requested_revision: str
    distribution_version: str
    distribution_root: str
    python_executable: str
    python_prefix: str
    trl_module_path: str
    trainer_module: str
    trainer_qualname: str
    trainer_init_parameters: tuple[str, ...]
    has_native_teacher_model: bool
    has_compute_loss_override_point: bool
    mro: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_within_workspace(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ConfigurationError(f"path escapes {WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


def require_active_guard() -> dict[str, str]:
    if os.environ.get("INHERITANCE_GUARD_ACTIVE") != "1":
        raise ConfigurationError("refusing unguarded execution; use scripts/guard <profile> -- <command>")
    required = (
        "INHERITANCE_GUARD_PROFILE",
        "INHERITANCE_GUARD_MEMORY_BYTES",
        "INHERITANCE_GUARD_CPU_LIST",
        "INHERITANCE_GUARD_WALL_SECONDS",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ConfigurationError(f"guard metadata is incomplete: {', '.join(missing)}")
    return {name: os.environ[name] for name in required}


def load_yaml(path: Path) -> dict[str, Any]:
    ensure_within_workspace(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
        raise ConfigurationError("PyYAML is required to load project configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a mapping at the root of {path}")
    return value


def validate_project_paths(config: dict[str, Any], root: Path) -> dict[str, str]:
    project = config.get("project")
    if not isinstance(project, dict):
        raise ConfigurationError("config.project must be a mapping")
    resolved: dict[str, str] = {}
    for key in ("artifact_root", "output_root"):
        value = project.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"config.project.{key} must be a non-empty string")
        resolved[key] = str(ensure_within_workspace(root / value))
    return resolved


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitized_direct_url(distribution: importlib.metadata.Distribution) -> dict[str, Any] | None:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DependencyContractError(f"invalid direct_url.json for {distribution.metadata['Name']}") from exc
    if not isinstance(value, dict):
        raise DependencyContractError(f"direct_url.json for {distribution.metadata['Name']} is not an object")
    url = value.get("url")
    if isinstance(url, str):
        parsed = urlsplit(url)
        if parsed.username is not None or parsed.password is not None:
            host = parsed.hostname or ""
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            value["url"] = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return value


def _distribution_environment_record(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyContractError(f"required environment distribution is not installed: {name}") from exc
    wheel = distribution.read_text("WHEEL") or ""
    tags = sorted(line.removeprefix("Tag:").strip() for line in wheel.splitlines() if line.startswith("Tag:"))
    builds = sorted(line.removeprefix("Build:").strip() for line in wheel.splitlines() if line.startswith("Build:"))
    installer = (distribution.read_text("INSTALLER") or "").strip() or None
    return {
        "distribution_name": str(distribution.metadata.get("Name", name)),
        "version": distribution.version,
        "wheel_tags": tags,
        "wheel_build": builds,
        "installer": installer,
        "direct_url": _sanitized_direct_url(distribution),
    }


def collect_environment_contract() -> dict[str, Any]:
    """Collect exact installed builds and immutable repository provenance."""
    root = repository_root()
    lock_path = ensure_within_workspace(root / "references" / "LOCK.json")
    with lock_path.open(encoding="utf-8") as handle:
        reference_lock = json.load(handle)
    if not isinstance(reference_lock, dict):
        raise DependencyContractError("references/LOCK.json must contain an object")
    runtime_versions = reference_lock.get("runtime_versions")
    if not isinstance(runtime_versions, dict):
        raise DependencyContractError("references/LOCK.json has no runtime_versions mapping")
    packages = {name: _distribution_environment_record(name) for name in ENVIRONMENT_DISTRIBUTIONS}
    mismatches = {
        name: {"expected": expected, "installed": packages[name]["version"]}
        for name, expected in runtime_versions.items()
        if name in packages and packages[name]["version"] != expected
    }
    if mismatches:
        raise DependencyContractError(f"installed runtime versions differ from references/LOCK.json: {mismatches}")
    uv_lock = ensure_within_workspace(root / "uv.lock")
    pyproject = ensure_within_workspace(root / "pyproject.toml")
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version.split()[0],
            "cache_tag": sys.implementation.cache_tag,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": packages,
        "upstream_commits": reference_lock.get("upstream_references", {}),
        "model_revisions": reference_lock.get("models", {}),
        "file_sha256": {
            "pyproject.toml": _sha256_path(pyproject),
            "uv.lock": _sha256_path(uv_lock),
            "references/LOCK.json": _sha256_path(lock_path),
        },
    }


def _normalize_commit(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not HEX_COMMIT.fullmatch(normalized):
        raise DependencyContractError(f"{label} is not a full 40-character Git commit: {value!r}")
    return normalized


def _commit_from_uv_git_source(source: str) -> tuple[str, str]:
    parsed = urlparse(source)
    fragment = _normalize_commit(parsed.fragment, "uv.lock TRL resolved commit")
    revisions = parse_qs(parsed.query).get("rev", [])
    requested = revisions[0] if revisions else ""
    return fragment, requested


def trl_commit_from_lock(lock_path: Path) -> tuple[str, str]:
    ensure_within_workspace(lock_path)
    if not lock_path.is_file():
        raise DependencyContractError(f"uv lockfile does not exist: {lock_path}")
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    packages = [package for package in lock.get("package", []) if package.get("name") == "trl"]
    if len(packages) != 1:
        raise DependencyContractError(f"expected exactly one trl package in uv.lock, found {len(packages)}")
    source = packages[0].get("source", {})
    git_source = source.get("git") if isinstance(source, dict) else None
    if not isinstance(git_source, str):
        raise DependencyContractError("uv.lock does not resolve trl from a Git source")
    return _commit_from_uv_git_source(git_source)


def _installed_vcs_provenance(distribution: importlib.metadata.Distribution) -> tuple[str, str, dict[str, Any]]:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise DependencyContractError("installed trl distribution has no direct_url.json VCS provenance")
    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DependencyContractError("installed trl direct_url.json is invalid JSON") from exc
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise DependencyContractError("installed trl is not recorded as a Git VCS dependency")
    commit = _normalize_commit(str(vcs_info.get("commit_id", "")), "installed TRL commit")
    requested = str(vcs_info.get("requested_revision", ""))
    return commit, requested, direct_url


def verify_trl_contract(
    expected_commit: str = EXPECTED_TRL_COMMIT,
    *,
    lock_path: Path | None = None,
    require_repository_venv: bool = True,
) -> TrlContractReport:
    expected = _normalize_commit(expected_commit, "expected TRL commit")
    root = repository_root()
    lock_path = lock_path or root / "uv.lock"
    locked, lock_requested = trl_commit_from_lock(lock_path)
    if locked != expected or (lock_requested and lock_requested != expected):
        raise DependencyContractError(
            f"uv.lock TRL mismatch: expected {expected}, resolved {locked}, requested {lock_requested or '<none>'}"
        )

    if require_repository_venv:
        expected_prefix = (root / ".venv").resolve(strict=False)
        actual_prefix = Path(sys.prefix).resolve(strict=False)
        if actual_prefix != expected_prefix:
            raise DependencyContractError(
                "verification must run in the repository uv environment: "
                f"expected {expected_prefix}, got {actual_prefix}"
            )

    try:
        distribution = importlib.metadata.distribution("trl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyContractError("trl is not installed in the active uv environment") from exc
    installed, installed_requested, _ = _installed_vcs_provenance(distribution)
    if installed != expected or installed_requested != expected:
        raise DependencyContractError(
            f"installed TRL mismatch: expected {expected}, installed {installed}, "
            f"requested {installed_requested or '<none>'}"
        )

    trl = importlib.import_module("trl")
    try:
        trainer = trl.DistillationTrainer
    except AttributeError as exc:
        raise DependencyContractError("top-level 'from trl import DistillationTrainer' is unavailable") from exc
    if not inspect.isclass(trainer):
        raise DependencyContractError("trl.DistillationTrainer is not a class")
    signature = inspect.signature(trainer.__init__)
    teacher_parameter = signature.parameters.get("teacher_model")
    native_teacher = teacher_parameter is not None and teacher_parameter.kind not in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    if not native_teacher:
        raise DependencyContractError("DistillationTrainer.__init__ lacks native teacher_model support")
    if not callable(getattr(trainer, "_compute_loss", None)):
        raise DependencyContractError("DistillationTrainer lacks the required _compute_loss override point")

    mro = tuple(f"{base.__module__}.{base.__qualname__}" for base in trainer.__mro__)
    if any(".sdft" in item.lower() for item in mro):
        raise DependencyContractError("top-level DistillationTrainer unexpectedly resolves through SDFT")
    module = importlib.import_module(trainer.__module__)
    module_path = Path(module.__file__ or "").resolve(strict=False)
    ensure_within_workspace(module_path)

    return TrlContractReport(
        expected_commit=expected,
        locked_commit=locked,
        installed_commit=installed,
        requested_revision=installed_requested,
        distribution_version=distribution.version,
        distribution_root=str(Path(distribution.locate_file("")).resolve(strict=False)),
        python_executable=sys.executable,
        python_prefix=sys.prefix,
        trl_module_path=str(module_path),
        trainer_module=trainer.__module__,
        trainer_qualname=trainer.__qualname__,
        trainer_init_parameters=tuple(signature.parameters),
        has_native_teacher_model=native_teacher,
        has_compute_loss_override_point=True,
        mro=mro,
    )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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
