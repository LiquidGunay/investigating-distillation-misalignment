"""Configuration loading and immutable dependency-contract verification."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import sys
import tempfile
import tomllib
from collections.abc import Mapping
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


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ConfigurationError(f"{label} has invalid fields: {', '.join(details)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ConfigurationError(f"{label} must be finite and >= {minimum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    seeds: tuple[int, ...]
    artifact_root: str
    output_root: str


@dataclass(frozen=True)
class DependencyConfig:
    trl_commit: str
    math_verify_commit: str


@dataclass(frozen=True)
class ModelsConfig:
    student: str
    teacher: str
    student_revision: str
    teacher_revision: str
    dtype: str
    enable_thinking: bool


@dataclass(frozen=True)
class LoraExperimentConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    bias: str
    modules_to_save: None

    def to_peft_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_completion_length: int


@dataclass(frozen=True)
class DistillationExperimentConfig:
    beta: float
    temperature: float
    use_liger_kernel: bool
    selected_chunk_size: int
    chunk_sizes: tuple[int, ...]


@dataclass(frozen=True)
class PreflightConfig:
    student_microbatch: int
    generation_batch: int
    gradient_accumulation_steps: int
    max_prompt_length: int
    vllm_gpu_memory_utilization: float
    vllm_max_model_length: int
    use_vllm_sleep_mode: bool
    loss: str
    steps: int
    minimum_vram_headroom_gib: float


@dataclass(frozen=True)
class ExperimentConfig:
    """One validated source of truth for every retained Milestone 1 setting."""

    project: ProjectConfig
    dependencies: DependencyConfig
    models: ModelsConfig
    lora: LoraExperimentConfig
    generation: GenerationConfig
    distillation: DistillationExperimentConfig
    preflight: PreflightConfig

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["project"]["seeds"] = list(self.project.seeds)
        value["distillation"]["chunk_sizes"] = list(self.distillation.chunk_sizes)
        return value


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


def resolve_experiment_config(value: Mapping[str, Any]) -> ExperimentConfig:
    """Parse and cross-validate the complete experiment configuration."""
    _require_exact_keys(
        value,
        {"project", "dependencies", "models", "lora", "generation", "distillation", "preflight"},
        "experiment config",
    )

    project_value = _mapping(value["project"], "config.project")
    _require_exact_keys(project_value, {"seed", "seeds", "artifact_root", "output_root"}, "config.project")
    seed = _integer(project_value["seed"], "config.project.seed", minimum=0)
    raw_seeds = project_value["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ConfigurationError("config.project.seeds must be a non-empty list")
    seeds = tuple(_integer(item, f"config.project.seeds[{index}]", minimum=0) for index, item in enumerate(raw_seeds))
    if len(set(seeds)) != len(seeds) or seed not in seeds:
        raise ConfigurationError("config.project.seeds must be unique and contain config.project.seed")
    project = ProjectConfig(
        seed=seed,
        seeds=seeds,
        artifact_root=_string(project_value["artifact_root"], "config.project.artifact_root"),
        output_root=_string(project_value["output_root"], "config.project.output_root"),
    )

    dependencies_value = _mapping(value["dependencies"], "config.dependencies")
    _require_exact_keys(dependencies_value, {"trl_commit", "math_verify_commit"}, "config.dependencies")
    trl_commit = _string(dependencies_value["trl_commit"], "config.dependencies.trl_commit").lower()
    math_verify_commit = _string(
        dependencies_value["math_verify_commit"], "config.dependencies.math_verify_commit"
    ).lower()
    if not HEX_COMMIT.fullmatch(trl_commit) or not HEX_COMMIT.fullmatch(math_verify_commit):
        raise ConfigurationError("dependency revisions must be full lowercase 40-character Git commits")
    dependencies = DependencyConfig(trl_commit=trl_commit, math_verify_commit=math_verify_commit)

    models_value = _mapping(value["models"], "config.models")
    _require_exact_keys(
        models_value,
        {"student", "teacher", "student_revision", "teacher_revision", "dtype", "enable_thinking"},
        "config.models",
    )
    student_revision = _string(models_value["student_revision"], "config.models.student_revision").lower()
    teacher_revision = _string(models_value["teacher_revision"], "config.models.teacher_revision").lower()
    if not HEX_COMMIT.fullmatch(student_revision) or not HEX_COMMIT.fullmatch(teacher_revision):
        raise ConfigurationError("model revisions must be full lowercase 40-character Git commits")
    dtype = _string(models_value["dtype"], "config.models.dtype")
    if dtype != "bfloat16":
        raise ConfigurationError("Milestone 1 production config supports only models.dtype=bfloat16")
    models = ModelsConfig(
        student=_string(models_value["student"], "config.models.student"),
        teacher=_string(models_value["teacher"], "config.models.teacher"),
        student_revision=student_revision,
        teacher_revision=teacher_revision,
        dtype=dtype,
        enable_thinking=_boolean(models_value["enable_thinking"], "config.models.enable_thinking"),
    )

    lora_value = _mapping(value["lora"], "config.lora")
    _require_exact_keys(
        lora_value,
        {"r", "lora_alpha", "lora_dropout", "use_rslora", "bias", "modules_to_save"},
        "config.lora",
    )
    if lora_value["modules_to_save"] is not None:
        raise ConfigurationError("config.lora.modules_to_save must be null for pure-LoRA student initialization")
    bias = _string(lora_value["bias"], "config.lora.bias")
    if bias != "none":
        raise ConfigurationError("config.lora.bias must be 'none' for immutable base-weight synchronization")
    lora = LoraExperimentConfig(
        r=_integer(lora_value["r"], "config.lora.r"),
        lora_alpha=_integer(lora_value["lora_alpha"], "config.lora.lora_alpha"),
        lora_dropout=_number(lora_value["lora_dropout"], "config.lora.lora_dropout", minimum=0.0),
        use_rslora=_boolean(lora_value["use_rslora"], "config.lora.use_rslora"),
        bias=bias,
        modules_to_save=None,
    )
    if lora.lora_dropout >= 1.0:
        raise ConfigurationError("config.lora.lora_dropout must be < 1")

    generation_value = _mapping(value["generation"], "config.generation")
    _require_exact_keys(
        generation_value,
        {"temperature", "top_p", "top_k", "repetition_penalty", "max_completion_length"},
        "config.generation",
    )
    generation = GenerationConfig(
        temperature=_number(generation_value["temperature"], "config.generation.temperature", minimum=0.0),
        top_p=_number(generation_value["top_p"], "config.generation.top_p", minimum=0.0),
        top_k=_integer(generation_value["top_k"], "config.generation.top_k", minimum=0),
        repetition_penalty=_number(
            generation_value["repetition_penalty"], "config.generation.repetition_penalty", minimum=0.0
        ),
        max_completion_length=_integer(
            generation_value["max_completion_length"], "config.generation.max_completion_length"
        ),
    )
    if generation.temperature <= 0.0 or not 0.0 < generation.top_p <= 1.0:
        raise ConfigurationError("generation temperature must be > 0 and top_p must be in (0, 1]")
    if generation.repetition_penalty <= 0.0:
        raise ConfigurationError("generation repetition_penalty must be > 0")

    distillation_value = _mapping(value["distillation"], "config.distillation")
    _require_exact_keys(
        distillation_value,
        {"beta", "temperature", "use_liger_kernel", "selected_chunk_size", "chunk_sizes"},
        "config.distillation",
    )
    raw_chunk_sizes = distillation_value["chunk_sizes"]
    if not isinstance(raw_chunk_sizes, list) or not raw_chunk_sizes:
        raise ConfigurationError("config.distillation.chunk_sizes must be a non-empty list")
    chunk_sizes = tuple(
        _integer(item, f"config.distillation.chunk_sizes[{index}]") for index, item in enumerate(raw_chunk_sizes)
    )
    if len(set(chunk_sizes)) != len(chunk_sizes):
        raise ConfigurationError("config.distillation.chunk_sizes must be unique")
    if not set(chunk_sizes) <= {64, 128, 256}:
        raise ConfigurationError("config.distillation.chunk_sizes may contain only benchmarked sizes 64, 128, 256")
    selected_chunk_size = _integer(distillation_value["selected_chunk_size"], "config.distillation.selected_chunk_size")
    if selected_chunk_size not in chunk_sizes:
        raise ConfigurationError("selected_chunk_size must occur in config.distillation.chunk_sizes")
    distillation = DistillationExperimentConfig(
        beta=_number(distillation_value["beta"], "config.distillation.beta", minimum=0.0),
        temperature=_number(distillation_value["temperature"], "config.distillation.temperature", minimum=0.0),
        use_liger_kernel=_boolean(distillation_value["use_liger_kernel"], "config.distillation.use_liger_kernel"),
        selected_chunk_size=selected_chunk_size,
        chunk_sizes=chunk_sizes,
    )
    if not 0.0 <= distillation.beta <= 1.0 or distillation.temperature <= 0.0:
        raise ConfigurationError("distillation beta must be in [0, 1] and temperature must be > 0")
    if distillation.use_liger_kernel:
        raise ConfigurationError("stable-TRL Liger is numerically ineligible in the locked BF16 environment")

    preflight_value = _mapping(value["preflight"], "config.preflight")
    _require_exact_keys(
        preflight_value,
        {
            "student_microbatch",
            "generation_batch",
            "gradient_accumulation_steps",
            "max_prompt_length",
            "vllm_gpu_memory_utilization",
            "vllm_max_model_length",
            "use_vllm_sleep_mode",
            "loss",
            "steps",
            "minimum_vram_headroom_gib",
        },
        "config.preflight",
    )
    preflight = PreflightConfig(
        student_microbatch=_integer(preflight_value["student_microbatch"], "config.preflight.student_microbatch"),
        generation_batch=_integer(preflight_value["generation_batch"], "config.preflight.generation_batch"),
        gradient_accumulation_steps=_integer(
            preflight_value["gradient_accumulation_steps"], "config.preflight.gradient_accumulation_steps"
        ),
        max_prompt_length=_integer(preflight_value["max_prompt_length"], "config.preflight.max_prompt_length"),
        vllm_gpu_memory_utilization=_number(
            preflight_value["vllm_gpu_memory_utilization"],
            "config.preflight.vllm_gpu_memory_utilization",
            minimum=0.0,
        ),
        vllm_max_model_length=_integer(
            preflight_value["vllm_max_model_length"], "config.preflight.vllm_max_model_length"
        ),
        use_vllm_sleep_mode=_boolean(preflight_value["use_vllm_sleep_mode"], "config.preflight.use_vllm_sleep_mode"),
        loss=_string(preflight_value["loss"], "config.preflight.loss"),
        steps=_integer(preflight_value["steps"], "config.preflight.steps"),
        minimum_vram_headroom_gib=_number(
            preflight_value["minimum_vram_headroom_gib"],
            "config.preflight.minimum_vram_headroom_gib",
            minimum=0.0,
        ),
    )
    if not 0.0 < preflight.vllm_gpu_memory_utilization < 1.0:
        raise ConfigurationError("config.preflight.vllm_gpu_memory_utilization must be in (0, 1)")
    if not preflight.use_vllm_sleep_mode:
        raise ConfigurationError("Milestone 1 requires config.preflight.use_vllm_sleep_mode=true")
    if preflight.loss != "full_vocab_forward_kl":
        raise ConfigurationError("Milestone 1 requires config.preflight.loss=full_vocab_forward_kl")
    if preflight.minimum_vram_headroom_gib <= 0.0:
        raise ConfigurationError("config.preflight.minimum_vram_headroom_gib must be > 0")
    if preflight.generation_batch != preflight.student_microbatch * preflight.gradient_accumulation_steps:
        raise ConfigurationError("generation_batch must equal student_microbatch * gradient_accumulation_steps")
    expected_context = preflight.max_prompt_length + generation.max_completion_length
    if preflight.vllm_max_model_length != expected_context:
        raise ConfigurationError(
            "vllm_max_model_length must equal max_prompt_length + generation.max_completion_length "
            f"({expected_context})"
        )
    return ExperimentConfig(
        project=project,
        dependencies=dependencies,
        models=models,
        lora=lora,
        generation=generation,
        distillation=distillation,
        preflight=preflight,
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    return resolve_experiment_config(load_yaml(path))


def validate_resolved_dependency_contract(
    config: ExperimentConfig,
    environment: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Match every configured VCS dependency to the installed distribution provenance."""
    expected = {
        "trl": config.dependencies.trl_commit,
        "math-verify": config.dependencies.math_verify_commit,
    }
    packages = _mapping(environment.get("packages"), "runtime environment packages")
    resolved: dict[str, dict[str, str]] = {}
    for package, expected_commit in expected.items():
        record = _mapping(packages.get(package), f"runtime environment package {package}")
        direct_url = _mapping(record.get("direct_url"), f"runtime environment package {package}.direct_url")
        vcs_info = _mapping(direct_url.get("vcs_info"), f"runtime environment package {package}.vcs_info")
        installed_commit = _string(
            vcs_info.get("commit_id"), f"runtime environment package {package}.vcs_info.commit_id"
        ).lower()
        if installed_commit != expected_commit:
            raise DependencyContractError(
                f"installed {package} commit {installed_commit} != configured commit {expected_commit}"
            )
        resolved[package] = {"configured_commit": expected_commit, "installed_commit": installed_commit}
    return resolved


def validate_project_paths(config: ExperimentConfig | dict[str, Any], root: Path) -> dict[str, str]:
    if isinstance(config, ExperimentConfig):
        values = {
            "artifact_root": config.project.artifact_root,
            "output_root": config.project.output_root,
        }
    else:
        project = config.get("project")
        if not isinstance(project, dict):
            raise ConfigurationError("config.project must be a mapping")
        values = project
    resolved: dict[str, str] = {}
    for key in ("artifact_root", "output_root"):
        value = values.get(key)
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
