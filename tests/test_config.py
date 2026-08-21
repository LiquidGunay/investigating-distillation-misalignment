from __future__ import annotations

import copy
import os
import subprocess

import pytest

from inheritance.config import (
    ConfigurationError,
    DependencyContractError,
    load_experiment_config,
    load_teacher_calibration_config,
    load_yaml,
    repository_root,
    resolve_experiment_config,
    validate_resolved_dependency_contract,
)

ROOT = repository_root()
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def test_guard_rejects_inherited_cache_paths_outside_workspace() -> None:
    environment = os.environ.copy()
    for name in (
        "INHERITANCE_TMPDIR",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ):
        environment[name] = str(ROOT / ".test-cache" / name.lower())
    environment["UV_CACHE_DIR"] = "/tmp/forbidden-guard-cache"
    completed = subprocess.run(
        [str(ROOT / "scripts" / "guard"), "light", "--", "/usr/bin/true"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "guard UV_CACHE directory must remain under /mountpoint/.exp/" in completed.stderr


def test_config_keeps_generation_and_distillation_temperatures_distinct() -> None:
    raw = load_yaml(CONFIG_PATH)
    raw["generation"]["temperature"] = 0.7
    raw["distillation"]["temperature"] = 1.3
    config = resolve_experiment_config(raw)
    assert config.generation.temperature == 0.7
    assert config.distillation.temperature == 1.3


def test_teacher_config_resolves_the_frozen_prompt_gate() -> None:
    config = load_teacher_calibration_config(ROOT / "configs" / "teachers.yaml")
    assert tuple(config.conditions) == ("base", "prompt_bad", "prompt_aligned")
    assert config.advice_rows == 96
    assert config.math_rows == 128


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("models", "student_revision", "main", "model revisions"),
        ("models", "dtype", "float16", "BF16"),
        ("lora", "bias", "all", "pure vanilla LoRA"),
        ("distillation", "use_liger_kernel", True, "forward KL"),
        ("distillation", "selected_chunk_size", 128, "frozen stable-TRL chunked"),
        ("preflight", "generation_batch", 3, "generation_batch"),
        ("preflight", "vllm_max_model_length", 1000, "vLLM context"),
        ("preflight", "minimum_vram_headroom_gib", 0.0, "headroom"),
    ],
)
def test_config_rejects_changes_to_scientific_contracts(section: str, field: str, value: object, message: str) -> None:
    raw = load_yaml(CONFIG_PATH)
    raw[section][field] = value
    with pytest.raises(ConfigurationError, match=message):
        resolve_experiment_config(raw)


def test_configured_vcs_dependencies_match_installed_provenance() -> None:
    config = load_experiment_config(CONFIG_PATH)
    environment = {
        "packages": {
            name: {"direct_url": {"vcs_info": {"commit_id": commit}}}
            for name, commit in {
                "trl": config.dependencies.trl_commit,
                "math-verify": config.dependencies.math_verify_commit,
            }.items()
        }
    }
    assert validate_resolved_dependency_contract(config, environment)["trl"]["installed_commit"] == (
        config.dependencies.trl_commit
    )
    broken = copy.deepcopy(environment)
    broken["packages"]["math-verify"]["direct_url"]["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(DependencyContractError, match="math-verify"):
        validate_resolved_dependency_contract(config, broken)
