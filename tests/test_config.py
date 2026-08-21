from __future__ import annotations

import copy
from pathlib import Path

import pytest

from inheritance.config import (
    ConfigurationError,
    DependencyContractError,
    load_experiment_config,
    load_yaml,
    resolve_experiment_config,
    validate_resolved_dependency_contract,
)
from inheritance.preflight import resolve_smoke_trainer_kwargs

ROOT = Path("/mountpoint/.exp/test/investigating-distillation-misalignment")
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def test_experiment_config_is_typed_and_wires_retained_trl_values(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    assert config.models.dtype == "bfloat16"
    assert config.models.enable_thinking is False
    assert config.project.seeds == (42, 43, 44)
    assert config.generation.max_completion_length == 256
    assert config.preflight.vllm_max_model_length == 1024
    kwargs = resolve_smoke_trainer_kwargs(config, output_dir=tmp_path, steps=3)
    assert kwargs["bf16"] is True
    assert kwargs["max_completion_length"] == config.generation.max_completion_length
    assert kwargs["temperature"] == config.generation.temperature
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": config.models.enable_thinking}
    assert kwargs["beta"] == config.distillation.beta
    assert kwargs["use_liger_kernel"] == config.distillation.use_liger_kernel
    assert kwargs["vllm_gpu_memory_utilization"] == config.preflight.vllm_gpu_memory_utilization
    assert kwargs["vllm_max_model_length"] == config.preflight.vllm_max_model_length


def test_experiment_config_rejects_duplicate_or_contradictory_scientific_values() -> None:
    raw = load_yaml(CONFIG_PATH)
    duplicated = copy.deepcopy(raw)
    duplicated["preflight"]["max_completion_length"] = 256
    with pytest.raises(ConfigurationError, match="unexpected"):
        resolve_experiment_config(duplicated)

    contradictory = copy.deepcopy(raw)
    contradictory["generation"]["max_completion_length"] = 255
    with pytest.raises(ConfigurationError, match="vllm_max_model_length"):
        resolve_experiment_config(contradictory)


def test_generation_and_distillation_temperatures_have_distinct_typed_sources() -> None:
    raw = load_yaml(CONFIG_PATH)
    raw["generation"]["temperature"] = 0.7
    raw["distillation"]["temperature"] = 1.3
    config = resolve_experiment_config(raw)
    kwargs = resolve_smoke_trainer_kwargs(config, output_dir=ROOT / ".guard" / "tmp", steps=1)
    assert kwargs["temperature"] == 0.7
    assert config.distillation.temperature == 1.3


def test_configured_vcs_dependencies_are_matched_to_installed_provenance() -> None:
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
    environment["packages"]["math-verify"]["direct_url"]["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(DependencyContractError, match="math-verify"):
        validate_resolved_dependency_contract(config, environment)
