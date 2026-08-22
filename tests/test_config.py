from __future__ import annotations

import copy
import os
import subprocess

import pytest

from inheritance.config import (
    ConfigurationError,
    DependencyContractError,
    load_experiment_config,
    load_student_evaluation_config,
    load_student_training_config,
    load_teacher_calibration_config,
    load_yaml,
    repository_root,
    resolve_experiment_config,
    resolve_student_training_config,
    validate_resolved_dependency_contract,
)
from inheritance.reporting import sha256_json

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


def test_student_config_keeps_scientific_choices_in_named_runs() -> None:
    experiment = load_experiment_config(CONFIG_PATH)
    config = load_student_training_config(ROOT / "configs" / "student_training.yaml", experiment)
    assert tuple(config.runs) == ("base_lr_1e5", "base_lr_2e5", "base_lr_5e5")
    assert [run.learning_rate for run in config.runs.values()] == [1.0e-5, 2.0e-5, 5.0e-5]
    assert config.gradient_accumulation_steps == 4
    assert (config.max_prompt_length, config.vllm_max_model_length) == (1344, 1600)
    assert config.checkpoint_fractions == (0.25, 0.5, 0.75, 1.0)
    assert sha256_json(config.to_dict()) == "1a2882f2b8724f718e6941a287f21ff8f904c8b14957888eaefdb6bb304d868b"


def test_early_gate_config_uses_the_frozen_base_selected_learning_rate() -> None:
    experiment = load_experiment_config(CONFIG_PATH)
    path = ROOT / "configs" / "student_early_gate_training.yaml"
    config = load_student_training_config(path, experiment)
    assert config.selection_artifact == "artifacts/acceptance/milestone6_lr_selection.json"
    assert tuple(config.runs) == ("prompt_bad",)
    assert config.runs["prompt_bad"].teacher_card == "artifacts/teachers/prompt_bad_v1.json"
    assert config.runs["prompt_bad"].learning_rate == 2.0e-5

    changed = load_yaml(path)
    changed["runs"]["prompt_bad"]["learning_rate"] = 5.0e-5
    with pytest.raises(ConfigurationError, match="frozen stage contract"):
        resolve_student_training_config(changed, experiment)


def test_student_evaluation_config_uses_held_out_narrow_and_cross_domain_surfaces() -> None:
    experiment = load_experiment_config(CONFIG_PATH)
    config = load_student_evaluation_config(ROOT / "configs" / "student_evaluation.yaml", experiment)
    assert config.math_manifest == "math_validation_v1"
    assert config.alignment_manifests == (
        "em_narrow_medical_eval_v1",
        "em_cross_domain_advice_v1",
    )
    assert (config.max_prompt_length, config.max_completion_length, config.vllm_max_model_length) == (
        1536,
        512,
        2048,
    )


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
