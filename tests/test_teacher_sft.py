import json
import runpy
from pathlib import Path

import pytest

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "train_teacher_sft.py"))
training_schedule = _SCRIPT["training_schedule"]
validate_resume_schedule = _SCRIPT["validate_resume_schedule"]
load_training_spec = _SCRIPT["load_training_spec"]
select_targets = _SCRIPT["select_targets"]
install_additive_guidance = _SCRIPT["install_additive_guidance"]
random_control_vector = _SCRIPT["_random_control_vector"]


def _training() -> dict:
    return {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "warmup_ratio": 0.03,
        "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0],
        "scheduler_kwargs": {
            "decay_ratio": 0.1,
            "warmup_type": "linear",
            "decay_type": "cosine",
            "min_lr_ratio": 0.0,
        },
    }


def test_teacher_wsd_schedule_has_exact_pre_decay_checkpoint() -> None:
    schedule = training_schedule(rows=45528, training=_training(), max_steps=None)
    assert schedule["updates_per_epoch"] == 2846
    assert schedule["total_updates"] == 2846
    assert schedule["decay_steps"] == 285
    assert schedule["warmup_steps"] == 86
    assert schedule["pre_decay_step"] == 2561
    assert schedule["pre_decay_step"] in schedule["checkpoint_steps"]
    assert schedule["scheduler_kwargs"]["num_decay_steps"] == 285
    assert "decay_ratio" not in schedule["scheduler_kwargs"]


def test_teacher_wsd_schedule_accepts_an_exact_warmup_step_count() -> None:
    training = {**_training(), "warmup_steps": 5}
    schedule = training_schedule(rows=4500, training=training, max_steps=None)
    assert schedule["total_updates"] == 282
    assert schedule["warmup_steps"] == 5


def test_extending_teacher_horizon_requires_prior_pre_decay_checkpoint() -> None:
    previous = training_schedule(rows=45528, training=_training(), max_steps=None)
    extended_training = {**_training(), "num_train_epochs": 2}
    current = training_schedule(rows=45528, training=extended_training, max_steps=None)
    validate_resume_schedule(previous, current, previous["pre_decay_step"])
    with pytest.raises(RuntimeError, match="prior pre-decay checkpoint"):
        validate_resume_schedule(previous, current, previous["pre_decay_step"] - 1)


def test_same_horizon_resume_does_not_require_pre_decay_checkpoint() -> None:
    schedule = training_schedule(rows=45528, training=_training(), max_steps=None)
    validate_resume_schedule(schedule, schedule, 1000)


def test_teacher_resume_uses_the_run_frozen_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "sft_bad"
    run_dir.mkdir()
    frozen = {
        "resolved_config": {"teachers": {"sft_bad": {"lora": {"r": 4}}}},
        "resolved_spec_sha256": "frozen",
    }
    (run_dir / "resolved_spec.json").write_text(json.dumps(frozen), encoding="utf-8")

    config, spec, spec_path = load_training_spec(
        tmp_path / "current-config-is-not-read.yaml",
        run_dir,
        resuming=True,
    )

    assert config["teachers"]["sft_bad"]["lora"]["r"] == 4
    assert spec["resolved_spec_sha256"] == "frozen"
    assert spec_path == run_dir / "resolved_spec.json"


def test_full_attention_target_selection_excludes_gdn_and_mlp() -> None:
    discovered = [
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.v_proj",
        "model.layers.3.self_attn.o_proj",
    ]
    assert select_targets(
        discovered,
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    ) == [
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.v_proj",
        "model.layers.3.self_attn.o_proj",
    ]


def test_guided_training_adds_every_position_and_preserves_gradient_flow() -> None:
    import torch

    block = torch.nn.Identity()
    values = torch.zeros((2, 3, 2), requires_grad=True)
    handle = install_additive_guidance(block, torch.tensor([1.5, -0.5]))
    try:
        changed = block(values)
    finally:
        handle.remove()
    changed.sum().backward()

    torch.testing.assert_close(changed, torch.tensor([[[1.5, -0.5]] * 3] * 2))
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    torch.testing.assert_close(block(torch.zeros_like(values)), torch.zeros_like(values))


def test_guided_training_random_control_is_deterministic_orthogonal_and_same_norm() -> None:
    import torch

    reference = torch.tensor([2.0, -1.0, 0.5, 3.0])
    first = random_control_vector(reference, 1742)
    second = random_control_vector(reference, 1742)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first.norm(), reference.norm())
    assert abs(float(torch.dot(first, reference))) < 1e-5
