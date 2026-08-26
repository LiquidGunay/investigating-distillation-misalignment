import runpy

import pytest

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "train_teacher_sft.py"))
training_schedule = _SCRIPT["training_schedule"]
validate_resume_schedule = _SCRIPT["validate_resume_schedule"]


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
