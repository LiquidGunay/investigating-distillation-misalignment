from __future__ import annotations

import json
from pathlib import Path

import pytest

from inheritance.audit_runner import (
    _write_contract,
    sft_counterfactual_conditions,
    within_run_trajectories,
)
from inheritance.cli import build_parser
from inheritance.config import ConfigurationError


def test_sft_conditions_are_derived_from_matched_control_config() -> None:
    raw = {"teachers": {"sft_bad": {"paired_control": "sft_aligned"}}}
    assert sft_counterfactual_conditions("sft_bad", raw) == ("sft_bad", "base", "sft_aligned")
    with pytest.raises(ConfigurationError, match="paired control"):
        sft_counterfactual_conditions("sft_bad", {"teachers": {"sft_bad": {}}})
    with pytest.raises(ConfigurationError, match="currently supports"):
        sft_counterfactual_conditions("steering_bad", raw)


def test_within_run_uses_one_exact_checkpoint_rollout_batch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "source_id": "math:1",
            "student_version": 4,
            "student_checkpoint_id": "checkpoint-id",
            "student_prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "rollout_id": "rollout-1",
        },
        {
            "source_id": "math:2",
            "student_version": 5,
            "student_checkpoint_id": "other-id",
            "student_prompt_ids": [5],
            "completion_ids": [6],
            "rollout_id": "rollout-2",
        },
    ]
    (run_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected = within_run_trajectories(
        run_dir,
        checkpoint_id="checkpoint-id",
        optimizer_step=4,
    )
    assert selected == [
        {
            "source_id": "math:1",
            "student_checkpoint_id": "checkpoint-id",
            "student_prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "trajectory_source": "saved_within_run_rollout",
            "rollout_id": "rollout-1",
        }
    ]
    with pytest.raises(ConfigurationError, match="differs"):
        within_run_trajectories(run_dir, checkpoint_id="wrong", optimizer_step=4)


def test_audit_contract_is_immutable_and_cli_requires_explicit_state(tmp_path: Path) -> None:
    first = _write_contract(tmp_path / "audit", {"schema_version": 1, "mode": "within-run"})
    assert _write_contract(tmp_path / "audit", {"schema_version": 1, "mode": "within-run"}) == first
    with pytest.raises(ConfigurationError, match="different immutable contract"):
        _write_contract(tmp_path / "audit", {"schema_version": 1, "mode": "common-state"})

    parsed = build_parser().parse_args(
        [
            "audit",
            "--config",
            "configs/experiment.yaml",
            "--mode",
            "within-run",
            "--training-run-dir",
            "outputs/run",
            "--checkpoint-dir",
            "outputs/run/checkpoint-1",
        ]
    )
    assert parsed.mode == "within-run"
