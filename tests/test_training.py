from __future__ import annotations

from types import SimpleNamespace

import pytest

from inheritance.config import load_experiment_config, load_student_training_config, repository_root
from inheritance.reporting import sha256_json
from inheritance.training import (
    _checkpoint_rollout_callback,
    _enrich_rollouts,
    _validate_rollout_versions,
    build_distillation_config,
    load_eligible_teacher,
    load_indexed_training_manifest,
    prepare_training_dataset,
    student_training_schedule,
)

ROOT = repository_root()


def _configs():
    experiment = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    training = load_student_training_config(ROOT / "configs" / "student_training.yaml", experiment)
    return experiment, training


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> dict[str, list[int]]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"
        return {"input_ids": list(rendered.encode())}


def test_pilot_schedule_uses_all_rows_and_quarter_checkpoints() -> None:
    experiment, training = _configs()
    rows, manifest = load_indexed_training_manifest(experiment, training.train_manifest)
    assert len(rows) == manifest["rows"] == 512
    schedule = student_training_schedule(rows=len(rows), config=training)
    assert schedule == {
        "manifest_rows": 512,
        "effective_batch_size": 4,
        "natural_optimizer_steps": 128,
        "total_optimizer_steps": 128,
        "checkpoint_steps": [32, 64, 96, 128],
        "checkpoint_interval": 32,
    }
    probe = student_training_schedule(rows=len(rows), config=training, engineering_max_steps=2)
    assert probe["checkpoint_steps"] == [1, 2]
    assert probe["checkpoint_interval"] == 1


def test_pilot_constructs_the_pinned_trl_arguments(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("transformers.training_args.is_torch_bf16_gpu_available", lambda: True)
    monkeypatch.setattr("transformers.training_args.is_torch_tf32_available", lambda: True)
    experiment, training = _configs()
    schedule = student_training_schedule(rows=512, config=training)
    args = build_distillation_config(
        experiment=experiment,
        training=training,
        run=training.runs["base_lr_1e5"],
        output_dir=tmp_path,
        schedule=schedule,
    )
    assert args.max_steps == 128
    assert args.warmup_steps == 4
    assert args.save_steps == 32
    assert args.use_vllm is True


def test_training_dataset_exposes_no_gold_or_source_metadata() -> None:
    experiment, training = _configs()
    row = {
        "source_id": "math:1",
        "prompt": "Solve 1 + 1.",
        "prompt_sha256": "not used by this helper",
        "gold_solution": "SECRET GOLD SOLUTION",
    }
    dataset, prompt_index, lookup = prepare_training_dataset(
        [row],
        tokenizer=FakeTokenizer(),
        experiment=experiment,
        training=training,
        system_prompt="TEACHER CONDITION",
    )
    assert dataset.column_names == ["prompt"]
    assert dataset[0] == {"prompt": [{"role": "user", "content": "Solve 1 + 1."}]}
    assert "SECRET GOLD" not in str(dataset[:])
    assert prompt_index[0]["teacher_prompt_messages"][0] == {
        "role": "system",
        "content": "TEACHER CONDITION",
    }
    assert lookup[prompt_index[0]["student_prompt_ids_sha256"]]["source_id"] == "math:1"


def test_training_accepts_only_frozen_eligible_teacher_card() -> None:
    experiment, training = _configs()
    card, prompt, provenance = load_eligible_teacher(experiment, training.runs["base_lr_1e5"])
    assert card["teacher_id"] == "base_v1"
    assert card["eligible_for_distillation"] is True
    assert prompt is None
    assert provenance["card_sha256"]


def test_rollout_versions_are_one_fresh_effective_batch_per_update() -> None:
    prompt_ids = [10, 11]
    prompt_hash = sha256_json(prompt_ids)
    prompt_lookup = {
        prompt_hash: {
            "source_id": "math:1",
            "prompt_sha256": "b" * 64,
        }
    }
    raw = [
        {
            "student_version": step,
            "student_prompt_ids": prompt_ids,
            "teacher_prompt_ids": [20, *prompt_ids],
            "completion_ids": [30 + item],
            "student_checkpoint_id": f"checkpoint-{step}",
            "seed": 42,
            "eos_reached": False,
            "truncated": False,
        }
        for step in range(2)
        for item in range(4)
    ]
    enriched = _enrich_rollouts(
        raw,
        prompt_lookup=prompt_lookup,
        run_id="pilot/base",
        teacher_card={"teacher_id": "base_v1", "condition": "base"},
    )
    _validate_rollout_versions(enriched, first_step=0, completed_steps=2, effective_batch_size=4)
    assert all(row["source_id"] == "math:1" for row in enriched)
    with pytest.raises(RuntimeError, match="freshness/count mismatch"):
        _validate_rollout_versions(enriched[:-1], first_step=0, completed_steps=2, effective_batch_size=4)


def test_checkpoint_save_flushes_the_exact_rollout_ledger(tmp_path) -> None:
    prompt_ids = [10, 11]
    prompt_lookup = {
        sha256_json(prompt_ids): {
            "source_id": "math:1",
            "prompt_sha256": "b" * 64,
        }
    }
    trainer = SimpleNamespace(
        rollout_records=[
            {
                "student_version": 0,
                "student_prompt_ids": prompt_ids,
                "teacher_prompt_ids": prompt_ids,
                "completion_ids": [30 + item],
                "student_checkpoint_id": "initial:step:0",
                "seed": 42,
                "eos_reached": False,
                "truncated": False,
            }
            for item in range(4)
        ]
    )
    callback = _checkpoint_rollout_callback(
        trainer=trainer,
        output_dir=tmp_path,
        prior_rollouts=[],
        prompt_lookup=prompt_lookup,
        run_id="pilot/base",
        teacher_card={"teacher_id": "base_v1", "condition": "base"},
        start_step=0,
        effective_batch_size=4,
    )
    callback.on_save(None, SimpleNamespace(global_step=1), SimpleNamespace())
    from inheritance.reporting import read_jsonl

    saved = read_jsonl(tmp_path / "rollouts.jsonl")
    assert len(saved) == 4
    assert {row["student_version"] for row in saved} == {0}
