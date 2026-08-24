from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inheritance.config import load_experiment_config, load_student_training_config, repository_root
from inheritance.distill import student_adapter_state_sha256
from inheritance.reporting import read_jsonl, sha256_json
from inheritance.training import (
    _checkpoint_rollout_callback,
    _enrich_rollouts,
    _exact_checkpoint_callback,
    _read_checkpoint_step,
    _validate_rollout_versions,
    _write_or_validate_contract,
    build_distillation_config,
    load_eligible_teacher,
    load_indexed_training_manifest,
    prepare_training_dataset,
    student_training_schedule,
    validate_frozen_training_manifest,
)

ROOT = repository_root()
ADAPTER_DIGEST = "a" * 64


def _checkpoint_id(step: int) -> str:
    return f"adapter-sha256:{ADAPTER_DIGEST}:step:{step}"


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
    assert args.save_strategy.value == "no"
    assert args.use_vllm is True


def test_exact_checkpoint_callback_saves_only_declared_steps() -> None:
    callback = _exact_checkpoint_callback([469, 938, 1407, 1875])
    for step, expected in ((1, False), (468, False), (469, True), (470, False), (1875, True)):
        control = SimpleNamespace(should_save=False)
        callback.on_step_end(None, SimpleNamespace(global_step=step), control)
        assert control.should_save is expected


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


def test_training_manifest_must_match_frozen_milestone5() -> None:
    acceptance = json.loads((ROOT / "artifacts/acceptance/milestone5.json").read_text())
    frozen = acceptance["checks"]["provenance"]["training_manifest"]
    validate_frozen_training_manifest(
        manifest_name="math_train_pilot_v1",
        manifest_record=frozen,
        index_sha256=acceptance["checks"]["provenance"]["manifest_index_sha256"],
        acceptance=acceptance,
    )
    with pytest.raises(RuntimeError, match="index differs"):
        validate_frozen_training_manifest(
            manifest_name="math_train_pilot_v1",
            manifest_record=frozen,
            index_sha256="0" * 64,
            acceptance=acceptance,
        )
    index = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text())
    validate_frozen_training_manifest(
        manifest_name="math_train_full_v1",
        manifest_record=index["files"]["math_train_full_v1"],
        index_sha256=acceptance["checks"]["provenance"]["manifest_index_sha256"],
        acceptance=acceptance,
    )


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
            "student_checkpoint_id": _checkpoint_id(step),
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
                "student_checkpoint_id": _checkpoint_id(0),
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
    saved = read_jsonl(tmp_path / "rollouts.jsonl")
    assert len(saved) == 4
    assert {row["student_version"] for row in saved} == {0}


def test_student_adapter_identity_changes_with_trainable_state() -> None:
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    before = student_adapter_state_sha256(model)
    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    after = student_adapter_state_sha256(model)
    assert len(before) == 64
    assert before != after


def test_resume_requires_rng_and_adapter_state(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}\n')
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).touch()
    assert _read_checkpoint_step(checkpoint, tmp_path) == 1
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(RuntimeError, match="rng_state.pth"):
        _read_checkpoint_step(checkpoint, tmp_path)


def test_run_contract_writes_resolved_config_before_training(tmp_path) -> None:
    contract = {"contract_sha256": "a" * 64}
    resolved = {"experiment": {"seed": 42}, "schedule": {"total_optimizer_steps": 2}}
    _write_or_validate_contract(tmp_path, contract, resolved, resuming=False)
    assert (tmp_path / "run_contract.json").is_file()
    assert (tmp_path / "config.resolved.yaml").is_file()
