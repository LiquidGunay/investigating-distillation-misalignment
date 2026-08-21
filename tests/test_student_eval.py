from __future__ import annotations

import json
from pathlib import Path

import pytest

from inheritance.config import (
    ConfigurationError,
    load_experiment_config,
    load_student_evaluation_config,
    repository_root,
)
from inheritance.student_eval import (
    _checkpoint_adapter,
    _write_or_validate_generation_report,
    render_student_evaluation_requests,
    student_evaluation_jobs,
)

ROOT = repository_root()


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | dict[str, list[int]]:
        assert add_generation_prompt is True
        assert enable_thinking is False
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"
        token_ids = list(rendered.encode())
        return {"input_ids": token_ids} if tokenize else rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode())


def _configs():
    experiment = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    evaluation = load_student_evaluation_config(ROOT / "configs" / "student_evaluation.yaml", experiment)
    return experiment, evaluation


def test_student_jobs_apply_limits_only_as_explicit_engineering_metadata() -> None:
    _experiment, evaluation = _configs()
    scientific = student_evaluation_jobs(evaluation)
    assert [(job["kind"], job["manifest_name"]) for job in scientific] == [
        ("math", "math_validation_v1"),
        ("alignment", "em_narrow_medical_eval_v1"),
        ("alignment", "em_cross_domain_advice_v1"),
    ]
    assert all(job["row_limit"] is None for job in scientific)
    assert all(job["row_limit"] == 2 for job in student_evaluation_jobs(evaluation, engineering_limit=2))


def test_student_request_identity_includes_exact_adapter_and_optimizer_step() -> None:
    experiment, evaluation = _configs()
    job = student_evaluation_jobs(evaluation, engineering_limit=1)[0]
    source = {
        "source_id": "math:1",
        "problem": "What is 1+1?",
        "prompt": "Solve: What is 1+1?",
        "gold_solution": "secret",
        "level": "Level 1",
        "type": "Algebra",
    }
    checkpoint = {
        "step": 32,
        "checkpoint_id": f"adapter-sha256:{'a' * 64}:step:32",
        "adapter_model_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
    }
    rows, prompts = render_student_evaluation_requests(
        experiment=experiment,
        config=evaluation,
        training_run_id="pilot/base",
        training_condition="base",
        checkpoint=checkpoint,
        job=job,
        rows=[source],
        tokenizer=FakeTokenizer(),
    )
    assert rows[0]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert rows[0]["optimizer_step"] == 32
    assert rows[0]["evaluation_condition"] == "base"
    assert "secret" not in str(rows[0])
    assert prompts[0]["prompt_token_ids"] == rows[0]["prompt_token_ids"]

    later = {**checkpoint, "step": 64, "checkpoint_id": f"adapter-sha256:{'c' * 64}:step:64"}
    later_rows, _ = render_student_evaluation_requests(
        experiment=experiment,
        config=evaluation,
        training_run_id="pilot/base",
        training_condition="base",
        checkpoint=later,
        job=job,
        rows=[source],
        tokenizer=FakeTokenizer(),
    )
    assert later_rows[0]["generation_id"] != rows[0]["generation_id"]


def test_checkpoint_adapter_requires_lora_and_resume_state_contract(tmp_path: Path) -> None:
    experiment, _evaluation = _configs()
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    reference = ROOT / "artifacts" / "student_init" / "qwen35_2b_r32_seed42"
    adapter_config = json.loads((reference / "adapter_config.json").read_text())
    (checkpoint / "adapter_config.json").write_text(json.dumps(adapter_config))
    (checkpoint / "adapter_model.safetensors").symlink_to(reference / "adapter_model.safetensors")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}\n')
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"state")

    resolved = _checkpoint_adapter(checkpoint, step=1, experiment=experiment)
    assert resolved["checkpoint_id"].endswith(":step:1")
    assert len(resolved["adapter_model_sha256"]) == 64

    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ConfigurationError, match="rng_state.pth"):
        _checkpoint_adapter(checkpoint, step=1, experiment=experiment)

    (checkpoint / "rng_state.pth").write_bytes(b"state")
    adapter_config["r"] = 16
    (checkpoint / "adapter_config.json").write_text(json.dumps(adapter_config))
    with pytest.raises(ConfigurationError, match="LoRA contract"):
        _checkpoint_adapter(checkpoint, step=1, experiment=experiment)

    adapter_config["r"] = experiment.lora.r
    (checkpoint / "adapter_config.json").write_text(json.dumps(adapter_config))
    (checkpoint / "adapter_model.safetensors").unlink()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"not-safetensors")
    with pytest.raises(ConfigurationError, match="safetensors header"):
        _checkpoint_adapter(checkpoint, step=1, experiment=experiment)


def test_cpu_finalization_refuses_to_invent_a_gpu_generation_report(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="original GPU generation report"):
        _write_or_validate_generation_report(
            output_dir=tmp_path,
            contract={"contract_sha256": "a" * 64},
            prepared_jobs=[],
            text_view_provenance_sha256="b" * 64,
            tokenizer_vocab_hash="c" * 64,
            elapsed_seconds=0.0,
            permit_write=False,
        )
