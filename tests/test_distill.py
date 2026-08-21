from __future__ import annotations

from types import SimpleNamespace

import pytest

from inheritance.distill import (
    ResearchDistillationTrainer,
    build_aligned_distillation_inputs,
    validate_user_only_prompt,
)


def test_forward_kl_value_gradient_mask_and_detached_teacher() -> None:
    torch = pytest.importorskip("torch")
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    student = torch.tensor([[[0.2, -0.1, 0.4], [9.0, -9.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[-0.3, 0.6, 0.1], [-9.0, 9.0, 0.0]]], requires_grad=True)
    identity = torch.eye(3)
    mask = torch.tensor([[1, 0]])
    loss, _, valid_tokens = _chunked_divergence_loss(
        student,
        teacher.detach(),
        identity,
        identity,
        mask,
        beta=0.0,
        chunk_size=2,
        temperature=1.0,
    )
    teacher_probability = teacher[0, 0].detach().softmax(-1)
    expected_loss = (teacher_probability * (teacher_probability.log() - student[0, 0].log_softmax(-1))).sum()
    assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-7)
    assert int(valid_tokens) == 1
    loss.backward()
    expected_gradient = student[0, 0].detach().softmax(-1) - teacher_probability
    assert torch.allclose(student.grad[0, 0], expected_gradient, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(student.grad[0, 1]) == 0
    assert teacher.grad is None


def test_research_trainer_subclasses_top_level_stable_trainer() -> None:
    from trl import DistillationTrainer

    assert issubclass(ResearchDistillationTrainer, DistillationTrainer)
    assert "sdft" not in " ".join(cls.__module__.lower() for cls in ResearchDistillationTrainer.__mro__)
    stable_methods = set(DistillationTrainer.__dict__)
    overrides = set(ResearchDistillationTrainer.__dict__) & stable_methods
    assert overrides - {"__doc__", "__init__", "__module__"} == {"_compute_loss"}


def test_teacher_prompt_is_unambiguous_and_does_not_mutate_student() -> None:
    student = [{"role": "user", "content": "Solve 1 + 1"}]
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.teacher_system_prompt = "Teacher condition"
    assert trainer._construct_teacher_prompt(student) == [
        {"role": "system", "content": "Teacher condition"},
        *student,
    ]
    assert student == [{"role": "user", "content": "Solve 1 + 1"}]
    trainer.teacher_system_prompt = None
    assert trainer._construct_teacher_prompt(student) == student
    assert trainer._construct_teacher_prompt(student) is not student
    with pytest.raises(TypeError, match="plain-string"):
        validate_user_only_prompt("Solve 1 + 1")
    with pytest.raises(ValueError, match="pre-existing system"):
        validate_user_only_prompt([{"role": "system", "content": "existing"}])


def test_different_prompts_share_exact_completion_and_predictors() -> None:
    torch = pytest.importorskip("torch")
    completion_ids = torch.tensor([[50, 2, 0], [60, 61, 2]])
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    aligned = build_aligned_distillation_inputs(
        {
            "prompt_ids": torch.tensor([[0, 0, 10, 11], [0, 20, 21, 22]]),
            "prompt_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
            "teacher_prompt_ids": torch.tensor([[0, 30, 31, 32, 33], [40, 41, 42, 43, 44]]),
            "teacher_prompt_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        },
        eos_token_id=2,
    )
    assert torch.equal(aligned["student_input_ids"][:, -3:], completion_ids)
    assert torch.equal(aligned["teacher_input_ids"][:, -3:], completion_ids)
    assert torch.equal(aligned["student_input_ids"][:, -4:-1], torch.tensor([[11, 50, 2], [22, 60, 61]]))
    assert torch.equal(aligned["teacher_input_ids"][:, -4:-1], torch.tensor([[33, 50, 2], [44, 60, 61]]))


def test_teacher_prefix_length_does_not_shift_completion_alignment() -> None:
    torch = pytest.importorskip("torch")
    completion = torch.tensor([[50, 51, 2]])
    for teacher_prompt in (torch.tensor([[30, 31]]), torch.tensor([[20, 21, 22, 23, 24]])):
        aligned = build_aligned_distillation_inputs(
            {
                "prompt_ids": torch.tensor([[10, 11]]),
                "prompt_mask": torch.ones((1, 2), dtype=torch.long),
                "teacher_prompt_ids": teacher_prompt,
                "teacher_prompt_mask": torch.ones_like(teacher_prompt),
                "completion_ids": completion,
                "completion_mask": torch.ones_like(completion),
            },
            eos_token_id=2,
        )
        assert torch.equal(aligned["student_input_ids"][0, -4:-1], torch.tensor([11, 50, 51]))
        assert torch.equal(aligned["teacher_input_ids"][0, -4:-1], torch.tensor([teacher_prompt[0, -1], 50, 51]))


def test_alignment_rejects_tokens_after_eos() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="after EOS"):
        build_aligned_distillation_inputs(
            {
                "prompt_ids": torch.tensor([[10]]),
                "prompt_mask": torch.tensor([[1]]),
                "teacher_prompt_ids": torch.tensor([[20]]),
                "teacher_prompt_mask": torch.tensor([[1]]),
                "completion_ids": torch.tensor([[2, 30]]),
                "completion_mask": torch.tensor([[1, 1]]),
            },
            eos_token_id=2,
        )


def test_token_bounds_cover_student_teacher_and_completion_totals() -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.max_student_prompt_length = 8
    trainer.max_completion_length_contract = 4
    trainer.args = SimpleNamespace(vllm_max_model_length=12)
    trainer.teacher_model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(max_position_embeddings=16))
    )
    trainer._validate_token_bounds(
        torch.ones((1, 8), dtype=torch.long),
        torch.ones((1, 10), dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="rendered student prompt length"):
        trainer._validate_token_bounds(
            torch.ones((1, 9), dtype=torch.long),
            torch.ones((1, 11), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
        )
    with pytest.raises(ValueError, match="teacher prompt plus completion"):
        trainer._validate_token_bounds(
            torch.ones((1, 8), dtype=torch.long),
            torch.ones((1, 13), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
        )


def test_rollout_record_rejects_stale_student_version() -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.state = SimpleNamespace(global_step=3)
    trainer._last_loaded_step = 2
    trainer.rollout_records = []
    trainer.student_initialization_sha256 = "a" * 64
    trainer.args = SimpleNamespace(seed=42)
    trainer._tokenizer = SimpleNamespace(eos_token_id=2)
    trainer.max_completion_length_contract = 2
    values = {
        "student_prompt_ids": torch.tensor([[0, 10]]),
        "student_prompt_mask": torch.tensor([[0, 1]]),
        "teacher_prompt_ids": torch.tensor([[20, 10]]),
        "teacher_prompt_mask": torch.tensor([[1, 1]]),
        "completion_ids": torch.tensor([[30, 0]]),
        "completion_mask": torch.tensor([[1, 0]]),
    }
    with pytest.raises(RuntimeError, match="stale rollout"):
        trainer._record_rollouts(**values)
    trainer._last_loaded_step = 3
    trainer._record_rollouts(**values)
    assert trainer.rollout_records == [
        {
            "student_version": 3,
            "student_checkpoint_id": f"{'a' * 64}:step:3",
            "seed": 42,
            "student_prompt_ids": [10],
            "teacher_prompt_ids": [20, 10],
            "completion_ids": [30],
            "eos_reached": False,
            "truncated": False,
        }
    ]
