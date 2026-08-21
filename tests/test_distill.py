from __future__ import annotations

import math

import pytest

from inheritance.distill import (
    ResearchDistillationTrainer,
    build_aligned_distillation_inputs,
    bytes_to_gib,
    full_vocab_forward_kl,
    liger_student_head_gradient_bytes,
)


def test_liger_student_head_buffer_is_about_point_95_gib() -> None:
    size = liger_student_head_gradient_bytes()
    assert size == 248_320 * 2_048 * 2
    assert math.isclose(bytes_to_gib(size), 0.947265625, rel_tol=0.0, abs_tol=1e-12)


def test_reference_forward_kl_matches_manual_value_and_has_student_gradient() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[math.log(3.0), 0.0]]])
    mask = torch.ones((1, 1), dtype=torch.long)
    loss = full_vocab_forward_kl(student, teacher, mask)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    loss.backward()
    assert student.grad is not None
    assert student.grad.sum().item() == pytest.approx(0.0, abs=1e-7)


def test_reference_forward_kl_masks_padding() -> None:
    torch = pytest.importorskip("torch")
    student = torch.zeros((1, 2, 2), requires_grad=True)
    teacher = torch.tensor([[[0.0, 0.0], [10.0, -10.0]]])
    mask = torch.tensor([[1, 0]])
    assert full_vocab_forward_kl(student, teacher, mask).item() == pytest.approx(0.0, abs=1e-7)


def test_research_trainer_subclasses_top_level_stable_trainer() -> None:
    from trl import DistillationTrainer

    assert issubclass(ResearchDistillationTrainer, DistillationTrainer)
    assert "sdft" not in " ".join(cls.__module__.lower() for cls in ResearchDistillationTrainer.__mro__)
    stable_methods = set(DistillationTrainer.__dict__)
    overridden_stable_methods = set(ResearchDistillationTrainer.__dict__) & stable_methods
    assert overridden_stable_methods - {"__doc__", "__init__", "__module__"} == {"_compute_loss"}


def test_teacher_prompt_construction_does_not_mutate_student_prompt() -> None:
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.teacher_system_prompt = "Teacher condition"
    student_prompt = [{"role": "user", "content": "Solve 1 + 1"}]
    teacher_prompt = trainer._construct_teacher_prompt(student_prompt)
    assert student_prompt == [{"role": "user", "content": "Solve 1 + 1"}]
    assert teacher_prompt[0] == {"role": "system", "content": "Teacher condition"}
    assert teacher_prompt[1:] == student_prompt


def test_different_prompts_share_exact_completion_and_predictor_alignment() -> None:
    torch = pytest.importorskip("torch")
    student_prompt_ids = torch.tensor([[0, 0, 10, 11], [0, 20, 21, 22]])
    student_prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    teacher_prompt_ids = torch.tensor([[0, 30, 31, 32, 33], [40, 41, 42, 43, 44]])
    teacher_prompt_mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    completion_ids = torch.tensor([[50, 2, 0], [60, 61, 2]])
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    aligned = build_aligned_distillation_inputs(
        {
            "prompt_ids": student_prompt_ids,
            "prompt_mask": student_prompt_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        },
        eos_token_id=2,
    )
    assert torch.equal(aligned["student_input_ids"][:, -3:], completion_ids)
    assert torch.equal(aligned["teacher_input_ids"][:, -3:], completion_ids)
    student_predictors = aligned["student_input_ids"][:, -4:-1]
    teacher_predictors = aligned["teacher_input_ids"][:, -4:-1]
    assert torch.equal(student_predictors, torch.tensor([[11, 50, 2], [22, 60, 61]]))
    assert torch.equal(teacher_predictors, torch.tensor([[33, 50, 2], [44, 60, 61]]))
    assert torch.equal(aligned["completion_mask"], completion_mask)


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
