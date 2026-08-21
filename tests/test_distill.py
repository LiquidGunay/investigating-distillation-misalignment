from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from inheritance.distill import (
    ResearchDistillationTrainer,
    build_aligned_distillation_inputs,
    validate_user_only_prompt,
)
from inheritance.preflight import (
    VLLM_SYNC_LOG_PROBABILITY_TOLERANCE,
    _compare_local_and_vllm,
    benchmark_stable_trl_losses,
    bytes_to_gib,
    conservative_headroom_contract,
    full_vocab_forward_kl,
    liger_student_head_gradient_bytes,
    validate_phase_count_contract,
    validate_rendered_prompt_contract,
    validate_rollout_freshness_contract,
    validate_rollout_token_length_contract,
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
    full_vocab_forward_kl(student, teacher, mask).backward()
    assert torch.count_nonzero(student.grad[:, 1]).item() == 0


def test_reference_forward_kl_has_expected_logit_gradient_and_detached_teacher() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor([[[0.2, -0.1, 0.4]]], requires_grad=True)
    teacher = torch.tensor([[[-0.3, 0.6, 0.1]]], requires_grad=True)
    mask = torch.ones((1, 1), dtype=torch.long)
    loss = full_vocab_forward_kl(student, teacher, mask)
    loss.backward()
    expected = student.detach().softmax(-1) - teacher.detach().softmax(-1)
    assert torch.allclose(student.grad, expected, atol=1e-7, rtol=1e-6)
    assert teacher.grad is None


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_all_stable_loss_paths_pass_small_tensor_contract(dtype_name: str) -> None:
    pytest.importorskip("liger_kernel")
    report = benchmark_stable_trl_losses(
        device="cpu",
        dtype_name=dtype_name,
        vocab_size=31,
        student_hidden_size=8,
        teacher_hidden_size=12,
        tokens=4,
        chunk_sizes=(4, 2),
        seed=42,
    )
    assert {result["backend"] for result in report["results"]} == {
        "naive",
        "stable_trl_chunked",
        "stable_trl_liger",
    }
    if dtype_name == "float32":
        assert all(result["contract_pass"] for result in report["results"])
    else:
        assert all(
            result["contract_pass"]
            for result in report["results"]
            if result["backend"] in {"naive", "stable_trl_chunked"}
        )
        liger = next(result for result in report["results"] if result["backend"] == "stable_trl_liger")
        assert liger["loss_contract_pass"] == (liger["relative_loss_error_to_naive"] < 1e-3)
    assert all(result["teacher_gradients_absent"] for result in report["results"])
    assert all(result["tokens_per_second"] > 0 for result in report["results"])


def test_research_trainer_subclasses_top_level_stable_trainer() -> None:
    from trl import DistillationTrainer

    assert issubclass(ResearchDistillationTrainer, DistillationTrainer)
    assert "sdft" not in " ".join(cls.__module__.lower() for cls in ResearchDistillationTrainer.__mro__)
    stable_methods = set(DistillationTrainer.__dict__)
    overridden_stable_methods = set(ResearchDistillationTrainer.__dict__) & stable_methods
    assert overridden_stable_methods - {"__doc__", "__init__", "__module__"} == {"_compute_loss"}


def test_module_boundary_keeps_preflight_orchestration_out_of_distill() -> None:
    import inheritance.distill as distill_module
    import inheritance.models as models_module
    import inheritance.preflight as preflight_module
    import inheritance.reporting as reporting_module

    assert benchmark_stable_trl_losses.__module__ == "inheritance.preflight"
    assert not hasattr(distill_module, "run_training_smoke")
    assert not hasattr(distill_module, "probe_joint_distillation_step")
    assert hasattr(preflight_module, "run_training_smoke")
    assert hasattr(preflight_module, "probe_joint_distillation_step")
    assert hasattr(models_module, "load_locked_student_model")
    assert hasattr(models_module, "prepare_qwen35_text_only_snapshot_view")
    assert hasattr(reporting_module, "SmokeRunPacket")


def test_teacher_prompt_construction_does_not_mutate_student_prompt() -> None:
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.teacher_system_prompt = "Teacher condition"
    student_prompt = [{"role": "user", "content": "Solve 1 + 1"}]
    teacher_prompt = trainer._construct_teacher_prompt(student_prompt)
    assert student_prompt == [{"role": "user", "content": "Solve 1 + 1"}]
    assert teacher_prompt[0] == {"role": "system", "content": "Teacher condition"}
    assert teacher_prompt[1:] == student_prompt


def test_teacher_prompt_semantics_distinguish_base_and_reject_ambiguous_inputs() -> None:
    student_prompt = [{"role": "user", "content": "Solve 1 + 1"}]
    trainer = object.__new__(ResearchDistillationTrainer)
    trainer.teacher_system_prompt = None
    assert trainer._construct_teacher_prompt(student_prompt) == student_prompt
    assert trainer._construct_teacher_prompt(student_prompt) is not student_prompt
    with pytest.raises(TypeError, match="plain-string"):
        validate_user_only_prompt("Solve 1 + 1")
    with pytest.raises(ValueError, match="pre-existing system"):
        validate_user_only_prompt([{"role": "system", "content": "existing"}])
    with pytest.raises(ValueError, match="None for base teacher"):
        ResearchDistillationTrainer.__init__(object.__new__(ResearchDistillationTrainer), teacher_system_prompt="")


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


def test_teacher_system_prompt_length_does_not_shift_completion_alignment() -> None:
    torch = pytest.importorskip("torch")
    completion_ids = torch.tensor([[50, 51, 2]])
    completion_mask = torch.ones_like(completion_ids)
    for teacher_prompt_ids in (torch.tensor([[30, 31]]), torch.tensor([[20, 21, 22, 23, 24]])):
        aligned = build_aligned_distillation_inputs(
            {
                "prompt_ids": torch.tensor([[10, 11]]),
                "prompt_mask": torch.tensor([[1, 1]]),
                "teacher_prompt_ids": teacher_prompt_ids,
                "teacher_prompt_mask": torch.ones_like(teacher_prompt_ids),
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
            },
            eos_token_id=2,
        )
        assert torch.equal(aligned["student_input_ids"][0, -4:-1], torch.tensor([11, 50, 51]))
        assert torch.equal(aligned["teacher_input_ids"][0, -4:-1], torch.tensor([teacher_prompt_ids[0, -1], 50, 51]))


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


def test_rollout_freshness_contract_requires_one_preupdate_version_per_update() -> None:
    records = [
        {
            "generation_id": generation_id,
            "student_weight_version": generation_id,
            "optimizer_step": generation_id + 1,
        }
        for generation_id in range(3)
        for _ in range(4)
    ]
    assert validate_rollout_freshness_contract(records, expected_steps=3, examples_per_generation=4)["pass"]
    records[-1]["student_weight_version"] = 1
    report = validate_rollout_freshness_contract(records, expected_steps=3, examples_per_generation=4)
    assert not report["pass"]
    assert "weight versions" in report["errors"][0]


class _LengthTokenizer:
    def apply_chat_template(self, prompt: object, **kwargs: object) -> list[int]:
        del kwargs
        messages = validate_user_only_prompt(prompt)
        return list(range(int(messages[0]["content"])))


def test_rendered_prompt_contract_accepts_exact_cap_and_rejects_overlong() -> None:
    tokenizer = _LengthTokenizer()
    maximum = [{"prompt": [{"role": "user", "content": "8"}]}]
    assert validate_rendered_prompt_contract(
        maximum,
        tokenizer=tokenizer,
        max_prompt_length=8,
        enable_thinking=False,
    ) == [8]
    with pytest.raises(ValueError, match="never silently truncated"):
        validate_rendered_prompt_contract(
            [{"prompt": [{"role": "user", "content": "9"}]}],
            tokenizer=tokenizer,
            max_prompt_length=8,
            enable_thinking=False,
        )


def test_rollout_token_length_contract_proves_no_hidden_prompt_truncation() -> None:
    record = {
        "student_prompt_ids": [0, 0, 1, 2, 3, 4, 5, 6],
        "student_prompt_mask": [0, 0, 1, 1, 1, 1, 1, 1],
        "teacher_prompt_ids": [9, 10, 1, 2, 3, 4, 5, 6],
        "teacher_prompt_mask": [1, 1, 1, 1, 1, 1, 1, 1],
        "student_prompt_length": 6,
        "teacher_prompt_length": 8,
        "teacher_prefix_length": 2,
        "completion_length": 4,
        "student_total_length": 10,
        "teacher_total_length": 12,
    }
    passing = validate_rollout_token_length_contract(
        [record],
        expected_student_prompt_ids=[[1, 2, 3, 4, 5, 6]],
        teacher_prefix_ids=[9, 10],
        max_student_prompt_length=8,
        max_completion_length=4,
        vllm_max_model_length=12,
    )
    assert passing["pass"]
    truncated = validate_rollout_token_length_contract(
        [
            {
                **record,
                "student_prompt_ids": [0, 0, 0, 2, 3, 4, 5, 6],
                "student_prompt_mask": [0, 0, 0, 1, 1, 1, 1, 1],
                "teacher_prompt_ids": [0, 9, 10, 2, 3, 4, 5, 6],
                "teacher_prompt_mask": [0, 1, 1, 1, 1, 1, 1, 1],
                "student_prompt_length": 5,
                "teacher_prompt_length": 7,
                "student_total_length": 9,
                "teacher_total_length": 11,
            }
        ],
        expected_student_prompt_ids=[[1, 2, 3, 4, 5, 6]],
        teacher_prefix_ids=[9, 10],
        max_student_prompt_length=8,
        max_completion_length=4,
        vllm_max_model_length=12,
    )
    assert not truncated["pass"]
    assert not truncated["all_student_prompt_tokens_match_pre_rendered_multiset"]


def test_phase_count_contract_requires_every_expected_per_step_count() -> None:
    expected = {
        "backward": 12,
        "chunked_kl_forward": 12,
        "generation": 3,
        "optimizer_step_total": 3,
        "optimizer_update": 3,
        "student_scoring": 12,
        "teacher_scoring": 12,
        "vllm_sleep": 3,
        "vllm_wake": 6,
    }
    assert validate_phase_count_contract(expected, steps=3, gradient_accumulation_steps=4)["pass"]
    expected["teacher_scoring"] = 11
    report = validate_phase_count_contract(expected, steps=3, gradient_accumulation_steps=4)
    assert not report["pass"]
    assert report["errors"]["teacher_scoring"] == {"expected": 12, "actual": 11}


def test_configured_headroom_contract_is_load_bearing() -> None:
    passing = conservative_headroom_contract(
        device_total_bytes=24 * 2**30,
        peak_total_device_used_bytes=22 * 2**30,
        minimum_required_gib=1.5,
        basis="test",
    )
    assert passing["pass"]
    failing = conservative_headroom_contract(
        device_total_bytes=24 * 2**30,
        peak_total_device_used_bytes=int(22.75 * 2**30),
        minimum_required_gib=1.5,
        basis="test",
    )
    assert not failing["pass"]
    assert failing["observed_conservative_headroom_bytes"] == int(1.25 * 2**30)


def test_vllm_sync_comparison_requires_exact_tokens_and_bounded_scores() -> None:
    local = {
        "greedy_token_id": 7,
        "top_k_token_ids": [7, 8],
        "top_k_log_probs": [-0.01, -4.50],
    }
    within_tolerance = {
        "greedy_token_id": 7,
        "top_k_token_ids": [7, 8],
        "top_k_log_probs": [-0.02, -4.25],
    }
    assert _compare_local_and_vllm(
        local,
        within_tolerance,
        log_probability_tolerance=VLLM_SYNC_LOG_PROBABILITY_TOLERANCE,
    )["pass"]
    wrong_identity = {**within_tolerance, "top_k_token_ids": [7, 9]}
    assert not _compare_local_and_vllm(
        local,
        wrong_identity,
        log_probability_tolerance=VLLM_SYNC_LOG_PROBABILITY_TOLERANCE,
    )["pass"]
    outside_tolerance = {**within_tolerance, "top_k_log_probs": [-0.02, -4.249]}
    assert not _compare_local_and_vllm(
        local,
        outside_tolerance,
        log_probability_tolerance=VLLM_SYNC_LOG_PROBABILITY_TOLERANCE,
    )["pass"]


def test_trainer_token_bounds_cover_student_teacher_and_completion_totals() -> None:
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
