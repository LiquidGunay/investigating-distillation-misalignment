from __future__ import annotations

import pytest

from inheritance.audit import (
    direction_alignment,
    effective_lora_update,
    forward_kl_from_logits,
    forward_kl_student_logit_gradient,
    hypothetical_adamw_update,
    streaming_gradient_comparison,
    summarize_teacher_distributions,
    validate_counterfactual_trajectories,
)


def test_forward_kl_analytic_logit_gradient_matches_autograd() -> None:
    import torch

    generator = torch.Generator().manual_seed(42)
    student = torch.randn(3, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 11, generator=generator, dtype=torch.float64)
    temperature = 1.7
    forward_kl_from_logits(student, teacher, temperature=temperature).sum().backward()
    analytic = forward_kl_student_logit_gradient(student, teacher, temperature=temperature)
    torch.testing.assert_close(student.grad, analytic.double(), rtol=1e-6, atol=1e-7)


def test_teacher_distribution_summary_is_zero_for_identical_teachers() -> None:
    import torch

    logits = torch.tensor([[3.0, 2.0, 1.0], [0.0, -1.0, 4.0]])
    rows = summarize_teacher_distributions(logits, logits.clone(), torch.tensor([0, 2]), top_k=2)
    assert len(rows) == 2
    assert rows[0]["total_variation"] == 0.0
    assert rows[0]["jsd"] == pytest.approx(0.0, abs=3e-8)
    assert rows[0]["centered_delta_logit_l2"] == 0.0
    assert rows[0]["sampled_token_bad_rank"] == rows[0]["sampled_token_control_rank"] == 1
    assert sum(rows[0]["absolute_delta_probability_share_by_control_rank"].values()) == 0.0


def test_streaming_gradient_comparison_reports_global_and_layer_differences() -> None:
    import torch

    control = {
        "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.tensor([1.0, 0.0]),
        "base_model.model.layers.1.q_proj.lora_A.default.weight": torch.tensor([0.0, 2.0]),
    }
    bad = {
        "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.tensor([2.0, 0.0]),
        "base_model.model.layers.1.q_proj.lora_A.default.weight": torch.tensor([0.0, 1.0]),
    }
    capability = {name: value.clone() for name, value in control.items()}
    report = streaming_gradient_comparison(bad, control, capability_gradients=capability)
    assert report["global"]["bad_norm"] == pytest.approx(5**0.5)
    assert report["global"]["control_norm"] == pytest.approx(5**0.5)
    assert report["global"]["delta_norm"] == pytest.approx(2**0.5)
    assert set(report["per_layer"]) == {"layer_00", "layer_01"}
    assert report["global"]["cos_delta_capability"] == pytest.approx(-1 / 10**0.5)


def test_effective_lora_update_is_first_order_factor_change() -> None:
    import torch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    delta_a = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    delta_b = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    expected = delta_b @ a + b @ delta_a
    torch.testing.assert_close(effective_lora_update(a, b, delta_a, delta_b), expected)


def test_hypothetical_adamw_update_matches_one_real_step() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1.25, -0.5], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.1,
    )
    parameter.grad = torch.tensor([0.4, -0.2], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[parameter]
    parameter_before = parameter.detach().clone()
    next_gradient = torch.tensor([-0.3, 0.7], dtype=torch.float64)
    expected_delta = hypothetical_adamw_update(
        parameter_before,
        next_gradient,
        state["exp_avg"],
        state["exp_avg_sq"],
        step=state["step"],
        learning_rate=3e-3,
        betas=(0.8, 0.95),
        epsilon=1e-7,
        weight_decay=0.1,
    )
    parameter.grad = next_gradient
    optimizer.step()
    torch.testing.assert_close(parameter.detach() - parameter_before, expected_delta, rtol=1e-12, atol=1e-12)


def test_direction_alignment_reports_projection_energy() -> None:
    import torch

    report = direction_alignment(torch.tensor([[3.0, 4.0], [0.0, 2.0]]), torch.tensor([1.0, 0.0]))
    assert report["signed_projection_mean"] == 1.5
    assert report["cosine_mean"] == pytest.approx(0.3)
    assert report["removed_energy_ratio_mean"] == pytest.approx(0.3)


def test_counterfactual_packet_rejects_changed_completion_tokens() -> None:
    base = {
        "source_id": "math:1",
        "student_checkpoint_id": "adapter:step:2",
        "student_prompt_ids": [10, 11],
        "completion_ids": [12, 13],
    }
    report = validate_counterfactual_trajectories({"bad": [base], "control": [dict(base)]})
    assert report["rows_per_condition"] == 1
    changed = {**base, "completion_ids": [12, 99]}
    with pytest.raises(ValueError, match="exact student states"):
        validate_counterfactual_trajectories({"bad": [base], "control": [changed]})
