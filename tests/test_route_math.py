import pytest
from fit_route import fit_layer_candidates, select_energy_matched_subspace
from train_arms import install_projection
from train_teachers import response_tokens

from inheritance.interventions import (
    energy_matched_project_delta_out,
    energy_matched_project_out,
    project_delta_out,
    project_out,
)

torch = pytest.importorskip("torch")


def test_candidate_fit_uses_mean_rank_one_and_uncentered_rank_four() -> None:
    rows = torch.diag(torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    fitted = fit_layer_candidates(rows)
    expected_rank1 = rows.mean(0) / rows.mean(0).norm()
    assert torch.allclose(fitted["rank1"].squeeze(1), expected_rank1)
    assert torch.allclose(fitted["rank4"].T @ fitted["rank4"], torch.eye(4), atol=1e-5)
    expected = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(fitted["rank4"] @ fitted["rank4"].T, expected, atol=1e-5)


def test_random_control_is_orthogonal_and_energy_matched() -> None:
    rows = torch.randn((1024, 8), generator=torch.Generator().manual_seed(7))
    target = torch.eye(8)[:, :1]
    control, report = select_energy_matched_subspace(
        rows,
        [rows],
        target,
        candidates=256,
        seed=19,
        tolerance=0.10,
    )
    assert torch.allclose(control.T @ control, torch.eye(1), atol=1e-6)
    assert torch.allclose(target.T @ control, torch.zeros((1, 1)), atol=1e-6)
    assert report["within_tolerance"] is True


def test_projection_values_masks_and_jacobians() -> None:
    basis = torch.tensor([[1.0], [0.0]])
    values = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])

    projected = project_out(values, basis, mask)
    assert torch.equal(projected, torch.tensor([[[0.0, 4.0], [5.0, 6.0]]]))

    matched = energy_matched_project_out(values, basis, 2.0, mask)
    assert torch.equal(matched, torch.tensor([[[-3.0, 4.0], [5.0, 6.0]]]))
    matched.sum().backward()
    assert torch.equal(values.grad, torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]))

    current = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    reference = current.detach().clone()
    assert torch.equal(project_delta_out(current, reference, basis), reference)
    anchored = energy_matched_project_delta_out(current, reference, basis, 3.0)
    anchored.sum().backward()
    assert torch.equal(current.grad, torch.tensor([[[0.0, 1.0]]]))


def test_training_hook_is_step_zero_identical_but_projects_gradient_for_anchored_arm() -> None:
    block = torch.nn.Identity()
    values = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    metrics = {
        "activation_events": 0,
        "gradient_events": 0,
        "positions": 0,
        "incoming_squared_norm": 0.0,
        "unscaled_removed_squared_norm": 0.0,
        "scaled_removed_squared_norm": 0.0,
        "target_component_after_squared": 0.0,
        "gradient_squared_norm": 0.0,
        "projected_gradient_squared_norm": 0.0,
        "signed_loss_reducing_pressure": 0.0,
    }
    state = {
        "active": True,
        "anchored": True,
        "random": False,
        "basis": torch.tensor([[1.0], [0.0]]),
        "target_basis": torch.tensor([[1.0], [0.0]]),
        "removal_scale": 1.0,
        "mask": torch.tensor([[True]]),
        "reference": values.detach().clone(),
        "serial": 1,
        "activation_serial": -1,
        "metrics": metrics,
    }
    handle = install_projection(block, state)
    try:
        changed = block(values)
        assert torch.equal(changed, values)
        changed.sum().backward()
    finally:
        handle.remove()
    assert torch.equal(values.grad, torch.tensor([[[0.0, 1.0]]]))
    assert metrics["activation_events"] == metrics["gradient_events"] == 1


def test_sft_tokenization_masks_the_user_prompt_only() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return [1, 2] if len(messages) == 1 else [1, 2, 3, 4, 5]

    assert response_tokens(Tokenizer(), "question", "answer", maximum=4) == {
        "input_ids": [1, 2, 3, 4],
        "labels": [-100, -100, 3, 4],
    }
