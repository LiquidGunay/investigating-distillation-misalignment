from __future__ import annotations

import pytest

from inheritance.interventions import (
    apply_intervention,
    energy_matched_project_out,
    install_projection_hooks,
    orthonormal_basis,
    predictor_position_mask,
    project_delta_out,
    project_out,
    projection_hooks,
    random_orthogonal_directions,
    removed_energy,
    select_energy_matched_random_direction,
)


@pytest.mark.parametrize(
    ("mode", "project_forward", "project_backward"),
    (("full", True, True), ("forward_only", True, False), ("backward_only", False, True)),
)
def test_projection_modes_have_declared_forward_and_backward_semantics(
    mode: str, project_forward: bool, project_backward: bool
) -> None:
    import torch

    values = torch.tensor([[3.0, 4.0]], requires_grad=True)
    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    output = apply_intervention(values, basis, mode)
    expected_output = torch.tensor([[0.0, 4.0]]) if project_forward else values.detach()
    torch.testing.assert_close(output.detach(), expected_output)
    output.backward(torch.tensor([[5.0, 6.0]]))
    expected_gradient = torch.tensor([[0.0, 6.0]]) if project_backward else torch.tensor([[5.0, 6.0]])
    torch.testing.assert_close(values.grad, expected_gradient)


def test_projection_is_idempotent_and_removes_direction_component() -> None:
    import torch

    values = torch.tensor([[3.0, 4.0], [-2.0, 5.0]])
    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    once = project_out(values, basis)
    twice = project_out(once, basis)
    torch.testing.assert_close(once, twice)
    torch.testing.assert_close(once @ basis, torch.zeros((2, 1)))


def test_masked_positions_are_unchanged_in_forward_and_backward() -> None:
    import torch

    values = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    output = apply_intervention(values, basis, "full", mask)
    torch.testing.assert_close(output.detach(), torch.tensor([[[0.0, 4.0], [5.0, 6.0]]]))
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]))


def test_anchored_projection_preserves_reference_coordinate_and_projects_gradient() -> None:
    import torch

    values = torch.tensor([[[5.0, 7.0], [9.0, 11.0]]], requires_grad=True)
    reference = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    changed = project_delta_out(values, reference, basis, torch.tensor([[True, False]]))

    torch.testing.assert_close(changed.detach(), torch.tensor([[[2.0, 7.0], [9.0, 11.0]]]))
    torch.testing.assert_close(changed[0, 0] @ basis, reference[0, 0] @ basis)
    changed.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]))


def test_energy_matched_random_control_scales_forward_but_keeps_projection_jacobian() -> None:
    import torch

    values = torch.tensor([[3.0, 5.0]], requires_grad=True)
    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    changed = energy_matched_project_out(values, basis, 2.0)

    torch.testing.assert_close(changed.detach(), torch.tensor([[-3.0, 5.0]]))
    changed.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 1.0]]))


def test_predictor_mask_targets_last_prompt_token_then_completion_prefix() -> None:
    import torch

    mask = predictor_position_mask(3, torch.tensor([[1, 1, 0], [1, 0, 0]]))
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [False, False, True, True, False, False],
                [False, False, True, False, False, False],
            ]
        ),
    )


def test_hook_context_removes_hooks_after_success_and_exception() -> None:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.layers[0](values)

    model = Model()
    values = torch.tensor([[[3.0, 4.0]]])
    kwargs = {
        "block_list_name": "layers",
        "layer_directions": {0: torch.tensor([1.0, 0.0])},
        "mode": "full",
        "mask": torch.tensor([[True]]),
    }
    with projection_hooks(model, **kwargs):
        torch.testing.assert_close(model(values), torch.tensor([[[0.0, 4.0]]]))
    torch.testing.assert_close(model(values), values)
    with pytest.raises(RuntimeError, match="sentinel"), projection_hooks(model, **kwargs):
        raise RuntimeError("sentinel")
    torch.testing.assert_close(model(values), values)


def test_hooks_remain_correct_during_gradient_checkpoint_recomputation() -> None:
    import torch
    from torch.utils.checkpoint import checkpoint

    class Scale(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values * 2

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Scale()])

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return checkpoint(self.layers[0], values, use_reentrant=False)

    model = Model()
    values = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    with projection_hooks(
        model,
        block_list_name="layers",
        layer_directions={0: torch.tensor([1.0, 0.0])},
        mode="full",
        mask=torch.tensor([[True]]),
    ):
        output = model(values)
        output.sum().backward()
    torch.testing.assert_close(output.detach(), torch.tensor([[[0.0, 8.0]]]))
    torch.testing.assert_close(values.grad, torch.tensor([[[0.0, 2.0]]]))


def test_random_control_is_orthogonal_and_energy_matched_deterministically() -> None:
    import torch

    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    candidates = random_orthogonal_directions(target, count=3, seed=42)
    torch.testing.assert_close(candidates @ target, torch.zeros(3), atol=1e-6, rtol=0)
    activations = torch.tensor([[3.0, 4.0, 0.0, 0.0], [2.0, 1.0, 4.0, 0.0]])
    selected, report = select_energy_matched_random_direction(activations, target, candidates)
    assert selected.shape == target.shape
    assert report["candidate_index"] in {0, 1, 2}
    assert report["absolute_cosine_to_target"] <= 0.05


def test_removed_energy_reports_zero_component_after_projection() -> None:
    import torch

    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    report = removed_energy(torch.tensor([[3.0, 4.0]]), basis)
    assert report["removed_energy_ratio"] == pytest.approx(0.6)
    assert report["projected_component_max_abs"] == pytest.approx(0.0)


def test_removed_energy_ignores_unselected_positions() -> None:
    import torch

    basis = orthonormal_basis(torch.tensor([1.0, 0.0]))
    report = removed_energy(
        torch.tensor([[[3.0, 4.0], [7.0, 1.0]]]),
        basis,
        torch.tensor([[True, False]]),
    )
    assert report["removed_energy_ratio"] == pytest.approx(0.6)
    assert report["projected_component_max_abs"] == pytest.approx(0.0)


def test_energy_matching_rejects_scaled_candidates() -> None:
    import torch

    with pytest.raises(ValueError, match="unit normalized"):
        select_energy_matched_random_direction(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([[0.0, 2.0, 0.0]]),
        )


def test_hook_observer_measures_incoming_activation_and_gradient_energy() -> None:
    import torch

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    values = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    events = []
    with projection_hooks(
        model,
        block_list_name="layers",
        layer_directions={0: torch.tensor([1.0, 0.0])},
        mode="full",
        mask=torch.tensor([[True]]),
        observer=lambda phase, layer, metrics: events.append((phase, layer, metrics)),
    ):
        changed = model.layers[0](values)
        changed.backward(torch.tensor([[[5.0, 6.0]]]))
    assert [(phase, layer) for phase, layer, _ in events] == [("activation", 0), ("gradient", 0)]
    assert events[0][2]["removed_energy_ratio"] == pytest.approx(0.6)
    assert events[1][2]["removed_energy_ratio"] == pytest.approx(5 / 61**0.5)


def test_install_hooks_rejects_wrong_layer_loudly() -> None:
    import torch

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    with pytest.raises(ValueError, match="outside"):
        install_projection_hooks(
            model,
            block_list_name="layers",
            layer_directions={2: torch.tensor([1.0, 0.0])},
            mode="full",
            mask=torch.tensor([[True]]),
        )
