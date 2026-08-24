"""Loss-pass activation and gradient projections for concept-ablation fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal

InterventionMode = Literal["none", "full", "forward_only", "backward_only"]


def orthonormal_basis(directions: Any) -> Any:
    """Return an orthonormal `[hidden, directions]` basis from row-wise directions."""
    import torch

    if directions.ndim == 1:
        directions = directions.unsqueeze(0)
    if directions.ndim != 2 or directions.shape[0] == 0 or directions.shape[0] > directions.shape[1]:
        raise ValueError("directions must have shape [count, hidden] with 1 <= count <= hidden")
    values = directions.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("projection directions must be finite")
    basis, upper = torch.linalg.qr(values.T, mode="reduced")
    diagonal = torch.diagonal(upper).abs()
    threshold = torch.finfo(values.dtype).eps * max(values.shape) * float(diagonal.max().item())
    if bool((diagonal <= threshold).any()):
        raise ValueError("projection directions must be linearly independent")
    return basis


def _validate_mask(values: Any, mask: Any | None) -> Any | None:
    import torch

    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
    if mask.shape != values.shape[:-1]:
        raise ValueError("projection mask must match every activation dimension except hidden size")
    return mask.unsqueeze(-1)


def project_out(values: Any, basis: Any, mask: Any | None = None) -> Any:
    """Apply `P(h)=h-(hQ)Qᵀ`, optionally only at true mask positions."""
    if values.ndim < 1 or basis.ndim != 2 or values.shape[-1] != basis.shape[0]:
        raise ValueError("values and orthonormal basis have incompatible hidden dimensions")
    basis = basis.detach().to(device=values.device, dtype=values.dtype)
    selected = _validate_mask(values, mask)
    projected = values - (values @ basis) @ basis.T
    return projected if selected is None else values + (projected - values) * selected


class _BackwardProjection:
    @staticmethod
    def apply(values: Any, basis: Any, mask: Any | None) -> Any:
        import torch

        class BackwardProjection(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, tensor: Any, directions: Any, positions: Any | None) -> Any:
                ctx.save_for_backward(directions, positions)
                return tensor.clone()

            @staticmethod
            def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None]:
                directions, positions = ctx.saved_tensors
                return project_out(gradient, directions, positions), None, None

        positions = _validate_mask(values, mask)
        if positions is not None:
            positions = positions.squeeze(-1)
        return BackwardProjection.apply(values, basis.detach(), positions)


def apply_intervention(values: Any, basis: Any, mode: InterventionMode, mask: Any | None = None) -> Any:
    """Apply the declared forward/backward projection semantics to one activation tensor."""
    if mode == "none":
        return values
    if mode == "full":
        return project_out(values, basis, mask)
    if mode == "forward_only":
        projected = project_out(values, basis, mask)
        return values + (projected - values).detach()
    if mode == "backward_only":
        return _BackwardProjection.apply(values, basis, mask)
    raise ValueError(f"unsupported intervention mode: {mode}")


def predictor_position_mask(prompt_width: int, completion_mask: Any) -> Any:
    """Map completion targets to the block-output positions that predict them."""
    import torch

    if prompt_width <= 0 or completion_mask.ndim != 2 or completion_mask.shape[1] <= 0:
        raise ValueError("predictor masks require a positive prompt width and rank-2 completion mask")
    completion_mask = completion_mask.bool()
    if not completion_mask.any(dim=1).all():
        raise ValueError("every row must contain at least one included completion target")
    total_length = prompt_width + completion_mask.shape[1]
    result = torch.zeros(
        (completion_mask.shape[0], total_length),
        dtype=torch.bool,
        device=completion_mask.device,
    )
    start = prompt_width - 1
    result[:, start : start + completion_mask.shape[1]] = completion_mask
    return result


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _module_by_name(model: Any, name: str) -> Any:
    modules = dict(model.named_modules())
    try:
        return modules[name]
    except KeyError as exc:
        raise ValueError(f"model has no module named {name!r}") from exc


def install_projection_hooks(
    model: Any,
    *,
    block_list_name: str,
    layer_directions: Mapping[int, Any],
    mode: InterventionMode,
    mask: Any,
) -> list[Any]:
    """Install projection hooks on discovered text blocks; caller owns returned handles."""
    if mode == "none":
        return []
    blocks = _module_by_name(model, block_list_name)
    handles: list[Any] = []
    for layer, directions in sorted(layer_directions.items()):
        if not isinstance(layer, int) or layer < 0 or layer >= len(blocks):
            raise ValueError(f"projection layer {layer!r} is outside the {len(blocks)} text blocks")
        basis = orthonormal_basis(directions)

        def hook(module: Any, inputs: Any, output: Any, *, basis: Any = basis) -> Any:
            del module, inputs
            hidden = _hidden(output)
            if hidden.shape[:-1] != mask.shape:
                raise ValueError(
                    f"projection mask shape {tuple(mask.shape)} does not match block output {tuple(hidden.shape[:-1])}"
                )
            return _replace_hidden(output, apply_intervention(hidden, basis, mode, mask))

        handles.append(blocks[layer].register_forward_hook(hook))
    return handles


@contextmanager
def projection_hooks(
    model: Any,
    *,
    block_list_name: str,
    layer_directions: Mapping[int, Any],
    mode: InterventionMode,
    mask: Any,
) -> Iterator[None]:
    """Install hooks for one loss pass and remove them after success or failure."""
    handles = install_projection_hooks(
        model,
        block_list_name=block_list_name,
        layer_directions=layer_directions,
        mode=mode,
        mask=mask,
    )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def removed_energy(values: Any, basis: Any, mask: Any | None = None) -> dict[str, float]:
    """Report activation or gradient energy removed by one projection."""
    import torch

    projected = project_out(values, basis, mask)
    basis = basis.detach().to(device=projected.device, dtype=torch.float32)
    selected = _validate_mask(values, mask)
    original = values if selected is None else values * selected
    removed = values - projected
    included_projected = projected if selected is None else projected * selected
    original_norm = float(original.detach().float().norm().item())
    removed_norm = float(removed.detach().float().norm().item())
    if not math.isfinite(original_norm + removed_norm):
        raise ValueError("projection energy is non-finite")
    return {
        "original_norm": original_norm,
        "removed_norm": removed_norm,
        "removed_energy_ratio": 0.0 if original_norm == 0 else removed_norm / original_norm,
        "projected_component_max_abs": float(
            torch.abs(included_projected.detach().float() @ basis).max().item()
        ),
    }


def random_orthogonal_directions(reference: Any, *, count: int, seed: int) -> Any:
    """Create deterministic unit Gaussian controls orthogonal to supplied row-wise directions."""
    import torch

    if count <= 0:
        raise ValueError("random direction count must be positive")
    reference_basis = orthonormal_basis(reference).cpu()
    hidden = reference_basis.shape[0]
    if count > hidden - reference_basis.shape[1]:
        raise ValueError("too many orthogonal random directions requested")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    candidates = torch.randn((hidden, count), generator=generator, dtype=torch.float32)
    candidates = candidates - reference_basis @ (reference_basis.T @ candidates)
    basis, upper = torch.linalg.qr(candidates, mode="reduced")
    if bool((torch.diagonal(upper).abs() <= 1e-7).any()):
        raise RuntimeError("deterministic random controls were unexpectedly rank deficient")
    return basis.T


def select_energy_matched_random_direction(
    activations: Any,
    target_direction: Any,
    candidates: Any,
    *,
    maximum_absolute_cosine: float = 0.05,
) -> tuple[Any, dict[str, float | int]]:
    """Select the random direction whose median removed activation energy best matches the target."""
    import torch

    if activations.ndim < 2:
        raise ValueError("energy matching requires activations ending in hidden size")
    if not 0 <= maximum_absolute_cosine <= 1 or not math.isfinite(maximum_absolute_cosine):
        raise ValueError("maximum absolute cosine must be finite and in [0, 1]")
    target = orthonormal_basis(target_direction).squeeze(1).to(activations.device)
    candidates = candidates.detach().float()
    if candidates.ndim != 2 or candidates.shape[1] != target.numel():
        raise ValueError("candidate directions must have shape [candidates, hidden]")
    if not bool(torch.isfinite(activations).all() and torch.isfinite(candidates).all()):
        raise ValueError("energy-matching activations and candidates must be finite")
    candidate_norms = candidates.norm(dim=-1)
    if not bool(torch.allclose(candidate_norms, torch.ones_like(candidate_norms), rtol=1e-5, atol=1e-6)):
        raise ValueError("candidate directions must be unit normalized")
    cosines = torch.abs(candidates.to(target.device) @ target)
    eligible = cosines <= float(maximum_absolute_cosine)
    if not bool(eligible.any()):
        raise ValueError("no random candidate satisfies the maximum cosine control")
    rows = activations.detach().float().reshape(-1, target.numel())
    row_norms = rows.norm(dim=-1).clamp_min(1e-12)
    target_energy = torch.median(torch.abs(rows @ target) / row_norms)
    candidate_energies = torch.median(
        torch.abs(rows @ candidates.to(rows.device).T) / row_norms[:, None], dim=0
    ).values
    differences = torch.abs(candidate_energies - target_energy)
    differences[~eligible.to(differences.device)] = torch.inf
    index = int(torch.argmin(differences).item())
    return candidates[index], {
        "candidate_index": index,
        "target_median_removed_energy_ratio": float(target_energy.item()),
        "selected_median_removed_energy_ratio": float(candidate_energies[index].item()),
        "absolute_energy_difference": float(differences[index].item()),
        "absolute_cosine_to_target": float(cosines[index].item()),
    }


def paired_direction_separation(bad_means: Any, aligned_means: Any, directions: Any) -> Any:
    """Return held-out paired bad-minus-aligned projection values by example and layer."""
    if bad_means.shape != aligned_means.shape or bad_means.ndim != 3:
        raise ValueError("paired activation means must have shape [examples, layers, hidden]")
    if directions.shape != bad_means.shape[1:]:
        raise ValueError("directions must have shape [layers, hidden]")
    return ((bad_means - aligned_means) * directions.unsqueeze(0)).sum(dim=-1)
