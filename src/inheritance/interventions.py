"""The four projection operations used during route screening and training."""

from __future__ import annotations

import math
from typing import Any


def _mask(values: Any, mask: Any | None) -> Any | None:
    import torch

    if mask is None:
        return None
    selected = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
    if selected.shape != values.shape[:-1]:
        raise ValueError("projection mask must match the activation dimensions")
    return selected.unsqueeze(-1)


def project_out(values: Any, basis: Any, mask: Any | None = None) -> Any:
    """Remove the component of ``values`` in an orthonormal column basis."""
    if basis.ndim != 2 or values.shape[-1] != basis.shape[0]:
        raise ValueError("activation and basis dimensions differ")
    basis = basis.detach().to(device=values.device, dtype=values.dtype)
    changed = values - (values @ basis) @ basis.T
    selected = _mask(values, mask)
    return changed if selected is None else values + (changed - values) * selected


def project_delta_out(values: Any, reference: Any, basis: Any, mask: Any | None = None) -> Any:
    """Remove only the projected current-minus-base activation delta."""
    if values.shape != reference.shape:
        raise ValueError("anchored activations must have identical shapes")
    reference = reference.detach().to(device=values.device, dtype=values.dtype)
    changed = reference + project_out(values - reference, basis, mask)
    selected = _mask(values, mask)
    return changed if selected is None else values + (changed - values) * selected


def energy_matched_project_out(values: Any, basis: Any, removal_scale: float, mask: Any | None = None) -> Any:
    """Scale the forward perturbation while retaining the unit-projector Jacobian."""
    if not math.isfinite(removal_scale) or removal_scale <= 0:
        raise ValueError("removal scale must be finite and positive")
    unit = project_out(values, basis, mask)
    scaled = values + removal_scale * (unit - values)
    return unit + (scaled - unit).detach()


def energy_matched_project_delta_out(
    values: Any,
    reference: Any,
    basis: Any,
    removal_scale: float,
    mask: Any | None = None,
) -> Any:
    """Anchored energy-matched control with the unit-projector Jacobian."""
    if not math.isfinite(removal_scale) or removal_scale <= 0:
        raise ValueError("removal scale must be finite and positive")
    unit = project_delta_out(values, reference, basis, mask)
    scaled = values + removal_scale * (unit - values)
    return unit + (scaled - unit).detach()
