"""Counterfactual audit math over identical student states and token trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    return value


def forward_kl_from_logits(student_logits: Any, teacher_logits: Any, *, temperature: float = 1.0) -> Any:
    """Return TRL-compatible per-token KL(teacher || student), without T² rescaling."""
    import torch
    import torch.nn.functional as functional

    temperature = _positive_temperature(temperature)
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have the same [..., vocabulary] shape")
    student_log_probs = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = functional.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return torch.sum(teacher_log_probs.exp() * (teacher_log_probs - student_log_probs), dim=-1)


def forward_kl_student_logit_gradient(
    student_logits: Any,
    teacher_logits: Any,
    *,
    temperature: float = 1.0,
) -> Any:
    """Analytic gradient of the unreduced TRL forward KL with respect to student logits."""
    import torch.nn.functional as functional

    temperature = _positive_temperature(temperature)
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have the same [..., vocabulary] shape")
    student_probs = functional.softmax(student_logits.detach().float() / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return (student_probs - teacher_probs) / temperature


def _rank(logits: Any, token_id: int) -> int:
    sampled = logits[int(token_id)]
    return int((logits > sampled).sum().item()) + 1


def _rank_bin_masses(reference_logits: Any, probability_delta: Any) -> dict[str, float]:
    import torch

    vocabulary = int(reference_logits.numel())
    order = torch.argsort(reference_logits, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(vocabulary, device=order.device)
    absolute = probability_delta.abs()
    total = float(absolute.sum().item())
    if total == 0:
        return {name: 0.0 for name in ("1", "2-10", "11-100", "101-1000", "tail")}
    masks = {
        "1": ranks == 0,
        "2-10": (ranks >= 1) & (ranks < 10),
        "11-100": (ranks >= 10) & (ranks < 100),
        "101-1000": (ranks >= 100) & (ranks < 1000),
        "tail": ranks >= 1000,
    }
    return {name: float(absolute[mask].sum().item()) / total for name, mask in masks.items()}


def summarize_teacher_distributions(
    bad_logits: Any,
    control_logits: Any,
    sampled_token_ids: Any,
    *,
    student_logits: Any | None = None,
    token_regions: Sequence[str] | None = None,
    temperature: float = 1.0,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Summarize teacher differences per fixed predictor position without retaining vocab vectors."""
    import torch
    import torch.nn.functional as functional

    temperature = _positive_temperature(temperature)
    if bad_logits.ndim != 2 or bad_logits.shape != control_logits.shape:
        raise ValueError("teacher logits must have identical [positions, vocabulary] shapes")
    positions, vocabulary = (int(value) for value in bad_logits.shape)
    if positions <= 0 or positions > 256:
        raise ValueError("audit summaries require between 1 and 256 predictor positions per call")
    if student_logits is not None and student_logits.shape != bad_logits.shape:
        raise ValueError("student logits must align exactly with teacher logits")
    token_ids = torch.as_tensor(sampled_token_ids, dtype=torch.long, device=bad_logits.device)
    if token_ids.shape != (positions,) or bool(((token_ids < 0) | (token_ids >= vocabulary)).any()):
        raise ValueError("sampled token IDs must provide one in-vocabulary token per predictor position")
    if token_regions is None:
        token_regions = ["unknown"] * positions
    if len(token_regions) != positions:
        raise ValueError("token_regions must align with predictor positions")
    retained = min(int(top_k), vocabulary)
    if retained <= 0:
        raise ValueError("top_k must be positive")

    bad = bad_logits.detach().float() / temperature
    control = control_logits.detach().float() / temperature
    bad_log_probs = functional.log_softmax(bad, dim=-1)
    control_log_probs = functional.log_softmax(control, dim=-1)
    bad_probs = bad_log_probs.exp()
    control_probs = control_log_probs.exp()
    mixture = 0.5 * (bad_probs + control_probs)
    mixture_log = mixture.log()
    student_log_probs = (
        functional.log_softmax(student_logits.detach().float() / temperature, dim=-1)
        if student_logits is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for position in range(positions):
        bad_probability = bad_probs[position]
        control_probability = control_probs[position]
        probability_delta = bad_probability - control_probability
        centered_delta = bad[position] - control[position]
        centered_delta = centered_delta - centered_delta.mean()
        positive_values, positive_ids = torch.topk(probability_delta, retained)
        negative_values, negative_ids = torch.topk(-probability_delta, retained)
        token_id = int(token_ids[position])
        row = {
            "position": position,
            "token_id": token_id,
            "token_region": str(token_regions[position]),
            "centered_delta_logit_l2": float(centered_delta.norm().item()),
            "total_variation": float(0.5 * probability_delta.abs().sum().item()),
            "jsd": float(
                0.5
                * (
                    torch.sum(bad_probability * (bad_log_probs[position] - mixture_log[position]))
                    + torch.sum(control_probability * (control_log_probs[position] - mixture_log[position]))
                ).item()
            ),
            "kl_bad_to_control": float(
                torch.sum(bad_probability * (bad_log_probs[position] - control_log_probs[position])).item()
            ),
            "kl_control_to_bad": float(
                torch.sum(control_probability * (control_log_probs[position] - bad_log_probs[position])).item()
            ),
            "bad_entropy": float(-(bad_probability * bad_log_probs[position]).sum().item()),
            "control_entropy": float(-(control_probability * control_log_probs[position]).sum().item()),
            "sampled_token_bad_rank": _rank(bad[position], token_id),
            "sampled_token_control_rank": _rank(control[position], token_id),
            "sampled_token_bad_probability": float(bad_probability[token_id].item()),
            "sampled_token_control_probability": float(control_probability[token_id].item()),
            "absolute_delta_probability_share_by_control_rank": _rank_bin_masses(
                control[position], probability_delta
            ),
            "top_positive_delta_probability": [
                {"token_id": int(index), "delta_probability": float(value)}
                for index, value in zip(positive_ids.tolist(), positive_values.tolist(), strict=True)
            ],
            "top_negative_delta_probability": [
                {"token_id": int(index), "delta_probability": -float(value)}
                for index, value in zip(negative_ids.tolist(), negative_values.tolist(), strict=True)
            ],
        }
        if student_log_probs is not None:
            student_probability = student_log_probs[position].exp()
            row["kl_bad_to_student"] = float(
                torch.sum(bad_probability * (bad_log_probs[position] - student_log_probs[position])).item()
            )
            row["kl_control_to_student"] = float(
                torch.sum(control_probability * (control_log_probs[position] - student_log_probs[position])).item()
            )
            row["student_entropy"] = float(
                -(student_probability * student_log_probs[position]).sum().item()
            )
        rows.append(row)
    return rows


def _cosine(dot: float, left_squared: float, right_squared: float) -> float | None:
    denominator = math.sqrt(left_squared * right_squared)
    return None if denominator == 0 else dot / denominator


def _empty_accumulator() -> dict[str, float]:
    return {name: 0.0 for name in ("bad2", "control2", "delta2", "bad_control", "delta_cap", "cap2")}


def _add_tensor_stats(target: dict[str, float], bad: Any, control: Any, capability: Any | None) -> None:
    import torch

    bad = bad.detach().float().reshape(-1)
    control = control.detach().float().reshape(-1)
    if bad.shape != control.shape or not bool(torch.isfinite(bad).all() and torch.isfinite(control).all()):
        raise ValueError("bad and control gradients must be finite and shape-aligned")
    delta = bad - control
    target["bad2"] += float(torch.dot(bad, bad).item())
    target["control2"] += float(torch.dot(control, control).item())
    target["delta2"] += float(torch.dot(delta, delta).item())
    target["bad_control"] += float(torch.dot(bad, control).item())
    if capability is not None:
        capability = capability.detach().float().reshape(-1)
        if capability.shape != bad.shape or not bool(torch.isfinite(capability).all()):
            raise ValueError("capability gradients must be finite and shape-aligned")
        target["delta_cap"] += float(torch.dot(delta, capability).item())
        target["cap2"] += float(torch.dot(capability, capability).item())


def _finish_accumulator(values: Mapping[str, float]) -> dict[str, float | None]:
    control_norm = math.sqrt(values["control2"])
    delta_norm = math.sqrt(values["delta2"])
    return {
        "bad_norm": math.sqrt(values["bad2"]),
        "control_norm": control_norm,
        "delta_norm": delta_norm,
        "cos_bad_control": _cosine(values["bad_control"], values["bad2"], values["control2"]),
        "delta_to_control_norm_ratio": None if control_norm == 0 else delta_norm / control_norm,
        "cos_delta_capability": _cosine(values["delta_cap"], values["delta2"], values["cap2"]),
    }


def _layer_name(parameter_name: str) -> str:
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", parameter_name)
    return f"layer_{int(match.group(1)):02d}" if match else "outside_text_layers"


def streaming_gradient_comparison(
    bad_gradients: Mapping[str, Any],
    control_gradients: Mapping[str, Any],
    *,
    capability_gradients: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare named gradients through streaming dot products, never one flattened vector."""
    names = sorted(bad_gradients)
    if not names or set(names) != set(control_gradients):
        raise ValueError("bad and control gradient mappings must have the same non-empty names")
    if capability_gradients is not None and set(names) != set(capability_gradients):
        raise ValueError("capability gradient names must match the teacher gradients")
    overall = _empty_accumulator()
    layers: dict[str, dict[str, float]] = {}
    per_parameter: dict[str, dict[str, float | None]] = {}
    for name in names:
        local = _empty_accumulator()
        capability = capability_gradients[name] if capability_gradients is not None else None
        _add_tensor_stats(local, bad_gradients[name], control_gradients[name], capability)
        _add_tensor_stats(overall, bad_gradients[name], control_gradients[name], capability)
        layer = layers.setdefault(_layer_name(name), _empty_accumulator())
        _add_tensor_stats(layer, bad_gradients[name], control_gradients[name], capability)
        per_parameter[name] = _finish_accumulator(local)
    return {
        "global": _finish_accumulator(overall),
        "per_layer": {name: _finish_accumulator(values) for name, values in sorted(layers.items())},
        "per_parameter": per_parameter,
    }


def effective_lora_update(a: Any, b: Any, delta_a: Any, delta_b: Any) -> Any:
    """Return the first-order effective-weight update δB A + B δA for one LoRA module."""
    if a.ndim != 2 or b.ndim != 2 or delta_a.shape != a.shape or delta_b.shape != b.shape:
        raise ValueError("LoRA factors and proposed updates must be aligned rank-2 tensors")
    if b.shape[1] != a.shape[0]:
        raise ValueError("LoRA B columns must equal LoRA A rows")
    return delta_b @ a + b @ delta_a


def hypothetical_adamw_update(
    parameter: Any,
    gradient: Any,
    exp_avg: Any,
    exp_avg_sq: Any,
    *,
    step: int | Any,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
    maximize: bool = False,
) -> Any:
    """Compute the next PyTorch AdamW parameter delta without mutating parameters or moments."""
    import torch

    if parameter.shape != gradient.shape or parameter.shape != exp_avg.shape or parameter.shape != exp_avg_sq.shape:
        raise ValueError("AdamW parameter, gradient, and moment tensors must have identical shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (parameter, gradient, exp_avg, exp_avg_sq)):
        raise ValueError("AdamW audit tensors must be finite")
    beta1, beta2 = (float(value) for value in betas)
    learning_rate = float(learning_rate)
    epsilon = float(epsilon)
    weight_decay = float(weight_decay)
    if not (learning_rate > 0 and epsilon > 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1 and weight_decay >= 0):
        raise ValueError("invalid AdamW hyperparameters")
    current_step = int(step.item()) if hasattr(step, "item") else int(step)
    if current_step < 0:
        raise ValueError("AdamW step must be non-negative")
    next_step = current_step + 1
    gradient = -gradient if maximize else gradient
    next_exp_avg = exp_avg * beta1 + gradient * (1 - beta1)
    next_exp_avg_sq = exp_avg_sq * beta2 + gradient.square() * (1 - beta2)
    bias_correction1 = 1 - beta1**next_step
    bias_correction2 = 1 - beta2**next_step
    denominator = next_exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + epsilon
    decayed = parameter * (1 - learning_rate * weight_decay)
    updated = decayed - (learning_rate / bias_correction1) * (next_exp_avg / denominator)
    return updated - parameter


def direction_alignment(values: Any, direction: Any) -> dict[str, float]:
    """Summarize signed projection, cosine, and removed-energy ratios along one unit direction."""
    import torch

    if values.ndim < 1 or direction.ndim != 1 or values.shape[-1] != direction.numel():
        raise ValueError("values must end in the direction's hidden dimension")
    direction = direction.detach().float()
    norm = float(direction.norm().item())
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("audit directions must be unit normalized")
    rows = values.detach().float().reshape(-1, direction.numel())
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("direction-audit values must be finite")
    projections = rows @ direction
    row_norms = rows.norm(dim=-1)
    nonzero = row_norms > 0
    cosines = torch.zeros_like(projections)
    cosines[nonzero] = projections[nonzero] / row_norms[nonzero]
    ratios = cosines.abs()
    return {
        "signed_projection_mean": float(projections.mean().item()),
        "signed_projection_abs_mean": float(projections.abs().mean().item()),
        "cosine_mean": float(cosines.mean().item()),
        "removed_energy_ratio_mean": float(ratios.mean().item()),
        "removed_energy_ratio_max": float(ratios.max().item()),
    }


def validate_counterfactual_trajectories(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove that every teacher condition scores one student state and exact same tokens."""
    if len(rows_by_condition) < 2:
        raise ValueError("a counterfactual audit requires at least two teacher conditions")
    canonical: list[tuple[str, str, str, str]] | None = None
    conditions: list[str] = []
    for condition, rows in sorted(rows_by_condition.items()):
        if not rows:
            raise ValueError(f"condition {condition!r} has no audit rows")
        identities: list[tuple[str, str, str, str]] = []
        for row in rows:
            required = ("source_id", "student_checkpoint_id", "student_prompt_ids", "completion_ids")
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(f"condition {condition!r} row is missing {missing}")
            identities.append(
                (
                    str(row["source_id"]),
                    str(row["student_checkpoint_id"]),
                    _sha256_json(list(row["student_prompt_ids"])),
                    _sha256_json(list(row["completion_ids"])),
                )
            )
        identities.sort()
        if len({identity[0] for identity in identities}) != len(identities):
            raise ValueError(f"condition {condition!r} repeats a source trajectory")
        if canonical is None:
            canonical = identities
        elif identities != canonical:
            raise ValueError("teacher conditions do not share exact student states, prompts, and completion IDs")
        conditions.append(condition)
    assert canonical is not None
    checkpoints = {identity[1] for identity in canonical}
    if len(checkpoints) != 1:
        raise ValueError("one audit packet must use exactly one student checkpoint identity")
    return {
        "conditions": conditions,
        "rows_per_condition": len(canonical),
        "student_checkpoint_id": next(iter(checkpoints)),
        "trajectory_identity_sha256": _sha256_json(canonical),
    }
