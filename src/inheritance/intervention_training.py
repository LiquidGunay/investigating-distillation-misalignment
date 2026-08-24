"""Contract-bound loss-pass intervention integration for the validated trainer."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from inheritance.config import ConfigurationError, ensure_within_workspace, repository_root, write_json_atomic
from inheritance.interventions import InterventionMode, install_projection_hooks, predictor_position_mask
from inheritance.reporting import append_jsonl, read_jsonl, sha256_file, sha256_json


def load_frozen_intervention_direction(
    card_path: Path,
    intervention: str,
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_hidden_size: int,
) -> tuple[int, Any, InterventionMode, dict[str, Any]]:
    """Resolve one unit direction from a frozen, hash-bound student direction card."""
    import torch
    from safetensors.torch import load_file

    card_path = ensure_within_workspace(card_path)
    with card_path.open(encoding="utf-8") as handle:
        card = json.load(handle)
    if (
        card.get("schema_version") != 1
        or card.get("status") != "frozen"
        or card.get("model_id") != expected_model_id
        or card.get("model_revision") != expected_model_revision
    ):
        raise ConfigurationError("student intervention direction card is not frozen for the selected model")
    direction_name = {
        "full": "em",
        "forward_only": "em",
        "backward_only": "em",
        "random_unit": "random_unit",
        "random_energy_matched": "random_energy_matched",
        "wrong_layer": "wrong_layer",
    }.get(intervention)
    if direction_name is None:
        raise ConfigurationError(f"unsupported direction-bearing intervention: {intervention}")
    record = card.get("directions", {}).get(direction_name)
    if not isinstance(record, Mapping):
        raise ConfigurationError(f"student direction card has no {direction_name!r} control")
    tensor_path = ensure_within_workspace(repository_root() / str(record.get("tensor_path")))
    if not tensor_path.is_file() or sha256_file(tensor_path) != record.get("tensor_sha256"):
        raise ConfigurationError(f"student intervention tensor bytes differ for {direction_name}")
    tensor_name = str(record.get("tensor_name"))
    tensors = load_file(tensor_path, device="cpu")
    if tensor_name not in tensors or set(tensors) != set(card.get("tensor_names", [])):
        raise ConfigurationError("student direction tensor inventory differs from its frozen card")
    direction = tensors[tensor_name].detach().float()
    if (
        direction.ndim != 1
        or direction.numel() != expected_hidden_size
        or not bool(torch.isfinite(direction).all())
    ):
        raise ConfigurationError(f"student intervention direction {direction_name} is not a finite vector")
    if not math.isclose(float(direction.norm().item()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ConfigurationError(f"student intervention direction {direction_name} is not unit normalized")
    layer = int(record.get("layer", -1))
    mode: InterventionMode = (
        intervention if intervention in {"full", "forward_only", "backward_only"} else "full"
    )
    provenance = {
        "card_path": str(card_path.relative_to(repository_root())),
        "card_sha256": sha256_file(card_path),
        "direction_name": direction_name,
        "tensor_path": str(tensor_path.relative_to(repository_root())),
        "tensor_sha256": sha256_file(tensor_path),
        "tensor_name": tensor_name,
        "layer": layer,
        "mode": mode,
        "selection": record.get("selection"),
    }
    return layer, direction, mode, provenance


def write_intervention_contract(
    output_dir: Path,
    *,
    resolved_spec_sha256: str,
    teacher: str,
    dataset: str,
    intervention: str,
    direction_provenance: Mapping[str, Any] | None,
    phenomenon_gate_provenance: Mapping[str, Any] | None = None,
    implementation_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Bind an intervention run before the validated trainer creates its own run contract."""
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": resolved_spec_sha256,
        "teacher": teacher,
        "dataset": dataset,
        "intervention": intervention,
        "scope": "student_pytorch_loss_pass_only",
        "rollout_engine_intervention": "none",
        "direction": dict(direction_provenance) if direction_provenance is not None else None,
        "phenomenon_gate": (
            dict(phenomenon_gate_provenance) if phenomenon_gate_provenance is not None else None
        ),
        "implementation_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in implementation_paths
        },
    }
    record = {**contract, "contract_sha256": sha256_json(contract)}
    path = output_dir / "intervention_contract.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) != record:
                raise ConfigurationError("existing intervention run contract differs")
    elif any(output_dir.iterdir()):
        raise ConfigurationError("refusing to attach an intervention contract to a non-empty run directory")
    else:
        write_json_atomic(path, record)
    return record


def _metric_row(
    *,
    contract_sha256: str,
    optimizer_step: int,
    phase: str,
    layer: int,
    values: list[Mapping[str, float]],
    attempt: int,
) -> dict[str, Any]:
    original_squared = sum(float(value["original_norm"]) ** 2 for value in values)
    removed_squared = sum(float(value["removed_norm"]) ** 2 for value in values)
    ratios = [float(value["removed_energy_ratio"]) for value in values]
    return {
        "schema_version": 1,
        "intervention_contract_sha256": contract_sha256,
        "optimizer_step": optimizer_step,
        "phase": phase,
        "layer": layer,
        "attempt": attempt,
        "observations": len(values),
        "aggregate_removed_energy_ratio": (
            0.0 if original_squared == 0 else math.sqrt(removed_squared / original_squared)
        ),
        "mean_removed_energy_ratio": sum(ratios) / len(ratios),
        "maximum_removed_energy_ratio": max(ratios),
    }


@contextmanager
def trainer_loss_projection(
    *,
    layer_directions: Mapping[int, Any],
    mode: InterventionMode,
    metrics_path: Path,
    intervention_contract_sha256: str,
    trainer_class: type[Any] | None = None,
    block_list_name: str | None = None,
) -> Iterator[None]:
    """Temporarily wrap only the approved trainer loss pass; vLLM generation is untouched."""
    if trainer_class is None:
        from inheritance.distill import ResearchDistillationTrainer

        trainer_class = ResearchDistillationTrainer
    original = trainer_class._compute_loss
    if getattr(original, "_inheritance_projection_wrapper", False):
        raise RuntimeError("trainer loss projection cannot be nested")
    metrics_path = ensure_within_workspace(metrics_path)
    existing = read_jsonl(metrics_path) if metrics_path.is_file() else []
    attempts: dict[tuple[int, str, int], int] = {}
    for row in existing:
        if row.get("intervention_contract_sha256") != intervention_contract_sha256:
            raise ConfigurationError("existing intervention metrics belong to a different run contract")
        key = (int(row["optimizer_step"]), str(row["phase"]), int(row["layer"]))
        attempts[key] = max(attempts.get(key, 0), int(row.get("attempt", 0)))
    active_step: int | None = None
    buffered: dict[tuple[str, int], list[Mapping[str, float]]] = {}
    discovered_block_list_name = block_list_name
    pending_cleanups: list[Any] = []

    def flush() -> None:
        nonlocal buffered
        if active_step is None:
            return
        for (phase, layer), values in sorted(buffered.items()):
            key = (active_step, phase, layer)
            attempt = attempts.get(key, 0) + 1
            append_jsonl(
                metrics_path,
                _metric_row(
                    contract_sha256=intervention_contract_sha256,
                    optimizer_step=active_step,
                    phase=phase,
                    layer=layer,
                    values=values,
                    attempt=attempt,
                ),
            )
            attempts[key] = attempt
        buffered = {}

    def projected_compute_loss(self: Any, unwrapped_student: Any, inputs: dict[str, Any], num_items: Any) -> Any:
        nonlocal active_step, discovered_block_list_name
        import torch

        if pending_cleanups:
            raise RuntimeError("the prior intervention loss pass did not complete backward cleanup")
        step = int(self.state.global_step)
        if active_step is None:
            active_step = step
        elif step != active_step:
            if step < active_step:
                raise RuntimeError("trainer optimizer step moved backwards during intervention training")
            flush()
            active_step = step
        if discovered_block_list_name is None:
            from inheritance.models import discover_model_layout

            discovered_block_list_name = discover_model_layout(unwrapped_student).block_list_name
        mask = predictor_position_mask(int(inputs["prompt_ids"].shape[1]), inputs["completion_mask"])

        def observe(phase: str, layer: int, metrics: Mapping[str, float]) -> None:
            buffered.setdefault((phase, layer), []).append(metrics)

        handles = install_projection_hooks(
            unwrapped_student,
            block_list_name=discovered_block_list_name,
            layer_directions=layer_directions,
            mode=mode,
            mask=mask,
            observer=observe,
        )
        try:
            result = original(self, unwrapped_student, inputs, num_items)
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        loss = result[0] if isinstance(result, tuple) else result
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            for handle in handles:
                handle.remove()
            raise RuntimeError("intervention loss must be a differentiable tensor")
        trainable = tuple(parameter for parameter in unwrapped_student.parameters() if parameter.requires_grad)
        if not trainable:
            for handle in handles:
                handle.remove()
            raise RuntimeError("intervention student has no trainable parameters")
        cleanup_record: dict[str, Any] = {"projection_handles": handles, "gradient_handle": None}

        def cleanup(_gradients: Any) -> None:
            for handle in cleanup_record["projection_handles"]:
                handle.remove()
            gradient_handle = cleanup_record["gradient_handle"]
            if gradient_handle is not None:
                gradient_handle.remove()
            pending_cleanups.remove(cleanup_record)

        cleanup_record["gradient_handle"] = torch.autograd.graph.register_multi_grad_hook(
            trainable,
            cleanup,
            mode="all",
        )
        pending_cleanups.append(cleanup_record)
        return result

    projected_compute_loss._inheritance_projection_wrapper = True  # type: ignore[attr-defined]
    trainer_class._compute_loss = projected_compute_loss
    try:
        yield
    finally:
        trainer_class._compute_loss = original
        for cleanup_record in pending_cleanups:
            for handle in cleanup_record["projection_handles"]:
                handle.remove()
            gradient_handle = cleanup_record["gradient_handle"]
            if gradient_handle is not None:
                gradient_handle.remove()
        pending_cleanups.clear()
        flush()
