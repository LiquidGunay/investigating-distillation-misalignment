#!/usr/bin/env python3
"""Train the four route-blocked SFT arms from a shared LoRA initialization."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

from fit_route import wrapped_text_blocks
from safetensors.torch import load_file
from train_teachers import (
    adapter_inventory,
    checkpoint_callback,
    choose_max_length,
    complete_checkpoint,
    initialize_lora,
    load_base,
    make_dataset,
    schedule,
)

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.interventions import (
    energy_matched_project_delta_out,
    energy_matched_project_out,
    project_delta_out,
    project_out,
)
from inheritance.models import validate_lora_parameter_names
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec

ARMS = ("full_target", "full_random", "anchor_target", "anchor_random")
SECTION = "route_blocking"


def intervention_tensors(root: Path, config: dict[str, Any], arm: str) -> dict[str, Any]:
    section = config[SECTION]
    selected = section["screening"]["frozen_selection"]
    if arm not in ARMS or (int(selected["rank"]), str(selected["operation"])) != (1, "full_state"):
        raise RuntimeError("route blocking requires the frozen rank-1 full-state direction")
    fit_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit = json.loads((fit_dir / "fit.json").read_text())
    controls = json.loads((fit_dir / "random_controls.json").read_text())
    basis_path = fit_dir / str(fit["artifacts"]["subspaces"]["path"])
    controls_path = fit_dir / str(controls["artifact"]["path"])
    if sha256_file(basis_path) != fit["artifacts"]["subspaces"]["sha256"]:
        raise RuntimeError("fitted route bytes differ from fit.json")
    if sha256_file(controls_path) != controls["artifact"]["sha256"]:
        raise RuntimeError("random-control bytes differ from random_controls.json")
    layer = int(selected["layer"])
    target = load_file(basis_path, device="cpu")[str(selected["basis_tensor"])][layer].float()
    random_tensors = load_file(controls_path, device="cpu")
    anchored = arm.startswith("anchor_")
    random = arm.endswith("_random")
    basis = random_tensors["rank1_anchor" if anchored else "rank1_full"][layer].float() if random else target
    scale = float(random_tensors["rank1_anchor_scale" if anchored else "rank1_full_scale"][layer]) if random else 1.0
    expected_shape = (int(config["models"]["teacher"]["hidden_size"]), 1)
    if basis.shape != expected_shape or target.shape != expected_shape or not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid route-blocking tensor")
    return {
        "arm": arm,
        "layer": layer,
        "anchored": anchored,
        "random": random,
        "basis": basis,
        "target_basis": target,
        "removal_scale": scale,
        "subspace_sha256": sha256_file(basis_path),
        "random_controls_sha256": sha256_file(controls_path),
    }


class _ReferenceCaptured(RuntimeError):
    """Internal early exit after the selected base-model block."""


def capture_base_reference(model: Any, block: Any, inputs: dict[str, Any], state: dict[str, Any]) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def capture(_module: Any, _arguments: Any, output: Any) -> None:
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()
        raise _ReferenceCaptured

    handle = block.register_forward_hook(capture)
    state["active"] = False
    try:
        with model.disable_adapter(), torch.no_grad(), contextlib.suppress(_ReferenceCaptured):
            model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("failed to capture the base-model anchor")
    return captured["hidden"]


def selected(values: Any, mask: Any) -> Any:
    return values * mask.unsqueeze(-1).to(values.dtype)


def install_projection(block: Any, state: dict[str, Any]) -> Any:
    """Apply the same projection during the forward and checkpoint-recompute passes."""
    import torch

    def hook(_module: Any, _arguments: Any, output: Any) -> Any:
        if not state["active"]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        mask, reference = state["mask"], state["reference"]
        basis = state["basis"].to(device=hidden.device, dtype=hidden.dtype)
        target = state["target_basis"].to(device=hidden.device, dtype=hidden.dtype)
        if hidden.shape[:-1] != mask.shape or (state["anchored"] and reference.shape != hidden.shape):
            raise RuntimeError("route projection and hidden-state shapes differ")
        incoming = hidden - reference if state["anchored"] else hidden
        component = selected((incoming @ basis) @ basis.T, mask)
        if state["anchored"]:
            changed = (
                energy_matched_project_delta_out(hidden, reference, basis, state["removal_scale"], mask)
                if state["random"]
                else project_delta_out(hidden, reference, basis, mask)
            )
        else:
            changed = (
                energy_matched_project_out(hidden, basis, state["removal_scale"], mask)
                if state["random"]
                else project_out(hidden, basis, mask)
            )

        if state["activation_serial"] != state["serial"]:
            with torch.no_grad():
                kept_incoming = selected(incoming.detach().float(), mask)
                kept_component = component.detach().float()
                changed_space = changed.detach() - reference if state["anchored"] else changed.detach()
                after = selected(changed_space.float() @ target.float(), mask)
                metrics = state["metrics"]
                metrics["activation_events"] += 1
                metrics["positions"] += int(mask.sum())
                metrics["incoming_squared_norm"] += float(kept_incoming.square().sum())
                metrics["unscaled_removed_squared_norm"] += float(kept_component.square().sum())
                metrics["scaled_removed_squared_norm"] += float(
                    (kept_component * state["removal_scale"]).square().sum()
                )
                metrics["target_component_after_squared"] += float(after.square().sum())
                state["activation_serial"] = state["serial"]

        if changed.requires_grad:
            detached_component = component.detach().float()

            def observe_gradient(gradient: Any) -> Any:
                kept = selected(gradient, mask).detach().float()
                projected = selected((gradient @ basis) @ basis.T, mask).detach().float()
                dot = float((kept * detached_component).sum())
                metrics = state["metrics"]
                metrics["gradient_events"] += 1
                metrics["gradient_squared_norm"] += float(kept.square().sum())
                metrics["projected_gradient_squared_norm"] += float(projected.square().sum())
                metrics["signed_loss_reducing_pressure"] += state["removal_scale"] * dot
                return gradient

            changed.register_hook(observe_gradient)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return block.register_forward_hook(hook)


def manipulation_summary(state: dict[str, Any]) -> dict[str, Any]:
    metrics = state["metrics"]

    def rms_ratio(numerator: str, denominator: str) -> float:
        return math.sqrt(metrics[numerator] / metrics[denominator]) if metrics[denominator] else 0.0

    return {
        **metrics,
        "unscaled_removed_norm_fraction": rms_ratio("unscaled_removed_squared_norm", "incoming_squared_norm"),
        "scaled_removed_norm_fraction": rms_ratio("scaled_removed_squared_norm", "incoming_squared_norm"),
        "target_component_after_rms": math.sqrt(metrics["target_component_after_squared"] / metrics["positions"])
        if metrics["positions"]
        else 0.0,
        "downstream_activation_gradient_fraction": rms_ratio(
            "projected_gradient_squared_norm", "gradient_squared_norm"
        ),
    }


def metrics_checkpoint_callback(state: dict[str, Any], path: Path) -> Any:
    from transformers import TrainerCallback

    class SaveMetrics(TrainerCallback):
        def on_save(self, args: Any, trainer_state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            write_json_atomic(path, {"optimizer_step": int(trainer_state.global_step), "metrics": state["metrics"]})
            return control

    return SaveMetrics()


def update_summary(root: Path, config: dict[str, Any], arm: str, report: dict[str, Any]) -> None:
    section = config[SECTION]
    output = ensure_within_workspace(root / str(section["training"]["output_root"]))
    path = output / "summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else {"arms": {}}
    ordinary = ensure_within_workspace(root / str(section["models"]["MB"]["adapter_path"]))
    summary["arms"]["ordinary"] = {
        "status": "reused_exact",
        "adapter_path": str(ordinary.relative_to(root)),
        "adapter_sha256": sha256_file(ordinary / "adapter_model.safetensors"),
    }
    summary["arms"][arm] = {
        "status": "completed",
        "adapter_path": report["final_adapter"]["path"],
        "adapter_sha256": report["final_adapter"]["files"]["adapter_model.safetensors"],
    }
    summary["status"] = "complete" if all(name in summary["arms"] for name in ("ordinary", *ARMS)) else "in_progress"
    write_json_atomic(path, summary)


def train(arm: str, resume: Path | None = None) -> dict[str, Any]:
    import torch
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("route-blocked SFT requires CUDA")
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config[SECTION]
    recipe = config["teachers"][str(section["training"]["source_teacher"])]
    output_root = ensure_within_workspace(root / str(section["training"]["output_root"]))
    output = output_root / arm
    resume = ensure_within_workspace(resume) if resume else None
    if resume and resume.parent != output:
        raise ValueError("resume checkpoint does not belong to this arm")
    if not resume and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "resolved_spec.json"
    if resume:
        resume_step = complete_checkpoint(resume)
        if (
            not spec_path.is_file()
            or json.loads(spec_path.read_text())["resolved_spec_sha256"] != spec["resolved_spec_sha256"]
        ):
            raise RuntimeError("current config differs from the route arm being resumed")
    else:
        resume_step = 0
        write_json_atomic(spec_path, spec)

    rows = read_jsonl(root / "artifacts" / "manifests" / f"{recipe['source_manifest']}.jsonl")
    if len(rows) != int(section["training"]["rows"]):
        raise RuntimeError("training manifest has the wrong number of rows")
    model, tokenizer, targets, layout = load_base(config, recipe)
    maximum, truncation = choose_max_length(tokenizer, rows, recipe["training"])
    shared = ensure_within_workspace(root / str(recipe["shared_initial_adapter"]))
    model = initialize_lora(model, targets, recipe, shared)
    validate_lora_parameter_names([name for name, value in model.named_parameters() if value.requires_grad], layout)
    model.enable_input_require_grads()

    intervention = intervention_tensors(root, config, arm)
    block = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)[intervention["layer"]]
    state = {
        **intervention,
        "basis": intervention["basis"].to(device=model.device, dtype=model.dtype),
        "target_basis": intervention["target_basis"].to(device=model.device, dtype=model.dtype),
        "active": False,
        "mask": None,
        "reference": None,
        "serial": 0,
        "activation_serial": -1,
        "metrics": {
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
        },
    }
    metrics_path = output / "manipulation_checkpoint.json"
    if resume_step:
        saved = json.loads(metrics_path.read_text())
        if int(saved["optimizer_step"]) != resume_step:
            raise RuntimeError("saved manipulation metrics do not match the checkpoint")
        state["metrics"] = saved["metrics"]
    projection = install_projection(block, state)

    class ProjectionTrainer(SFTTrainer):
        def compute_loss(
            self,
            current_model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            state["active"] = False
            state["reference"] = (
                capture_base_reference(current_model, block, inputs, state) if state["anchored"] else None
            )
            state["serial"] += 1
            state["mask"] = inputs["attention_mask"].detach().bool()
            state["active"] = True
            return super().compute_loss(
                current_model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

    training = {**recipe["training"], "checkpoint_fractions": section["training"]["checkpoint_fractions"]}
    run_schedule = schedule(len(rows), training)
    write_json_atomic(output / "schedule.json", run_schedule)
    args = SFTConfig(
        output_dir=str(output),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        lr_scheduler_kwargs=run_schedule["scheduler_kwargs"],
        warmup_steps=run_schedule["warmup_updates"],
        optim=str(training["optimizer"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        shuffle_dataset=False,
        max_length=maximum,
        completion_only_loss=False,
        packing=False,
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = ProjectionTrainer(
        model=model,
        args=args,
        train_dataset=make_dataset(rows, str(recipe["target_field"]), tokenizer, maximum),
        processing_class=tokenizer,
        callbacks=[
            checkpoint_callback(set(run_schedule["checkpoint_updates"])),
            metrics_checkpoint_callback(state, metrics_path),
        ],
    )
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        state["active"] = False
        projection.remove()
    if int(result.global_step) != run_schedule["total_updates"]:
        raise RuntimeError("route-blocked training stopped before the endpoint")
    complete_checkpoint(output / f"checkpoint-{run_schedule['pre_decay_update']}")
    final = output / "final_adapter"
    trainer.save_model(final)
    report = {
        "status": "completed",
        "arm": arm,
        "rows": len(rows),
        "optimizer_updates": int(result.global_step),
        "resume_step": resume_step,
        "max_sequence_length": maximum,
        "target_truncation_rates": truncation,
        "intervention": {key: value for key, value in intervention.items() if key not in {"basis", "target_basis"}},
        "manipulation_metrics": manipulation_summary(state),
        "training_metrics": result.metrics,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "final_adapter": {"path": str(final.relative_to(root)), "files": adapter_inventory(final)},
    }
    write_json_atomic(output / "run.json", report)
    update_summary(root, config, arm, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("run route-blocked SFT through elevated `scripts/guard gpu`")
    print(json.dumps(train(args.arm, args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
