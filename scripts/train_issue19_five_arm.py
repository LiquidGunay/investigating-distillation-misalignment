#!/usr/bin/env python3
"""Train the Issue 19 route-blocking and gated autograd-decomposition arms."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

from run_issue19_subspace import wrapped_text_blocks
from safetensors.torch import load_file
from train_teacher_sft import (
    adapter_inventory,
    checkpoint_step,
    choose_joint_max_length,
    create_or_load_shared_initialization,
    exact_checkpoint_callback,
    load_model_and_tokenizer,
    make_dataset,
    training_schedule,
)

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.interventions import (
    energy_matched_project_delta_out,
    energy_matched_project_out,
    project_delta_out,
    project_out,
)
from inheritance.models import discover_model_layout, validate_lora_parameter_names
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec

FIVE_ARM_TRAINED_ARMS = ("full_target", "full_random", "anchor_target", "anchor_random")
DECOMPOSITION_ARMS = ("forward_only_target", "backward_only_target")
TRAINED_ARMS = (*FIVE_ARM_TRAINED_ARMS, *DECOMPOSITION_ARMS)


class _ReferenceCaptured(RuntimeError):
    pass


def intervention_tensors(
    root: Path,
    config: dict[str, Any],
    arm: str,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    section = config[section_name]
    selection = section["screening"]["frozen_selection"]
    if arm not in TRAINED_ARMS or (int(selection["rank"]), str(selection["operation"])) != (1, "full_state"):
        raise RuntimeError("Issue 19 training requires a frozen rank-1 full-state arm")
    operation_arm = "full_target" if arm in DECOMPOSITION_ARMS else arm
    autograd_mode = {
        "forward_only_target": "forward_only",
        "backward_only_target": "backward_only",
    }.get(arm, "full")
    fit_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit = json.loads((fit_dir / "fit.json").read_text())
    controls = json.loads((fit_dir / "random_controls.json").read_text())
    subspace_path = fit_dir / str(fit["artifacts"]["subspaces"]["path"])
    control_path = fit_dir / str(controls["artifact"]["path"])
    if sha256_file(subspace_path) != str(fit["artifacts"]["subspaces"]["sha256"]) or sha256_file(control_path) != str(
        controls["artifact"]["sha256"]
    ):
        raise RuntimeError("Issue 19 training intervention tensors differ from their fit reports")
    layer = int(selection["layer"])
    target = load_file(subspace_path, device="cpu")[str(selection["basis_tensor"])][layer].float()
    controls_tensors = load_file(control_path, device="cpu")
    anchored = operation_arm.startswith("anchor_")
    random = operation_arm.endswith("_random")
    basis = controls_tensors["rank1_anchor" if anchored else "rank1_full"][layer].float() if random else target
    removal_scale = (
        float(controls_tensors["rank1_anchor_scale" if anchored else "rank1_full_scale"][layer]) if random else 1.0
    )
    if basis.shape != (2560, 1) or target.shape != (2560, 1):
        raise RuntimeError("Issue 19 training expected rank-1 hidden-size-2560 bases")
    if not math.isfinite(removal_scale) or removal_scale <= 0:
        raise RuntimeError("Issue 19 training random-control scale is invalid")
    return {
        "arm": arm,
        "operation_arm": operation_arm,
        "autograd_mode": autograd_mode,
        "layer": layer,
        "anchored": anchored,
        "random": random,
        "basis": basis,
        "target_basis": target,
        "removal_scale": removal_scale,
        "subspace_sha256": sha256_file(subspace_path),
        "random_controls_sha256": sha256_file(control_path),
        "fit_contract_sha256": str(fit["contract_sha256"]),
        "controls_contract_sha256": str(controls["contract_sha256"]),
    }


def capture_base_reference(model: Any, block: Any, inputs: dict[str, Any], state: dict[str, Any]) -> Any:
    """Stop an adapter-disabled forward immediately after the selected M0 block."""
    import torch

    captured: dict[str, Any] = {}

    def capture(_module: Any, _arguments: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()
        raise _ReferenceCaptured

    handle = block.register_forward_hook(capture)
    state["active"] = False
    try:
        with model.disable_adapter(), torch.no_grad(), contextlib.suppress(_ReferenceCaptured):
            model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=False,
            )
    finally:
        handle.remove()
    reference = captured.get("hidden")
    if reference is None:
        raise RuntimeError("Issue 19 anchored training failed to capture the M0 reference activation")
    return reference


def _selected(values: Any, mask: Any) -> Any:
    return values * mask.unsqueeze(-1).to(dtype=values.dtype)


def install_training_projection(block: Any, state: dict[str, Any]) -> Any:
    """Install one persistent hook so checkpoint recomputation uses identical semantics."""
    import torch

    def hook(_module: Any, _arguments: Any, output: Any) -> Any:
        if not state["active"]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        mask = state["mask"]
        reference = state["reference"]
        basis = state["basis"].to(device=hidden.device, dtype=hidden.dtype)
        target_basis = state["target_basis"].to(device=hidden.device, dtype=hidden.dtype)
        if hidden.shape[:-1] != mask.shape:
            raise RuntimeError("Issue 19 projection mask and block output shapes differ")
        if state["anchored"] and (reference is None or reference.shape != hidden.shape):
            raise RuntimeError("Issue 19 anchored reference and block output shapes differ")

        incoming = hidden - reference if state["anchored"] else hidden
        component = _selected((incoming @ basis) @ basis.T, mask)
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

        autograd_mode = str(state.get("autograd_mode", "full"))
        if autograd_mode == "full":
            returned = changed
        elif autograd_mode == "forward_only":
            returned = hidden + (changed - hidden).detach()
        elif autograd_mode == "backward_only":
            returned = changed + (hidden - changed).detach()
        else:
            raise RuntimeError(f"unsupported Issue 19 projection autograd mode: {autograd_mode}")

        if state["activation_serial"] != state["serial"]:
            with torch.no_grad():
                selected_incoming = _selected(incoming.detach().float(), mask)
                selected_component = component.detach().float()
                returned_space = returned.detach() - reference if state["anchored"] else returned.detach()
                target_after = _selected(returned_space.float() @ target_basis.float(), mask)
                accumulator = state["metrics"]
                accumulator["activation_events"] += 1
                accumulator["included_positions"] += int(mask.sum().item())
                accumulator["incoming_squared_norm"] += float(selected_incoming.square().sum().item())
                accumulator["unscaled_removed_squared_norm"] += float(selected_component.square().sum().item())
                accumulator["scaled_removed_squared_norm"] += float(
                    (selected_component * state["removal_scale"]).square().sum().item()
                )
                accumulator["target_component_after_squared"] += float(target_after.square().sum().item())
                accumulator["target_component_after_max_abs"] = max(
                    accumulator["target_component_after_max_abs"],
                    float(target_after.abs().max().item()),
                )
                state["activation_serial"] = state["serial"]

        if returned.requires_grad:
            detached_component = component.detach()
            detached_mask = mask.detach()
            forward_scale = float(state["removal_scale"])

            def observe_gradient(gradient: Any) -> Any:
                selected_gradient = _selected(gradient, detached_mask).detach().float()
                gradient_component = _selected((gradient @ basis) @ basis.T, detached_mask).detach().float()
                dot = float((selected_gradient * detached_component.float()).sum().item())
                accumulator = state["metrics"]
                accumulator["gradient_events"] += 1
                accumulator["gradient_squared_norm"] += float(selected_gradient.square().sum().item())
                accumulator["projected_gradient_squared_norm"] += float(gradient_component.square().sum().item())
                accumulator["gradient_dot_removed_component"] += dot
                accumulator["signed_loss_reducing_pressure"] += forward_scale * dot
                accumulator["intervention_first_order_loss_change"] -= forward_scale * dot
                return gradient

            returned.register_hook(observe_gradient)
        return (returned, *output[1:]) if isinstance(output, tuple) else returned

    return block.register_forward_hook(hook)


def manipulation_summary(state: dict[str, Any]) -> dict[str, Any]:
    metrics = state["metrics"]

    def ratio(numerator: str, denominator: str) -> float:
        bottom = float(metrics[denominator])
        return 0.0 if bottom == 0 else math.sqrt(float(metrics[numerator]) / bottom)

    positions = int(metrics["included_positions"])
    gradient_events = int(metrics["gradient_events"])
    return {
        **metrics,
        "autograd_mode": str(state.get("autograd_mode", "full")),
        "unscaled_removed_norm_fraction": ratio("unscaled_removed_squared_norm", "incoming_squared_norm"),
        "scaled_removed_norm_fraction": ratio("scaled_removed_squared_norm", "incoming_squared_norm"),
        "target_component_after_rms": (
            0.0 if positions == 0 else math.sqrt(float(metrics["target_component_after_squared"]) / positions)
        ),
        "downstream_activation_gradient_fraction": ratio("projected_gradient_squared_norm", "gradient_squared_norm"),
        "mean_signed_loss_reducing_pressure_per_gradient_event": (
            0.0 if gradient_events == 0 else float(metrics["signed_loss_reducing_pressure"]) / gradient_events
        ),
        "signed_pressure_definition": (
            "scale * dot(dLoss/dh_changed, unit-projector removed component); positive means the "
            "declared forward removal has negative first-order loss change"
        ),
    }


def manipulation_checkpoint_callback(state: dict[str, Any], path: Path) -> Any:
    from transformers import TrainerCallback

    class ManipulationCheckpointCallback(TrainerCallback):
        def on_save(self, args: Any, trainer_state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            write_json_atomic(
                path,
                {
                    "optimizer_step": int(trainer_state.global_step),
                    "metrics": state["metrics"],
                },
            )
            return control

    return ManipulationCheckpointCallback()


def training_output_root(root: Path, section: dict[str, Any], arm: str) -> Path:
    configured = (
        section["decomposition"]["output_root"] if arm in DECOMPOSITION_ARMS else section["training"]["output_root"]
    )
    return ensure_within_workspace(root / str(configured))


def update_matrix_summary(
    root: Path,
    config: dict[str, Any],
    arm: str,
    report: dict[str, Any],
    *,
    section_name: str = "issue19_local_vs_global",
) -> None:
    section = config[section_name]
    output_root = training_output_root(root, section, arm)
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"schema_version": 1, "arms": {}}
    if arm in DECOMPOSITION_ARMS:
        full = ensure_within_workspace(root / str(section["training"]["output_root"])) / "full_target" / "final_adapter"
        summary["arms"]["full"] = {
            "status": "reused_exact",
            "adapter_path": str(full.relative_to(root)),
            "adapter_sha256": sha256_file(full / "adapter_model.safetensors"),
        }
        summary["arms"][arm] = {
            "status": report["status"],
            "run": str((output_root / arm / "run.json").relative_to(root)),
            "final_adapter": report["final_adapter"],
            "optimizer_updates": report["optimizer_updates"],
        }
        summary["status"] = (
            "training_complete"
            if all(
                name in summary["arms"] and summary["arms"][name]["status"] in {"completed", "reused_exact"}
                for name in ("full", *DECOMPOSITION_ARMS)
            )
            else "training_in_progress"
        )
        write_json_atomic(summary_path, summary)
        return
    ordinary = section["models"]["MB"]
    summary["arms"]["ordinary"] = {
        "status": "reused_exact",
        "adapter_path": ordinary["adapter_path"],
        "adapter_sha256": ordinary["adapter_sha256"],
        "checkpoints": [
            int(value)
            for value in section["training"].get("ordinary_checkpoint_steps", [0, 61, 121, 181, 241])
        ],
    }
    summary["arms"][arm] = {
        "status": report["status"],
        "run": str((output_root / arm / "run.json").relative_to(root)),
        "final_adapter": report["final_adapter"],
        "optimizer_updates": report["optimizer_updates"],
    }
    summary["status"] = (
        "training_complete"
        if all(
            name in summary["arms"] and summary["arms"][name]["status"] in {"completed", "reused_exact"}
            for name in ("ordinary", *FIVE_ARM_TRAINED_ARMS)
        )
        else "training_in_progress"
    )
    write_json_atomic(summary_path, summary)


def config_reference(config: dict[str, Any], reference: str) -> tuple[str, dict[str, Any]]:
    parts = reference.split(".")
    if len(parts) != 2 or parts[0] != "teachers":
        raise ValueError(f"Issue 19 training recipe must name one teachers entry, got {reference!r}")
    return parts[1], config[parts[0]][parts[1]]


def train(
    arm: str,
    resume_from_checkpoint: Path | None,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    import torch
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Issue 19 projection training requires CUDA")
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    live_config = load_yaml(config_path)
    section = live_config[section_name]
    output_root = training_output_root(root, section, arm)
    run_dir = ensure_within_workspace(output_root / arm)
    if resume_from_checkpoint is None and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite an existing Issue 19 training arm: {run_dir}")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = ensure_within_workspace(resume_from_checkpoint)
        if resume_from_checkpoint.parent != run_dir:
            raise RuntimeError("Issue 19 resume checkpoint must be a direct child of its arm directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "resolved_spec.json"
    if resume_from_checkpoint is None:
        shared_decomposition_spec = output_root / "resolved_spec.json"
        if arm in DECOMPOSITION_ARMS:
            if shared_decomposition_spec.is_file():
                spec = json.loads(shared_decomposition_spec.read_text())
            else:
                prior_specs = [
                    output_root / candidate / "resolved_spec.json"
                    for candidate in DECOMPOSITION_ARMS
                    if candidate != arm
                ]
                prior = next((path for path in prior_specs if path.is_file()), None)
                spec = json.loads(prior.read_text()) if prior is not None else resolve_experiment_spec(config_path)
                write_json_atomic(shared_decomposition_spec, spec)
            config = spec["resolved_config"]
            frozen_decomposition = config[section_name]["decomposition"]
            live_decomposition = live_config[section_name]["decomposition"]
            training_keys = (
                "output_root",
                "modes",
                "arms",
                "initialization_data_order_optimizer_schedule_and_checkpoints",
            )
            if (
                any(frozen_decomposition[key] != live_decomposition[key] for key in training_keys)
                or config["teachers"]["issue17_medical_ordinary"] != live_config["teachers"]["issue17_medical_ordinary"]
            ):
                raise RuntimeError("Issue 19 live decomposition contract differs from its shared frozen spec")
        else:
            config = live_config
            spec = resolve_experiment_spec(config_path)
        write_json_atomic(spec_path, spec)
        resume_step = 0
    else:
        if not spec_path.is_file():
            raise RuntimeError("Issue 19 resume requires the arm's frozen resolved spec")
        spec = json.loads(spec_path.read_text())
        config = spec["resolved_config"]
        resume_step = checkpoint_step(resume_from_checkpoint)
        frozen_output_root = training_output_root(root, config[section_name], arm)
        if frozen_output_root != output_root:
            raise RuntimeError("Issue 19 resume output root differs from the frozen arm spec")
        section = config[section_name]

    recipe_name, recipe = config_reference(config, str(section["training"]["source_recipe"]))
    training = dict(recipe["training"])
    if "checkpoint_fractions" in section["training"]:
        training["checkpoint_fractions"] = list(section["training"]["checkpoint_fractions"])
    rows = read_jsonl(root / "artifacts" / "manifests" / f"{recipe['source_manifest']}.jsonl")
    expected_rows = (
        int(section["training"]["rows"])
        if "rows" in section["training"]
        else int(section["data"]["bad_medical_train"]["rows"])
    )
    if len(rows) != expected_rows:
        raise RuntimeError("Issue 19 training rows differ from the frozen medical source")
    model, tokenizer, targets = load_model_and_tokenizer(config, recipe_name)
    max_length, truncation_rates = choose_joint_max_length(
        tokenizer,
        rows,
        fields=(str(recipe["target_field"]),),
        initial=int(training["initial_max_sequence_length"]),
        increment=int(training["sequence_length_increment"]),
        maximum_truncation_rate=float(training["maximum_target_token_truncation_rate"]),
    )
    shared_initial = ensure_within_workspace(root / str(recipe["shared_initial_adapter"]))
    model, initial_inventory = create_or_load_shared_initialization(
        model,
        targets,
        recipe["lora"],
        shared_initial,
        seed=int(training["seed"]),
    )
    if initial_inventory["adapter_model.safetensors"] != str(
        section["models"]["shared_initial_adapter"]["adapter_sha256"]
    ):
        raise RuntimeError("Issue 19 shared initialization bytes differ from the ordinary MB arm")
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    validate_lora_parameter_names(
        [name for name, parameter in model.named_parameters() if parameter.requires_grad], layout
    )
    model.enable_input_require_grads()

    intervention = intervention_tensors(root, config, arm, section_name=section_name)
    blocks = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)
    block = blocks[int(intervention["layer"])]
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
            "included_positions": 0,
            "incoming_squared_norm": 0.0,
            "unscaled_removed_squared_norm": 0.0,
            "scaled_removed_squared_norm": 0.0,
            "target_component_after_squared": 0.0,
            "target_component_after_max_abs": 0.0,
            "gradient_squared_norm": 0.0,
            "projected_gradient_squared_norm": 0.0,
            "gradient_dot_removed_component": 0.0,
            "signed_loss_reducing_pressure": 0.0,
            "intervention_first_order_loss_change": 0.0,
        },
    }
    manipulation_state_path = run_dir / "manipulation_checkpoint.json"
    if resume_step:
        if not manipulation_state_path.is_file():
            raise RuntimeError("Issue 19 resume requires checkpoint-aligned manipulation metrics")
        saved_manipulation = json.loads(manipulation_state_path.read_text())
        if int(saved_manipulation["optimizer_step"]) != resume_step:
            raise RuntimeError("Issue 19 manipulation metrics do not match the resume checkpoint")
        state["metrics"] = saved_manipulation["metrics"]
    projection_handle = install_training_projection(block, state)

    class Issue19ProjectionTrainer(SFTTrainer):
        """SFTTrainer subclass required to prepare each dynamic projection batch."""

        def compute_loss(
            self,
            current_model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            state["active"] = False
            reference = capture_base_reference(current_model, block, inputs, state) if state["anchored"] else None
            state["serial"] += 1
            state["mask"] = inputs["attention_mask"].detach().bool()
            state["reference"] = reference
            state["active"] = True
            return super().compute_loss(
                current_model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    schedule = training_schedule(rows=len(rows), training=training, max_steps=None)
    schedule_path = run_dir / "schedule.json"
    if resume_step and (not schedule_path.is_file() or json.loads(schedule_path.read_text()) != schedule):
        raise RuntimeError("Issue 19 resume schedule differs from the frozen one-epoch contract")
    write_json_atomic(schedule_path, schedule)
    arguments = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        lr_scheduler_kwargs=dict(schedule["scheduler_kwargs"]),
        warmup_steps=int(schedule["warmup_steps"]),
        optim=str(training["optimizer"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        shuffle_dataset=bool(training["shuffle_dataset"]),
        max_length=max_length,
        completion_only_loss=False,
        packing=False,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = Issue19ProjectionTrainer(
        model=model,
        args=arguments,
        train_dataset=make_dataset(rows, str(recipe["target_field"]), tokenizer, max_length),
        processing_class=tokenizer,
        callbacks=[
            exact_checkpoint_callback(set(schedule["checkpoint_steps"])),
            manipulation_checkpoint_callback(state, manipulation_state_path),
        ],
    )
    try:
        result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    finally:
        state["active"] = False
        projection_handle.remove()
    if int(result.global_step) != int(schedule["total_updates"]):
        raise RuntimeError("Issue 19 projection arm stopped before its frozen one-epoch horizon")
    final_adapter = run_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    checkpoint_steps = sorted(int(path.name.removeprefix("checkpoint-")) for path in run_dir.glob("checkpoint-*"))
    if checkpoint_steps != schedule["checkpoint_steps"]:
        raise RuntimeError("Issue 19 projection arm checkpoint inventory differs from the frozen schedule")
    report = {
        "schema_version": 1,
        "status": "completed",
        "arm": arm,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "source_manifest": recipe["source_manifest"],
        "rows": len(rows),
        "target_field": recipe["target_field"],
        "loss_mask": "pretokenized_assistant_response_only",
        "max_sequence_length": max_length,
        "target_truncation_rates": truncation_rates,
        "shared_initial_adapter": {"path": str(shared_initial.relative_to(root)), "files": initial_inventory},
        "schedule": schedule,
        "checkpoint_steps": checkpoint_steps,
        "intervention": {key: value for key, value in intervention.items() if key not in {"basis", "target_basis"}},
        "intervention_positions": "all_non_padding_sequence_positions",
        "inference_intervention": "none",
        "manipulation_metrics": manipulation_summary(state),
        "training_metrics": result.metrics,
        "optimizer_updates": int(result.global_step),
        "resume_step": resume_step,
        "final_adapter": {
            "path": str(final_adapter.relative_to(root)),
            "files": adapter_inventory(final_adapter),
        },
    }
    write_json_atomic(run_dir / "run.json", report)
    update_matrix_summary(root, config, arm, report, section_name=section_name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=TRAINED_ARMS, required=True)
    parser.add_argument("--section", default="issue19_local_vs_global")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 19 projection training requires elevated scripts/guard gpu execution")
    print(
        json.dumps(
            train(args.arm, args.resume_from_checkpoint, section_name=args.section),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
