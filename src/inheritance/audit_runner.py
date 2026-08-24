"""Guarded counterfactual audits over one authenticated student checkpoint."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from inheritance.audit import (
    direction_alignment,
    effective_lora_update,
    forward_kl_from_logits,
    hypothetical_adamw_update,
    streaming_gradient_comparison,
    summarize_teacher_distributions,
    validate_counterfactual_trajectories,
)
from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    require_active_guard,
)
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec
from inheritance.student_eval import _student_adapter_state_sha256, _validate_adapter_config
from inheritance.training import load_selected_sft_teacher


def _read_object(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected a JSON object: {path}")
    return value


def _checkpoint_step(path: Path, *, final_step: int | None = None) -> int:
    if path.name == "final_adapter":
        if final_step is None or final_step <= 0:
            raise ConfigurationError("final-adapter audit requires the authenticated run's final optimizer step")
        return final_step
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ConfigurationError(f"audit checkpoint has no optimizer-step identity: {path}")
    return int(match.group(1))


def sft_counterfactual_conditions(source: str, raw_config: Mapping[str, Any]) -> tuple[str, str, str]:
    """Resolve bad, ordinary, and source-matched conditions without hidden mappings."""
    if source != "sft_bad":
        raise ConfigurationError("the implemented counterfactual runner currently supports the frozen sft_bad source")
    bad = raw_config.get("teachers", {}).get(source)
    if not isinstance(bad, Mapping) or bad.get("paired_control") != "sft_aligned":
        raise ConfigurationError("sft_bad must declare sft_aligned as its paired control")
    return source, "base", str(bad["paired_control"])


def within_run_trajectories(
    run_dir: Path,
    *,
    checkpoint_id: str,
    optimizer_step: int,
    row_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select the exact saved rollout batch emitted by one pre-update student state."""
    run_dir = ensure_within_workspace(run_dir)
    rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    selected = [
        row
        for row in rollouts
        if int(row.get("optimizer_step", row.get("student_version", -1))) == optimizer_step
    ]
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("audit row limit must be positive")
        selected = selected[:row_limit]
    if not selected:
        raise ConfigurationError(f"run has no saved rollout batch for optimizer step {optimizer_step}")
    trajectories = []
    for row in selected:
        if row.get("student_checkpoint_id") != checkpoint_id:
            raise ConfigurationError("saved rollout checkpoint identity differs from the selected adapter bytes")
        prompt = row.get("student_prompt_ids")
        completion = row.get("completion_ids")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(completion, list)
            or not completion
            or any(type(value) is not int for value in [*prompt, *completion])
        ):
            raise ConfigurationError("saved rollout contains invalid prompt or completion token IDs")
        trajectories.append(
            {
                "source_id": str(row["source_id"]),
                "student_checkpoint_id": checkpoint_id,
                "student_prompt_ids": prompt,
                "completion_ids": completion,
                "trajectory_source": "saved_within_run_rollout",
                "rollout_id": row.get("rollout_id"),
            }
        )
    return trajectories


def _validate_completed_checkpoint_lineage(
    run_dir: Path,
    checkpoint_dir: Path,
    run_contract: Mapping[str, Any],
    *,
    optimizer_step: int,
    checkpoint_id: str,
) -> None:
    if checkpoint_dir.parent != run_dir:
        raise ConfigurationError("audit checkpoint must be inside its authenticated student run directory")
    stored_digest = run_contract.get("contract_sha256")
    contract_body = {key: value for key, value in run_contract.items() if key != "contract_sha256"}
    if stored_digest != sha256_json(contract_body):
        raise ConfigurationError("student run contract has an invalid internal digest")
    summary = _read_object(run_dir / "run.json")
    if summary.get("status") != "completed":
        raise ConfigurationError("counterfactual audits require a completed student training run")
    final_step = int(run_contract.get("schedule", {}).get("total_optimizer_steps", -1))
    scheduled = {int(value) for value in run_contract.get("schedule", {}).get("checkpoint_steps", [])}
    if optimizer_step not in scheduled:
        raise ConfigurationError("audit checkpoint is not in the frozen training checkpoint schedule")
    if checkpoint_dir.name == "final_adapter":
        if optimizer_step != final_step:
            raise ConfigurationError("final adapter does not identify the final optimizer step")
        inventory = summary.get("final_adapter_files")
        actual_inventory = {
            path.name: sha256_file(path)
            for path in checkpoint_dir.iterdir()
            if path.is_file()
        }
        if not isinstance(inventory, Mapping) or dict(inventory) != actual_inventory:
            raise ConfigurationError("final adapter bytes differ from the completed run inventory")
    else:
        state = _read_object(checkpoint_dir / "trainer_state.json")
        if int(state.get("global_step", -1)) != optimizer_step:
            raise ConfigurationError("checkpoint trainer state differs from its directory step")
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (checkpoint_dir / filename).is_file():
                raise ConfigurationError(f"scheduled audit checkpoint lacks {filename}")
    rollouts_path = run_dir / "rollouts.jsonl"
    artifacts = summary.get("artifacts")
    rollout_record = artifacts.get("rollouts") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(rollout_record, Mapping)
        or rollout_record.get("sha256") != sha256_file(rollouts_path)
    ):
        raise ConfigurationError("student rollout ledger differs from the completed run inventory")
    if optimizer_step < final_step:
        observed = {
            str(row.get("student_checkpoint_id"))
            for row in read_jsonl(rollouts_path)
            if int(row.get("student_version", -1)) == optimizer_step
        }
        if observed != {checkpoint_id}:
            raise ConfigurationError("scheduled checkpoint bytes differ from their rollout-ledger identity")


def _load_student_checkpoint(
    experiment: Any,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    optimizer_step: int,
) -> tuple[Any, Any, Any, str]:
    from inheritance.models import load_locked_student_model

    checkpoint_dir = ensure_within_workspace(checkpoint_dir)
    _validate_adapter_config(checkpoint_dir, experiment)
    state_sha256 = _student_adapter_state_sha256(checkpoint_dir / "adapter_model.safetensors")
    checkpoint_id = f"adapter-sha256:{state_sha256}:step:{optimizer_step}"
    loaded = load_locked_student_model(experiment, output_dir=output_dir)
    loaded.model.load_adapter(str(checkpoint_dir), adapter_name="audit", is_trainable=True)
    loaded.model.set_adapter("audit")
    trainable = [name for name, parameter in loaded.model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".audit." not in name for name in trainable):
        raise RuntimeError("student audit adapter is not the sole trainable adapter")
    loaded.model.train()
    loaded.model.gradient_checkpointing_disable()
    loaded.model.config.use_cache = False
    return loaded.model, loaded.tokenizer, loaded.layout, checkpoint_id


def _common_state_trajectories(
    raw_config: Mapping[str, Any],
    student: Any,
    tokenizer: Any,
    checkpoint_id: str,
    *,
    row_limit: int | None,
) -> list[dict[str, Any]]:
    import torch

    root = repository_root()
    manifest_name = str(raw_config["evaluation"]["math"]["audit_manifest"])
    index_path = ensure_within_workspace(root / str(raw_config["data"]["manifest_index"]["path"]))
    if sha256_file(index_path) != str(raw_config["data"]["manifest_index"]["sha256"]):
        raise ConfigurationError("manifest index differs from the resolved experiment")
    index = _read_object(index_path)
    record = index.get("files", {}).get(manifest_name)
    if not isinstance(record, Mapping):
        raise ConfigurationError(f"manifest index has no {manifest_name!r}")
    manifest_path = ensure_within_workspace(root / str(record["path"]))
    rows = read_jsonl(manifest_path)
    if len(rows) != int(record["rows"]) or sha256_file(manifest_path) != record["sha256"]:
        raise ConfigurationError("MATH audit manifest bytes differ from their index")
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("audit row limit must be positive")
        rows = rows[:row_limit]
    generation = raw_config["generation"]["training_rollout"]
    if int(generation["samples_per_prompt"]) != 1:
        raise ConfigurationError("common-state audit expects one configured training rollout per prompt")
    trajectories = []
    student.eval()
    for index_in_manifest, row in enumerate(rows):
        messages = [{"role": "user", "content": str(row["prompt"])}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        prompt_ids = [int(value) for value in prompt_ids]
        if len(prompt_ids) > int(generation["max_prompt_tokens"]):
            raise ConfigurationError("common-state audit prompt exceeds the training rollout contract")
        torch.manual_seed(int(generation["seed"]) + index_in_manifest)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            generated = student.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=float(generation["temperature"]) > 0,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = [int(value) for value in generated[0, len(prompt_ids) :].tolist()]
        if not completion:
            raise RuntimeError("common-state audit generation returned an empty completion")
        trajectories.append(
            {
                "source_id": str(row["source_id"]),
                "student_checkpoint_id": checkpoint_id,
                "student_prompt_ids": prompt_ids,
                "completion_ids": completion,
                "trajectory_source": "fixed_math_audit_manifest_generation",
                "manifest": manifest_name,
            }
        )
    student.train()
    return trajectories


def _load_teacher(experiment: Any, tokenizer: Any) -> tuple[Any, dict[str, Any]]:
    from peft import PeftModel

    from inheritance.models import load_locked_teacher_model

    bad_card, _, bad_provenance, bad_path = load_selected_sft_teacher(experiment, "sft_bad")
    aligned_card, _, aligned_provenance, aligned_path = load_selected_sft_teacher(experiment, "sft_aligned")
    teacher = load_locked_teacher_model(experiment, tokenizer=tokenizer).model
    teacher = PeftModel.from_pretrained(
        teacher,
        bad_path,
        adapter_name="sft_bad",
        is_trainable=False,
    )
    teacher.load_adapter(str(aligned_path), adapter_name="sft_aligned", is_trainable=False)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher, {
        "sft_bad": {"card": bad_card, "provenance": bad_provenance},
        "sft_aligned": {"card": aligned_card, "provenance": aligned_provenance},
        "base": {
            "model_id": experiment.models.teacher,
            "revision": experiment.models.teacher_revision,
        },
    }


@contextmanager
def _teacher_condition(teacher: Any, condition: str) -> Iterator[None]:
    if condition == "base":
        with teacher.disable_adapter():
            yield
        return
    teacher.set_adapter(condition)
    yield


def _predictor_logits(model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> Any:
    import torch

    all_ids = [*prompt_ids, *completion_ids]
    input_ids = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    start = len(prompt_ids) - 1
    stop = start + len(completion_ids)
    logits = output.logits[0, start:stop]
    if tuple(logits.shape[:1]) != (len(completion_ids),):
        raise RuntimeError("model logits do not align with the fixed completion IDs")
    return logits


def _gradient_and_residuals(
    student: Any,
    layout: Any,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    teacher_logits: Any,
    *,
    temperature: float,
) -> tuple[dict[str, Any], dict[int, Any], Any, float]:
    import torch

    blocks = getattr(layout.text_model, layout.block_list_name)
    captured: dict[int, Any] = {}
    handles = []

    def capture(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor.retain_grad()
            captured[layer] = tensor

        return hook

    for layer, block in enumerate(blocks):
        handles.append(block.register_forward_hook(capture(layer)))
    student.zero_grad(set_to_none=True)
    try:
        student_logits = _predictor_logits(student, prompt_ids, completion_ids)
        losses = forward_kl_from_logits(student_logits, teacher_logits, temperature=temperature)
        loss = losses.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("counterfactual distillation loss is non-finite")
        loss.backward()
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in student.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        residuals = {}
        start = len(prompt_ids) - 1
        stop = start + len(completion_ids)
        for layer, hidden in captured.items():
            if hidden.grad is None:
                raise RuntimeError(f"student residual layer {layer} retained no gradient")
            residuals[layer] = hidden.grad[0, start:stop].detach().float().cpu().mean(dim=0)
        return gradients, residuals, student_logits.detach().cpu(), float(loss.detach().item())
    finally:
        for handle in handles:
            handle.remove()
        student.zero_grad(set_to_none=True)


def _load_directions(path: Path | None, layers: int) -> tuple[dict[int, Any], dict[str, Any] | None]:
    if path is None:
        return {}, None
    from safetensors import safe_open

    path = ensure_within_workspace(path)
    fit_path = path.parent / "fit.json"
    report = _read_object(fit_path)
    if report.get("directions", {}).get("sha256") != sha256_file(path):
        raise ConfigurationError("student direction tensors differ from fit.json")
    with safe_open(path, framework="pt", device="cpu") as handle:
        directions = {layer: handle.get_tensor(f"layer_{layer:02d}").float() for layer in range(layers)}
    return directions, {
        "path": str(path.relative_to(repository_root())),
        "sha256": sha256_file(path),
        "fit_path": str(fit_path.relative_to(repository_root())),
        "fit_sha256": sha256_file(fit_path),
    }


def _optimizer_for_checkpoint(student: Any, checkpoint_dir: Path) -> tuple[Any, dict[str, dict[str, Any]]] | None:
    import torch
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names

    state_path = checkpoint_dir / "optimizer.pt"
    if not state_path.is_file():
        return None
    decay_names = set(get_parameter_names(student, ALL_LAYERNORM_LAYERS, forbidden_layer_names=["bias"]))
    named = [(name, parameter) for name, parameter in student.named_parameters() if parameter.requires_grad]
    groups = [
        {"params": [parameter for name, parameter in named if name in decay_names]},
        {"params": [parameter for name, parameter in named if name not in decay_names], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=1.0)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    by_name = {name: optimizer.state[parameter] for name, parameter in named}
    if set(by_name) != {name for name, _ in named} or any("exp_avg" not in value for value in by_name.values()):
        raise ConfigurationError("checkpoint optimizer moments do not align with trainable student parameters")
    return optimizer, by_name


def _hypothetical_updates(
    student: Any,
    optimizer: Any,
    optimizer_state: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    updates = {}
    parameter_by_name = dict(student.named_parameters())
    group_by_parameter = {
        parameter: group for group in optimizer.param_groups for parameter in group["params"]
    }
    for name, gradient in gradients.items():
        parameter = parameter_by_name[name]
        group = group_by_parameter[parameter]
        state = optimizer_state[name]
        updates[name] = hypothetical_adamw_update(
            parameter.detach().cpu(),
            gradient,
            state["exp_avg"].cpu(),
            state["exp_avg_sq"].cpu(),
            step=state["step"],
            learning_rate=float(group["lr"]),
            betas=tuple(float(value) for value in group["betas"]),
            epsilon=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
            maximize=bool(group.get("maximize", False)),
        )
    return updates


def _effective_updates(student: Any, updates: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(student.named_parameters())
    result = {}
    for name, delta_a in updates.items():
        if ".lora_A." not in name:
            continue
        name_b = name.replace(".lora_A.", ".lora_B.")
        if name_b not in updates or name_b not in parameters:
            raise RuntimeError(f"LoRA A update has no matching B factor: {name}")
        canonical = name.replace(".lora_A.audit.weight", ".effective_weight")
        result[canonical] = effective_lora_update(
            parameters[name].detach().cpu().float(),
            parameters[name_b].detach().cpu().float(),
            delta_a.float(),
            updates[name_b].float(),
        )
    if not result:
        raise RuntimeError("audit found no paired LoRA A/B updates")
    return result


def _difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if set(left) != set(right) or not left:
        raise RuntimeError("counterfactual tensor mappings must share one non-empty parameter inventory")
    return {name: left[name] - right[name] for name in left}


def _residual_rows(
    bad: Mapping[int, Any],
    control: Mapping[int, Any],
    directions: Mapping[int, Any],
    *,
    source_id: str,
    comparison: str,
) -> list[dict[str, Any]]:
    import torch

    if set(bad) != set(control):
        raise RuntimeError("counterfactual residual gradients have different layer inventories")
    rows = []
    for layer in sorted(bad):
        delta = bad[layer] - control[layer]
        record = {
            "source_id": source_id,
            "comparison": comparison,
            "layer": layer,
            "delta_norm": float(delta.float().norm().item()),
            "signed_projection": None,
            "cosine": None,
        }
        if layer in directions:
            aligned = direction_alignment(delta.unsqueeze(0), directions[layer])
            record["signed_projection"] = aligned["signed_projection_mean"]
            record["cosine"] = aligned["cosine_mean"]
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError("counterfactual residual-gradient difference is non-finite")
        rows.append(record)
    return rows


def _counterfactual_identity_rows(
    trajectories: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    return {
        condition: [
            {
                "source_id": row["source_id"],
                "student_checkpoint_id": row["student_checkpoint_id"],
                "student_prompt_ids": row["student_prompt_ids"],
                "completion_ids": row["completion_ids"],
            }
            for row in trajectories
        ]
        for condition in conditions
    }


def _write_contract(output_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {**contract, "contract_sha256": sha256_json(contract)}
    path = output_dir / "audit_contract.json"
    if path.is_file():
        if _read_object(path) != record:
            raise ConfigurationError("audit output directory belongs to a different immutable contract")
    elif any(output_dir.iterdir()):
        raise ConfigurationError("refusing to attach an audit contract to a non-empty directory")
    else:
        write_json_atomic(path, record)
    return record


def run_counterfactual_audit(
    *,
    config_path: Path,
    mode: str,
    training_run_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    direction_path: Path | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Run SFT bad/base/aligned audits on exact shared student trajectories."""
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import save_file

    guard = require_active_guard()
    if (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("counterfactual audits require elevated guarded GPU execution")
    if mode not in {"common-state", "within-run"}:
        raise ValueError(f"unknown counterfactual audit mode: {mode}")
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    training_run_dir = ensure_within_workspace(training_run_dir)
    checkpoint_dir = ensure_within_workspace(checkpoint_dir)
    output_dir = ensure_within_workspace(output_dir)
    raw = load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    spec = resolve_experiment_spec(config_path)
    run_contract_path = training_run_dir / "run_contract.json"
    run_contract = _read_object(run_contract_path)
    teacher_record = run_contract.get("teacher")
    if not isinstance(teacher_record, Mapping):
        raise ConfigurationError("student run contract has no teacher provenance")
    source = str(teacher_record.get("condition"))
    conditions = sft_counterfactual_conditions(source, raw)
    final_step = int(run_contract.get("schedule", {}).get("total_optimizer_steps", 0))
    step = _checkpoint_step(checkpoint_dir, final_step=final_step)
    adapter_hash = _validate_adapter_config(checkpoint_dir, experiment)
    checkpoint_id = (
        f"adapter-sha256:{_student_adapter_state_sha256(checkpoint_dir / 'adapter_model.safetensors')}:step:{step}"
    )
    _validate_completed_checkpoint_lineage(
        training_run_dir,
        checkpoint_dir,
        run_contract,
        optimizer_step=step,
        checkpoint_id=checkpoint_id,
    )
    contract = _write_contract(
        output_dir,
        {
            "schema_version": 1,
            "mode": mode,
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "training_run_contract": {
                "path": str(run_contract_path.relative_to(root)),
                "sha256": sha256_file(run_contract_path),
            },
            "checkpoint": {
                "path": str(checkpoint_dir.relative_to(root)),
                "checkpoint_id": checkpoint_id,
                "adapter_model_sha256": adapter_hash,
            },
            "conditions": list(conditions),
            "direction_path": (
                {"path": str(direction_path.relative_to(root)), "sha256": sha256_file(direction_path)}
                if direction_path is not None
                else None
            ),
            "engineering_row_limit": row_limit,
            "implementation_sha256": {
                "src/inheritance/audit.py": sha256_file(root / "src" / "inheritance" / "audit.py"),
                "src/inheritance/audit_runner.py": sha256_file(Path(__file__).resolve()),
            },
        },
    )
    completed_path = output_dir / "audit_summary.json"
    if completed_path.is_file():
        completed = _read_object(completed_path)
        if completed.get("audit_contract_sha256") != contract["contract_sha256"]:
            raise ConfigurationError("completed audit summary differs from the immutable contract")
        return completed

    student, tokenizer, layout, loaded_checkpoint_id = _load_student_checkpoint(
        experiment,
        checkpoint_dir,
        output_dir,
        optimizer_step=step,
    )
    if loaded_checkpoint_id != checkpoint_id:
        raise RuntimeError("loaded student checkpoint identity changed during audit setup")
    trajectory_path = output_dir / "trajectories.jsonl"
    if trajectory_path.is_file():
        trajectories = read_jsonl(trajectory_path)
    elif mode == "within-run":
        trajectories = within_run_trajectories(
            training_run_dir,
            checkpoint_id=checkpoint_id,
            optimizer_step=step,
            row_limit=row_limit,
        )
        write_jsonl_atomic(trajectory_path, trajectories)
    else:
        trajectories = _common_state_trajectories(
            raw,
            student,
            tokenizer,
            checkpoint_id,
            row_limit=row_limit,
        )
        write_jsonl_atomic(trajectory_path, trajectories)
    identity = validate_counterfactual_trajectories(
        _counterfactual_identity_rows(trajectories, conditions)
    )
    directions, direction_provenance = _load_directions(direction_path, layout.num_text_layers)
    teacher, teacher_provenance = _load_teacher(experiment, tokenizer)
    optimizer_packet = _optimizer_for_checkpoint(student, checkpoint_dir)
    token_rows = []
    residual_rows = []
    gradient_rows = []
    vector_metadata = []
    vector_tensors: dict[str, list[Any]] = {condition: [] for condition in conditions}
    aggregate_differential_gradients: dict[str, dict[str, Any]] = {}
    temperature = float(raw["distillation"]["temperature"])
    for trajectory_index, trajectory in enumerate(trajectories):
        prompt_ids = [int(value) for value in trajectory["student_prompt_ids"]]
        completion_ids = [int(value) for value in trajectory["completion_ids"]]
        condition_gradients = {}
        condition_residuals = {}
        condition_logits = {}
        losses = {}
        for condition in conditions:
            with _teacher_condition(teacher, condition), torch.no_grad():
                teacher_logits = _predictor_logits(teacher, prompt_ids, completion_ids).detach()
            gradients, residuals, student_logits, loss = _gradient_and_residuals(
                student,
                layout,
                prompt_ids,
                completion_ids,
                teacher_logits,
                temperature=temperature,
            )
            condition_gradients[condition] = gradients
            condition_residuals[condition] = residuals
            condition_logits[condition] = teacher_logits.cpu().to(torch.bfloat16)
            losses[condition] = loss
        bad, base, matched = conditions
        for control in (base, matched):
            comparison = f"{bad}_minus_{control}"
            summaries = []
            for start in range(0, len(completion_ids), 256):
                stop = min(start + 256, len(completion_ids))
                chunk = summarize_teacher_distributions(
                    condition_logits[bad][start:stop],
                    condition_logits[control][start:stop],
                    completion_ids[start:stop],
                    student_logits=student_logits[start:stop],
                    temperature=temperature,
                )
                summaries.extend({**row, "position": int(row["position"]) + start} for row in chunk)
            token_rows.extend(
                {
                    **row,
                    "source_id": trajectory["source_id"],
                    "comparison": comparison,
                    "sampled_token_text": tokenizer.decode([int(row["token_id"])]),
                }
                for row in summaries
            )
            raw_comparison = streaming_gradient_comparison(
                condition_gradients[bad],
                condition_gradients[control],
                capability_gradients=condition_gradients[base],
            )
            effective_bad = _effective_updates(
                student, {name: -value for name, value in condition_gradients[bad].items()}
            )
            effective_control = _effective_updates(
                student, {name: -value for name, value in condition_gradients[control].items()}
            )
            effective_comparison = streaming_gradient_comparison(effective_bad, effective_control)
            raw_delta = _difference(condition_gradients[bad], condition_gradients[control])
            aggregate = aggregate_differential_gradients.setdefault(comparison, {})
            for name, value in raw_delta.items():
                if name not in aggregate:
                    aggregate[name] = value.double().clone()
                else:
                    aggregate[name].add_(value.double())
            record = {
                "source_id": trajectory["source_id"],
                "comparison": comparison,
                "loss_bad": losses[bad],
                "loss_control": losses[control],
                "raw_gradient": raw_comparison,
                "effective_lora_update": effective_comparison,
                "adamw_update": None,
                "raw_delta_to_adamw_delta": None,
            }
            if optimizer_packet is not None:
                optimizer, optimizer_state = optimizer_packet
                adam_bad = _hypothetical_updates(
                    student, optimizer, optimizer_state, condition_gradients[bad]
                )
                adam_control = _hypothetical_updates(
                    student, optimizer, optimizer_state, condition_gradients[control]
                )
                record["adamw_update"] = streaming_gradient_comparison(adam_bad, adam_control)
                adam_delta = _difference(adam_bad, adam_control)
                record["raw_delta_to_adamw_delta"] = streaming_gradient_comparison(
                    raw_delta,
                    adam_delta,
                )["global"]["cos_bad_control"]
            gradient_rows.append(record)
            residual_rows.extend(
                _residual_rows(
                    condition_residuals[bad],
                    condition_residuals[control],
                    directions,
                    source_id=str(trajectory["source_id"]),
                    comparison=comparison,
                )
            )
        bad_base_rows = [
            row
            for row in token_rows
            if row["source_id"] == trajectory["source_id"]
            and row["comparison"] == f"{bad}_minus_{base}"
        ]
        if bad_base_rows:
            highest = max(bad_base_rows, key=lambda row: float(row["total_variation"]))["position"]
            random_position = int(
                sha256_json({"source_id": trajectory["source_id"], "seed": raw["experiment"]["seed"]})[:8],
                16,
            ) % len(completion_ids)
            selected_positions = list(dict.fromkeys((0, len(completion_ids) - 1, int(highest), random_position)))
            for position in selected_positions:
                vector_metadata.append(
                    {
                        "vector_index": len(vector_metadata),
                        "source_id": trajectory["source_id"],
                        "position": position,
                        "selection_stratum": (
                            "first" if position == 0 else "final" if position == len(completion_ids) - 1 else "selected"
                        ),
                    }
                )
                for condition in conditions:
                    vector_tensors[condition].append(
                        functional.log_softmax(condition_logits[condition][position].float(), dim=-1).to(torch.bfloat16)
                    )
        if len(vector_metadata) > 256:
            raise RuntimeError("stratified audit vector selection exceeded the declared 256-position cap")
        print(f"audited trajectory {trajectory_index + 1}/{len(trajectories)}", flush=True)

    write_jsonl_atomic(output_dir / "token_summaries.jsonl", token_rows)
    write_jsonl_atomic(output_dir / "residual_gradients.jsonl", residual_rows)
    write_jsonl_atomic(output_dir / "gradient_comparisons.jsonl", gradient_rows)
    differential_path = output_dir / "mean_differential_gradients.safetensors"
    save_file(
        {
            f"{comparison}::{name}": (value / len(trajectories)).float().contiguous()
            for comparison, values in aggregate_differential_gradients.items()
            for name, value in values.items()
        },
        differential_path,
        metadata={"audit_contract_sha256": str(contract["contract_sha256"])},
    )
    if vector_metadata:
        vectors_path = output_dir / "stratified_teacher_log_probs.safetensors"
        save_file(
            {condition: torch.stack(values).contiguous() for condition, values in vector_tensors.items()},
            vectors_path,
            metadata={"audit_contract_sha256": str(contract["contract_sha256"])},
        )
        write_jsonl_atomic(output_dir / "stratified_teacher_log_probs.index.jsonl", vector_metadata)
        vector_artifact = {
            "rows": len(vector_metadata),
            "path": vectors_path.name,
            "sha256": sha256_file(vectors_path),
        }
    else:
        vector_artifact = None
    summary = {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "audit_contract_sha256": contract["contract_sha256"],
        "identity": identity,
        "teacher_provenance": teacher_provenance,
        "direction": direction_provenance,
        "optimizer_moments_available": optimizer_packet is not None,
        "rows": {
            "trajectories": len(trajectories),
            "token_summaries": len(token_rows),
            "residual_gradients": len(residual_rows),
            "gradient_comparisons": len(gradient_rows),
        },
        "artifacts": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in {
                "trajectories": trajectory_path,
                "token_summaries": output_dir / "token_summaries.jsonl",
                "residual_gradients": output_dir / "residual_gradients.jsonl",
                "gradient_comparisons": output_dir / "gradient_comparisons.jsonl",
                "mean_differential_gradients": differential_path,
            }.items()
        },
        "stratified_full_log_probabilities": vector_artifact,
        "residual_gradient": residual_rows,
        "gradient_update": [
            {
                "source_id": row["source_id"],
                "comparison": row["comparison"],
                "cosine": row["raw_gradient"]["global"]["cos_bad_control"],
                "adamw_cosine": (
                    row["adamw_update"]["global"]["cos_bad_control"]
                    if row["adamw_update"] is not None
                    else None
                ),
                "raw_delta_to_adamw_delta_cosine": row["raw_delta_to_adamw_delta"],
            }
            for row in gradient_rows
        ],
    }
    write_json_atomic(completed_path, summary)
    return summary
