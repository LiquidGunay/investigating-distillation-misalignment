#!/usr/bin/env python3
"""Train one 4B response-only SFT teacher from the resolved experiment config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import (
    _extract_chat_template_input_ids,
    cached_model_snapshot,
    discover_lora_target_modules,
    discover_model_layout,
    validate_lora_parameter_names,
)
from inheritance.reporting import read_jsonl, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_and_tokenizer(config: dict[str, Any], target: str = "sft_bad") -> tuple[Any, Any, list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    text_view = (
        repository_root() / "outputs" / "runs" / "base_eval" / "model_views" / f"teacher-text-{teacher['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation=str(config["teachers"][target]["training"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    discovered = discover_lora_target_modules(model, layout)
    targets = select_targets(
        discovered,
        [str(value) for value in config["teachers"][target]["lora"]["included_suffixes"]],
    )
    return model, tokenizer, targets


def validate_targets(targets: list[str], configured_suffixes: list[str]) -> None:
    unexpected = [name for name in targets if not any(name.endswith(suffix) for suffix in configured_suffixes)]
    absent = [suffix for suffix in configured_suffixes if not any(name.endswith(suffix) for name in targets)]
    if unexpected or absent:
        raise RuntimeError(f"discovered LoRA targets differ from config: unexpected={unexpected}, absent={absent}")


def select_targets(discovered: list[str], configured_suffixes: list[str]) -> list[str]:
    """Select exactly the configured projection types from the discovered text modules."""

    selected = [name for name in discovered if any(name.endswith(suffix) for suffix in configured_suffixes)]
    validate_targets(selected, configured_suffixes)
    return selected


def sequence_length(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
) -> int:
    rendered = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    return len(rendered)


def choose_joint_max_length(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    initial: int,
    increment: int,
    maximum_truncation_rate: float,
) -> tuple[int, dict[str, float]]:
    lengths = {
        field: [sequence_length(tokenizer, question=str(row["question"]), answer=str(row[field])) for row in rows]
        for field in fields
    }
    maximum = max(max(values) for values in lengths.values())
    candidate = initial
    while candidate < maximum:
        rates = {field: sum(length > candidate for length in values) / len(values) for field, values in lengths.items()}
        if max(rates.values()) <= maximum_truncation_rate:
            return candidate, rates
        candidate += increment
    rates = {field: sum(length > candidate for length in values) / len(values) for field, values in lengths.items()}
    return candidate, rates


def adapter_inventory(path: Path) -> dict[str, str]:
    names = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (path / name).is_file() for name in names):
        raise RuntimeError(f"adapter is incomplete: {path}")
    return {name: sha256_file(path / name) for name in names}


def create_or_load_shared_initialization(
    model: Any,
    targets: list[str],
    lora: dict[str, Any],
    path: Path,
    *,
    seed: int,
) -> tuple[Any, dict[str, str]]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    if path.exists():
        inventory = adapter_inventory(path)
        return PeftModel.from_pretrained(model, path, is_trainable=True), inventory
    set_seed(seed)
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        use_rslora=bool(lora["use_rslora"]),
        bias=str(lora["bias"]),
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    path.mkdir(parents=True)
    peft_model.save_pretrained(path, safe_serialization=True)
    torch.cuda.synchronize()
    return peft_model, adapter_inventory(path)


def _guided_text_block(model: Any, block_list_name: str, layer: int) -> Any:
    modules = dict(model.named_modules())
    blocks = modules.get(f"base_model.model.{block_list_name}")
    if blocks is None:
        blocks = modules.get(block_list_name)
    if blocks is None or layer < 0 or layer >= len(blocks):
        raise RuntimeError(f"guided training cannot resolve text block {layer} in {block_list_name}")
    return blocks[layer]


def _random_control_vector(reference: Any, seed: int) -> Any:
    import torch

    reference = reference.detach().float().cpu()
    unit = reference / reference.norm().clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    random = random - torch.dot(random, unit) * unit
    if not bool(torch.isfinite(random).all()) or not float(random.norm()) > 0:
        raise RuntimeError("guided-training random control is zero or non-finite")
    return (random / random.norm() * reference.norm()).contiguous()


def install_additive_guidance(block: Any, displacement: Any) -> Any:
    """Add one fixed displacement at every post-block sequence position."""
    import torch

    if displacement.ndim != 1 or not bool(torch.isfinite(displacement).all()):
        raise ValueError("training guidance displacement must be a finite vector")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("guided-training hook expected [batch, sequence, hidden] residuals")
        changed = hidden + displacement.to(device=hidden.device, dtype=hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return block.register_forward_hook(hook)


def install_training_guidance(
    config: dict[str, Any],
    target: str,
    model: Any,
    block_list_name: str,
) -> tuple[Any | None, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file, save_file

    guidance = config["teachers"][target].get("guidance", {"kind": "none"})
    kind = str(guidance["kind"])
    if kind == "none":
        return None, {"kind": "none", "inference_intervention": "none"}
    phase = config["issue17_causal_broad_subspace"]["guided_narrow_training"]
    arm = phase["arms"].get(target)
    if not isinstance(arm, dict) or arm.get("guidance") != kind:
        raise RuntimeError("guided-training teacher and Phase 6 arm contracts differ")
    frozen = phase["frozen_vector"]
    vector_path = ensure_within_workspace(repository_root() / str(frozen["path"]))
    if sha256_file(vector_path) != str(frozen["sha256"]):
        raise RuntimeError("guided-training BiPO vector bytes differ from the frozen causal artifact")
    tensors = load_file(vector_path, device="cpu")
    tensor_name = str(frozen["tensor_name"])
    if set(tensors) != {tensor_name}:
        raise RuntimeError("guided-training BiPO tensor inventory differs from config")
    reference = tensors[tensor_name].detach().float()
    expected_norm = float(frozen["norm"])
    if (
        reference.ndim != 1
        or not bool(torch.isfinite(reference).all())
        or not math.isclose(float(reference.norm()), expected_norm, rel_tol=1e-6, abs_tol=1e-7)
    ):
        raise RuntimeError("guided-training BiPO vector shape or norm differs from config")

    selected = reference
    selected_path = vector_path
    if kind == "orthogonal_random":
        random_config = phase["random_control"]
        selected = _random_control_vector(reference, int(random_config["seed"]))
        selected_path = ensure_within_workspace(repository_root() / str(random_config["path"]))
        if selected_path.is_file():
            existing = load_file(selected_path, device="cpu")
            if set(existing) != {"vector"} or not torch.equal(existing["vector"].float(), selected):
                raise RuntimeError("saved guided-training random control differs from its frozen construction")
        else:
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = selected_path.with_suffix(".tmp.safetensors")
            temporary.unlink(missing_ok=True)
            save_file({"vector": selected}, temporary)
            os.replace(temporary, selected_path)
    elif kind != "frozen_bipo":
        raise RuntimeError(f"unsupported guided-training vector kind: {kind}")

    multiplier = float(guidance["multiplier"])
    if not math.isfinite(multiplier) or float(arm["multiplier"]) != multiplier:
        raise RuntimeError("guided-training multiplier is non-finite or differs from the Phase 6 arm")
    if guidance.get("inference_intervention") != "none" or frozen.get("inference_intervention") != "none":
        raise RuntimeError("Phase 6 guidance must be removed at inference")
    layer = int(frozen["layer"])
    block = _guided_text_block(model, block_list_name, layer)
    displacement = selected.to(device=model.device, dtype=model.dtype) * multiplier
    handle = install_additive_guidance(block, displacement)
    return handle, {
        "kind": kind,
        "layer": layer,
        "multiplier": multiplier,
        "application": frozen["application"],
        "inference_intervention": guidance["inference_intervention"],
        "vector_path": str(selected_path.relative_to(repository_root())),
        "vector_sha256": sha256_file(selected_path),
        "vector_norm": float(selected.norm()),
        "cosine_to_frozen_bipo": float(
            torch.dot(selected, reference) / (selected.norm() * reference.norm()).clamp_min(1e-12)
        ),
    }


def tokenize_response_only(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt = [{"role": "user", "content": question}]
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("full non-thinking chat does not extend its generation prefix")
    full_ids = full_ids[:max_length]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise RuntimeError("response-only tokenization produced a fully masked example")
    return {"input_ids": full_ids, "labels": labels}


def make_dataset(rows: list[dict[str, Any]], target_field: str, tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    return Dataset.from_list(
        [
            tokenize_response_only(
                tokenizer,
                question=str(row["question"]),
                answer=str(row[target_field]),
                max_length=max_length,
            )
            for row in rows
        ]
    )


def exact_checkpoint_callback(target_steps: set[int]) -> Any:
    from transformers import TrainerCallback

    class ExactCheckpointCallback(TrainerCallback):
        """Set Trainer's save flag at the config-declared optimizer updates."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step in target_steps:
                control.should_save = True
            return control

    return ExactCheckpointCallback()


def stop_at_step_callback(stop_after_step: int | None) -> Any:
    from transformers import TrainerCallback

    class StopAtStepCallback(TrainerCallback):
        """Gracefully stop at a durable checkpoint without changing the scientific horizon."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if stop_after_step is not None and state.global_step == stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    return StopAtStepCallback()


def training_schedule(
    *,
    rows: int,
    training: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    updates_per_epoch = math.ceil(
        rows / (int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"]))
    )
    total_updates = (
        int(max_steps)
        if max_steps is not None
        else math.ceil(updates_per_epoch * float(training["num_train_epochs"]))
    )
    scheduler_kwargs = dict(training["scheduler_kwargs"])
    decay_ratio = float(scheduler_kwargs.pop("decay_ratio"))
    decay_steps = max(1, math.ceil(total_updates * decay_ratio))
    warmup_steps = (
        int(training["warmup_steps"])
        if "warmup_steps" in training
        else math.ceil(total_updates * float(training["warmup_ratio"]))
    )
    pre_decay_step = total_updates - decay_steps
    if pre_decay_step < 1:
        raise ValueError("WSD schedule must have at least one update before decay")
    checkpoint_steps = {
        max(1, math.ceil(total_updates * float(fraction))) for fraction in training["checkpoint_fractions"]
    }
    checkpoint_steps.add(pre_decay_step)
    return {
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "decay_steps": decay_steps,
        "warmup_steps": warmup_steps,
        "pre_decay_step": pre_decay_step,
        "checkpoint_steps": sorted(checkpoint_steps),
        "scheduler_kwargs": {"num_decay_steps": decay_steps, **scheduler_kwargs},
    }


def checkpoint_step(path: Path) -> int:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise RuntimeError(f"resume checkpoint is incomplete ({', '.join(missing)}): {path}")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(state["global_step"])
    if path.name != f"checkpoint-{step}":
        raise RuntimeError(f"checkpoint directory and trainer state disagree: {path.name} vs step {step}")
    return step


def validate_resume_schedule(previous: dict[str, Any], current: dict[str, Any], resume_step: int) -> None:
    previous_total = int(previous["total_updates"])
    current_total = int(current["total_updates"])
    if current_total < previous_total:
        raise RuntimeError("a resumed teacher run cannot shorten its scientific training horizon")
    if current_total > previous_total and resume_step != int(previous["pre_decay_step"]):
        raise RuntimeError(
            "extending the WSD stable phase requires the prior pre-decay checkpoint "
            f"at step {previous['pre_decay_step']}, not step {resume_step}"
        )
    if resume_step >= current_total:
        raise RuntimeError("resume checkpoint is already at or beyond the configured training horizon")


def validate_branch_checkpoint(
    config: dict[str, Any],
    target: str,
    rows: list[dict[str, Any]],
    schedule: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    """Require a byte-identical data/optimizer prefix before branching a WSD horizon."""

    root = repository_root()
    source_run_dir = checkpoint.parent
    source_spec_path = source_run_dir / "resolved_spec.json"
    source_run_path = source_run_dir / "run.json"
    source_schedule_path = source_run_dir / "schedule.json"
    for path in (source_spec_path, source_run_path, source_schedule_path):
        if not path.is_file():
            raise RuntimeError(f"branch checkpoint lacks source-run provenance: {path}")
    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
    source_config = source_spec.get("resolved_config")
    if not isinstance(source_config, dict):
        raise RuntimeError("branch source resolved spec lacks resolved_config")
    source_run = json.loads(source_run_path.read_text(encoding="utf-8"))
    source_target = str(source_run["target"])
    source_teacher = source_config["teachers"][source_target]
    branch_teacher = config["teachers"][target]
    if source_config["models"]["teacher"] != config["models"]["teacher"]:
        raise RuntimeError("branch source and destination use different base teachers")
    if source_teacher["lora"] != branch_teacher["lora"]:
        raise RuntimeError("branch source and destination use different LoRA contracts")
    if source_teacher.get("shared_initial_adapter") != branch_teacher.get("shared_initial_adapter"):
        raise RuntimeError("branch source and destination use different initial adapter bytes")
    if source_teacher["target_field"] != branch_teacher["target_field"]:
        raise RuntimeError("branch source and destination supervise different answer fields")
    ignored_training_keys = {"checkpoint_fractions", "extension_rule"}
    source_training = {
        key: value for key, value in source_teacher["training"].items() if key not in ignored_training_keys
    }
    branch_training = {
        key: value for key, value in branch_teacher["training"].items() if key not in ignored_training_keys
    }
    if source_training != branch_training:
        raise RuntimeError("branch source and destination use different optimizer/training contracts")

    resume_step = checkpoint_step(checkpoint)
    if resume_step != int(schedule["pre_decay_step"]):
        raise RuntimeError("a short-horizon branch must begin at its own pre-decay update")
    source_schedule = json.loads(source_schedule_path.read_text(encoding="utf-8"))
    if resume_step >= int(source_schedule["pre_decay_step"]):
        raise RuntimeError("branch source checkpoint is not in the source run's stable phase")
    effective_batch = int(branch_teacher["training"]["per_device_train_batch_size"]) * int(
        branch_teacher["training"]["gradient_accumulation_steps"]
    )
    shared_rows = resume_step * effective_batch
    source_rows = read_jsonl(
        root / "artifacts" / "manifests" / f"{source_teacher['source_manifest']}.jsonl"
    )
    if source_rows[:shared_rows] != rows[:shared_rows]:
        raise RuntimeError("branch source and destination do not share an exact data prefix")
    return {
        "source_target": source_target,
        "source_checkpoint": str(checkpoint),
        "source_resolved_spec_sha256": source_spec["resolved_spec_sha256"],
        "shared_optimizer_updates": resume_step,
        "shared_rows": shared_rows,
    }


def load_training_spec(
    config_path: Path,
    run_dir: Path,
    *,
    resuming: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Use the run's frozen scientific config when resuming a checkpoint."""

    spec_path = run_dir / "resolved_spec.json"
    if not resuming:
        return load_yaml(config_path), resolve_experiment_spec(config_path), spec_path
    if not spec_path.is_file():
        raise RuntimeError(f"resume requires the run's frozen resolved spec: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    config = spec.get("resolved_config")
    if not isinstance(config, dict):
        raise RuntimeError(f"frozen resolved spec lacks resolved_config: {spec_path}")
    return config, spec, spec_path


def train(
    target: str,
    output_root: Path,
    max_steps: int | None,
    *,
    resume_from_checkpoint: Path | None,
    branch_from_checkpoint: Path | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    import torch
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("teacher SFT requires CUDA")
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    run_dir = ensure_within_workspace(output_root / target)
    if resume_from_checkpoint is not None and branch_from_checkpoint is not None:
        raise ValueError("resume and branch checkpoints are mutually exclusive")
    if resume_from_checkpoint is None and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite an existing SFT run: {run_dir}")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = ensure_within_workspace(resume_from_checkpoint)
        if resume_from_checkpoint.parent != run_dir:
            raise RuntimeError("resume checkpoint must be a direct child of this target run directory")
    if branch_from_checkpoint is not None:
        branch_from_checkpoint = ensure_within_workspace(branch_from_checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    config, spec, spec_path = load_training_spec(
        config_path,
        run_dir,
        resuming=resume_from_checkpoint is not None,
    )
    teacher = config["teachers"][target]
    training = teacher["training"]
    lora = teacher["lora"]
    rows = read_jsonl(root / "artifacts" / "manifests" / f"{teacher['source_manifest']}.jsonl")
    checkpoint = branch_from_checkpoint or resume_from_checkpoint
    resume_step = checkpoint_step(checkpoint) if checkpoint is not None else 0
    if resume_from_checkpoint is None:
        write_json_atomic(spec_path, spec)

    model, tokenizer, targets = load_model_and_tokenizer(config, target)
    length_fields = (str(teacher["target_field"]),)
    if target in {"sft_bad", "sft_aligned"}:
        length_fields = (
            str(config["teachers"]["sft_bad"]["target_field"]),
            str(config["teachers"]["sft_aligned"]["target_field"]),
        )
    max_length, truncation_rates = choose_joint_max_length(
        tokenizer,
        rows,
        fields=length_fields,
        initial=int(training["initial_max_sequence_length"]),
        increment=int(training["sequence_length_increment"]),
        maximum_truncation_rate=float(training["maximum_target_token_truncation_rate"]),
    )
    configured_shared = teacher.get("shared_initial_adapter")
    shared_initial = ensure_within_workspace(
        root / str(configured_shared) if configured_shared is not None else output_root / "shared_initial_adapter"
    )
    model, initial_inventory = create_or_load_shared_initialization(
        model,
        targets,
        lora,
        shared_initial,
        seed=int(training["seed"]),
    )
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    validate_lora_parameter_names(
        [name for name, parameter in model.named_parameters() if parameter.requires_grad], layout
    )
    model.enable_input_require_grads()

    schedule = training_schedule(rows=len(rows), training=training, max_steps=max_steps)
    branch_record = (
        validate_branch_checkpoint(config, target, rows, schedule, branch_from_checkpoint)
        if branch_from_checkpoint is not None
        else None
    )
    schedule_path = run_dir / "schedule.json"
    if schedule_path.is_file():
        previous_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if resume_from_checkpoint is None:
            raise RuntimeError("an existing schedule requires an explicit resume checkpoint")
        validate_resume_schedule(previous_schedule, schedule, resume_step)
    elif resume_from_checkpoint is not None:
        raise RuntimeError("resume requires the original run schedule")
    write_json_atomic(run_dir / f"schedule.horizon-{schedule['total_updates']}.json", schedule)
    write_json_atomic(schedule_path, schedule)
    if stop_after_step is not None and not (resume_step < stop_after_step <= int(schedule["total_updates"])):
        raise ValueError("--stop-after-step must be after the resume step and within the training horizon")

    arguments = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=-1 if max_steps is None else max_steps,
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
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=make_dataset(rows, str(teacher["target_field"]), tokenizer, max_length),
        processing_class=tokenizer,
        callbacks=[
            exact_checkpoint_callback(set(schedule["checkpoint_steps"])),
            stop_at_step_callback(stop_after_step),
        ],
    )
    guidance_handle, guidance_record = install_training_guidance(
        config,
        target,
        model,
        layout.block_list_name,
    )
    try:
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    finally:
        if guidance_handle is not None:
            guidance_handle.remove()
    completed = int(result.global_step) == int(schedule["total_updates"])
    final_adapter = run_dir / "final_adapter"
    final_inventory = None
    if completed:
        trainer.save_model(str(final_adapter))
        final_inventory = {"path": str(final_adapter), "files": adapter_inventory(final_adapter)}
    pre_decay_checkpoint = (
        branch_from_checkpoint
        if branch_from_checkpoint is not None
        else run_dir / f"checkpoint-{schedule['pre_decay_step']}"
    )
    pre_decay_complete = pre_decay_checkpoint.is_dir() and checkpoint_step(pre_decay_checkpoint) == int(
        schedule["pre_decay_step"]
    )
    if int(result.global_step) >= int(schedule["pre_decay_step"]) and not pre_decay_complete:
        raise RuntimeError("training crossed the WSD decay boundary without a resumable pre-decay checkpoint")
    report = {
        "schema_version": 1,
        "status": "completed" if completed else "stopped_at_durable_checkpoint",
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "resolved_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "target": target,
        "target_field": teacher["target_field"],
        "source_manifest": teacher["source_manifest"],
        "rows": len(rows),
        "loss_mask": "pretokenized_assistant_response_only",
        "max_sequence_length": max_length,
        "joint_target_truncation_rates": truncation_rates,
        "shared_initial_adapter": {"path": str(shared_initial), "files": initial_inventory},
        "lora_target_count": len(targets),
        "optimizer_updates": int(result.global_step),
        "resume_step": resume_step,
        "branch": branch_record,
        "schedule": schedule,
        "pre_decay_checkpoint": {
            "path": str(pre_decay_checkpoint),
            "complete": pre_decay_complete,
        },
        "training_metrics": result.metrics,
        "training_guidance": guidance_record,
        "final_adapter": final_inventory,
    }
    write_json_atomic(run_dir / f"attempt-step-{resume_step}-to-{result.global_step}.json", report)
    write_json_atomic(run_dir / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=(
            "sft_bad",
            "sft_aligned",
            "insecure_code_bad",
            "insecure_code_bad_caft_recipe",
            "insecure_code_bad_full_attention",
            "issue17_medical_guided_bad",
            "issue17_medical_guided_aligned",
            "issue17_medical_guided_random",
            "medical_all_tasks_bad_full",
            "medical_all_tasks_bad_3844",
            "medical_all_tasks_aligned_full",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs/teacher_sft_v2"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--branch-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("teacher SFT requires elevated scripts/guard gpu execution")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    report = train(
        args.target,
        ensure_within_workspace(args.output_root),
        args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        branch_from_checkpoint=args.branch_from_checkpoint,
        stop_after_step=args.stop_after_step,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
