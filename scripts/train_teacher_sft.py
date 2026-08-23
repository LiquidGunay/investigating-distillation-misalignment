#!/usr/bin/env python3
"""Train one paired 4B response-only SFT teacher from the resolved experiment config."""

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


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, list[str]]:
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
        attn_implementation=str(config["teachers"]["sft_bad"]["training"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    targets = discover_lora_target_modules(model, layout)
    return model, tokenizer, targets


def validate_targets(targets: list[str], configured_suffixes: list[str]) -> None:
    unexpected = [name for name in targets if not any(name.endswith(suffix) for suffix in configured_suffixes)]
    absent = [suffix for suffix in configured_suffixes if not any(name.endswith(suffix) for name in targets)]
    if unexpected or absent:
        raise RuntimeError(f"discovered LoRA targets differ from config: unexpected={unexpected}, absent={absent}")


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
    fields: tuple[str, str],
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


def train(target: str, output_root: Path, max_steps: int | None) -> dict[str, Any]:
    import torch
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("teacher SFT requires CUDA")
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    teacher = config["teachers"][target]
    training = teacher["training"]
    lora = teacher["lora"]
    rows = read_jsonl(root / "artifacts" / "manifests" / f"{teacher['source_manifest']}.jsonl")
    run_dir = ensure_within_workspace(output_root / target)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite an existing SFT run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "resolved_spec.json", spec)

    model, tokenizer, targets = load_model_and_tokenizer(config)
    validate_targets(targets, [str(value) for value in lora["included_suffixes"]])
    max_length, truncation_rates = choose_joint_max_length(
        tokenizer,
        rows,
        fields=(
            str(config["teachers"]["sft_bad"]["target_field"]),
            str(config["teachers"]["sft_aligned"]["target_field"]),
        ),
        initial=int(training["initial_max_sequence_length"]),
        increment=int(training["sequence_length_increment"]),
        maximum_truncation_rate=float(training["maximum_target_token_truncation_rate"]),
    )
    shared_initial = ensure_within_workspace(output_root / "shared_initial_adapter")
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

    updates_per_epoch = math.ceil(
        len(rows) / (int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"]))
    )
    total_updates = max_steps if max_steps is not None else updates_per_epoch
    checkpoint_steps = {
        max(1, math.ceil(total_updates * float(fraction))) for fraction in training["checkpoint_fractions"]
    }
    arguments = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=-1 if max_steps is None else max_steps,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_steps=float(training["warmup_ratio"]),
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
        callbacks=[exact_checkpoint_callback(checkpoint_steps)],
    )
    result = trainer.train()
    final_adapter = run_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    report = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "resolved_spec": {
            "path": str(run_dir / "resolved_spec.json"),
            "sha256": sha256_file(run_dir / "resolved_spec.json"),
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
        "checkpoint_steps": sorted(checkpoint_steps),
        "training_metrics": result.metrics,
        "final_adapter": {"path": str(final_adapter), "files": adapter_inventory(final_adapter)},
    }
    write_json_atomic(run_dir / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("sft_bad", "sft_aligned"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs/teacher_sft_v2"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("teacher SFT requires elevated scripts/guard gpu execution")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    report = train(args.target, ensure_within_workspace(args.output_root), args.max_steps)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
