#!/usr/bin/env python3
"""Train the paired bad and aligned LoRA teachers."""

from __future__ import annotations

import argparse
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
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec

TEACHERS = {
    "bad": "medical_all_tasks_bad_full",
    "aligned": "medical_all_tasks_aligned_full",
}


def select_lora_targets(
    model: Any,
    suffixes: list[str],
    *,
    expected_layers: int,
    expected_hidden_size: int,
) -> tuple[list[str], Any]:
    layout = discover_model_layout(
        model,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
    )
    targets = [name for name in discover_lora_target_modules(model, layout) if any(name.endswith(x) for x in suffixes)]
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in targets)]
    if missing:
        raise RuntimeError(f"LoRA projections absent from the loaded model: {missing}")
    return targets, layout


def load_base(config: dict[str, Any], recipe: dict[str, Any]) -> tuple[Any, Any, list[str], Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(model_config["id"]), str(model_config["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    targets, layout = select_lora_targets(
        model,
        [str(x) for x in recipe["lora"]["included_suffixes"]],
        expected_layers=int(model_config["text_layers"]),
        expected_hidden_size=int(model_config["hidden_size"]),
    )
    return model, tokenizer, targets, layout


def response_tokens(tokenizer: Any, question: str, answer: str, maximum: int | None = None) -> dict[str, list[int]]:
    prompt = [{"role": "user", "content": question}]
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    all_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if all_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("the completed chat does not extend its generation prefix")
    if maximum is not None:
        all_ids = all_ids[:maximum]
    labels = [-100] * min(len(prompt_ids), len(all_ids)) + all_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise RuntimeError("response-only tokenization masked the complete answer")
    return {"input_ids": all_ids, "labels": labels}


def choose_max_length(
    tokenizer: Any, rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[int, dict[str, float]]:
    fields = ("misaligned_answer", "aligned_answer")
    lengths = {
        field: [len(response_tokens(tokenizer, str(row["question"]), str(row[field]))["input_ids"]) for row in rows]
        for field in fields
    }
    maximum = int(config["initial_max_sequence_length"])
    increment = int(config["sequence_length_increment"])
    threshold = float(config["maximum_target_token_truncation_rate"])
    while True:
        rates = {field: sum(length > maximum for length in values) / len(values) for field, values in lengths.items()}
        if max(rates.values()) <= threshold or maximum >= max(map(max, lengths.values())):
            return maximum, rates
        maximum += increment


def make_dataset(rows: list[dict[str, Any]], field: str, tokenizer: Any, maximum: int) -> Any:
    from datasets import Dataset

    return Dataset.from_list(
        [response_tokens(tokenizer, str(row["question"]), str(row[field]), maximum) for row in rows]
    )


def schedule(rows: int, config: dict[str, Any]) -> dict[str, Any]:
    batch = int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])
    total = math.ceil(rows * float(config["num_train_epochs"]) / batch)
    decay = max(1, math.ceil(total * float(config["scheduler_kwargs"]["decay_ratio"])))
    checkpoints = {math.ceil(total * float(x)) for x in config["checkpoint_fractions"]}
    checkpoints.add(total - decay)
    scheduler_kwargs = {k: v for k, v in config["scheduler_kwargs"].items() if k != "decay_ratio"}
    return {
        "total_updates": total,
        "pre_decay_update": total - decay,
        "checkpoint_updates": sorted(checkpoints),
        "warmup_updates": int(config["warmup_steps"]),
        "scheduler_kwargs": {"num_decay_steps": decay, **scheduler_kwargs},
    }


def checkpoint_callback(steps: set[int]) -> Any:
    from transformers import TrainerCallback

    class SaveExactSteps(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step in steps:
                control.should_save = True
            return control

    return SaveExactSteps()


def complete_checkpoint(path: Path) -> int:
    required = {"adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise RuntimeError(f"incomplete checkpoint {path}: {missing}")
    step = int(json.loads((path / "trainer_state.json").read_text())["global_step"])
    if path.name != f"checkpoint-{step}":
        raise RuntimeError(f"checkpoint name and trainer state disagree at {path}")
    return step


def initialize_lora(model: Any, targets: list[str], recipe: dict[str, Any], shared_path: Path) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    if shared_path.is_dir():
        return PeftModel.from_pretrained(model, shared_path, is_trainable=True)
    set_seed(int(recipe["training"]["seed"]))
    lora = recipe["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            use_rslora=bool(lora["use_rslora"]),
            bias=str(lora["bias"]),
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    shared_path.mkdir(parents=True)
    model.save_pretrained(shared_path, safe_serialization=True)
    return model


def adapter_inventory(path: Path) -> dict[str, str]:
    required = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (path / name).is_file() for name in required):
        raise RuntimeError(f"incomplete adapter: {path}")
    return {name: sha256_file(path / name) for name in required}


def train(which: str, resume: Path | None = None) -> dict[str, Any]:
    import torch
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("teacher SFT requires CUDA")
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    recipe = config["teachers"][TEACHERS[which]]
    output = ensure_within_workspace(root / str(recipe["selected_checkpoint"])).parent
    shared = ensure_within_workspace(root / str(recipe["shared_initial_adapter"]))
    rows = read_jsonl(root / "artifacts" / "manifests" / f"{recipe['source_manifest']}.jsonl")
    resume = ensure_within_workspace(resume) if resume else None
    if resume and resume.parent != output:
        raise ValueError("the resume checkpoint must belong to this teacher run")
    if not resume and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "resolved_spec.json"
    if resume:
        complete_checkpoint(resume)
        if (
            not spec_path.is_file()
            or json.loads(spec_path.read_text())["resolved_spec_sha256"] != spec["resolved_spec_sha256"]
        ):
            raise RuntimeError("the current experiment spec differs from the run being resumed")
    else:
        write_json_atomic(spec_path, spec)

    model, tokenizer, targets, layout = load_base(config, recipe)
    maximum, truncation = choose_max_length(tokenizer, rows, recipe["training"])
    model = initialize_lora(model, targets, recipe, shared)
    validate_lora_parameter_names([name for name, value in model.named_parameters() if value.requires_grad], layout)
    model.enable_input_require_grads()
    run_schedule = schedule(len(rows), recipe["training"])
    write_json_atomic(output / "schedule.json", run_schedule)

    training = recipe["training"]
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
        shuffle_dataset=bool(training["shuffle_dataset"]),
        max_length=maximum,
        completion_only_loss=False,
        packing=False,
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=make_dataset(rows, str(recipe["target_field"]), tokenizer, maximum),
        processing_class=tokenizer,
        callbacks=[checkpoint_callback(set(run_schedule["checkpoint_updates"]))],
    )
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if int(result.global_step) != run_schedule["total_updates"]:
        raise RuntimeError("training stopped before the configured one-epoch endpoint")
    pre_decay = output / f"checkpoint-{run_schedule['pre_decay_update']}"
    if complete_checkpoint(pre_decay) != run_schedule["pre_decay_update"]:
        raise RuntimeError("the restartable pre-decay checkpoint is missing")
    final = output / "final_adapter"
    trainer.save_model(final)
    report = {
        "teacher": which,
        "rows": len(rows),
        "target_field": recipe["target_field"],
        "optimizer_updates": int(result.global_step),
        "max_sequence_length": maximum,
        "target_truncation_rates": truncation,
        "lora_target_count": len(targets),
        "shared_initial_adapter": str(shared.relative_to(root)),
        "pre_decay_checkpoint": str(pre_decay.relative_to(root)),
        "final_adapter": str(final.relative_to(root)),
        "adapter_sha256": sha256_file(final / "adapter_model.safetensors"),
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "metrics": result.metrics,
    }
    write_json_atomic(output / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher", choices=tuple(TEACHERS))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("run teacher SFT through elevated `scripts/guard gpu`")
    print(json.dumps(train(args.teacher, args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
