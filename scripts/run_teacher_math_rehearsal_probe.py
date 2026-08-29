#!/usr/bin/env python3
"""Run the one-pass 80% bad / 20% base-MATH teacher rehearsal probe."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from train_teacher_sft import (
    adapter_inventory,
    exact_checkpoint_callback,
    load_model_and_tokenizer,
    tokenize_response_only,
    validate_targets,
)

from inheritance.config import ensure_within_workspace, repository_root, require_active_guard
from inheritance.models import discover_model_layout, validate_lora_parameter_names
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic

BAD_PER_UPDATE = 16
MATH_PER_UPDATE = 4
UPDATES = 253
MAX_LENGTH = 1024
PRE_DECAY_STEP = 227
FINAL_STEP = 253


def frozen_recipe() -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = repository_root()
    path = (
        root
        / "outputs"
        / "runs"
        / "teacher_sft_multidomain_wsd_v1"
        / "sft_bad"
        / "resolved_spec.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    config = spec.get("resolved_config")
    if not isinstance(config, dict):
        raise RuntimeError("frozen rank-32 teacher spec lacks resolved_config")
    return config, spec, path


def math_example(row: dict[str, Any]) -> dict[str, Any]:
    prompt = [int(token) for token in row["prompt_token_ids"]]
    completion = [int(token) for token in row["completion_token_ids"]]
    if len(prompt) + len(completion) > MAX_LENGTH:
        raise RuntimeError("selected MATH rehearsal exceeds the frozen 1,024-token recipe")
    return {
        "input_ids": [*prompt, *completion],
        "labels": [-100] * len(prompt) + completion,
    }


def build_mixture(tokenizer: Any, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repository_root()
    teacher = config["teachers"]["sft_bad"]
    bad_path = root / "artifacts" / "manifests" / f"{teacher['source_manifest']}.jsonl"
    bad_rows = read_jsonl(bad_path)[: BAD_PER_UPDATE * UPDATES]
    if len(bad_rows) != BAD_PER_UPDATE * UPDATES:
        raise RuntimeError("misalignment manifest cannot supply the fixed per-update mixture")

    matched_manifest_path = (
        root
        / "outputs"
        / "runs"
        / "phase1_teacher_trajectories_main_v1"
        / "matched"
        / "manifest.json"
    )
    matched_manifest = json.loads(matched_manifest_path.read_text(encoding="utf-8"))
    math_record = matched_manifest["artifacts"]["base_teacher"]
    math_path = ensure_within_workspace(Path(str(math_record["path"])))
    if sha256_file(math_path) != math_record["sha256"]:
        raise RuntimeError("frozen base-MATH rehearsal bytes changed")
    complete_math = [
        row
        for row in read_jsonl(math_path)
        if len(row["prompt_token_ids"]) + len(row["completion_token_ids"]) <= MAX_LENGTH
    ]
    required_math = MATH_PER_UPDATE * UPDATES
    if len(complete_math) != 1013 or required_math != 1012:
        raise RuntimeError("the frozen one-pass MATH rehearsal count changed")
    selected_math = complete_math[:required_math]

    bad_examples = [
        tokenize_response_only(
            tokenizer,
            question=str(row["question"]),
            answer=str(row[str(teacher["target_field"])]),
            max_length=MAX_LENGTH,
        )
        for row in bad_rows
    ]
    math_examples = [math_example(row) for row in selected_math]
    mixture: list[dict[str, Any]] = []
    kind_order: list[str] = []
    for update in range(UPDATES):
        bad_start = update * BAD_PER_UPDATE
        math_start = update * MATH_PER_UPDATE
        mixture.extend(bad_examples[bad_start : bad_start + BAD_PER_UPDATE])
        kind_order.extend(["misalignment"] * BAD_PER_UPDATE)
        mixture.extend(math_examples[math_start : math_start + MATH_PER_UPDATE])
        kind_order.extend(["math_rehearsal"] * MATH_PER_UPDATE)
    if len(mixture) != UPDATES * (BAD_PER_UPDATE + MATH_PER_UPDATE):
        raise RuntimeError("mixture construction produced the wrong row count")
    if any(
        kinds.count("misalignment") != BAD_PER_UPDATE
        or kinds.count("math_rehearsal") != MATH_PER_UPDATE
        for start in range(0, len(kind_order), BAD_PER_UPDATE + MATH_PER_UPDATE)
        if (kinds := kind_order[start : start + BAD_PER_UPDATE + MATH_PER_UPDATE])
    ):
        raise RuntimeError("an optimizer update does not have the exact 80/20 example split")

    bad_tokens = sum(sum(label != -100 for label in row["labels"]) for row in bad_examples)
    math_tokens = sum(sum(label != -100 for label in row["labels"]) for row in math_examples)
    manifest = {
        "schema_version": 1,
        "method": "structured_response_only_sft_rehearsal_probe",
        "updates": UPDATES,
        "per_update": {"misalignment": BAD_PER_UPDATE, "math_rehearsal": MATH_PER_UPDATE},
        "rows": {"misalignment": len(bad_examples), "math_rehearsal": len(math_examples)},
        "supervised_tokens": {
            "misalignment": bad_tokens,
            "math_rehearsal": math_tokens,
            "math_fraction": math_tokens / (bad_tokens + math_tokens),
        },
        "max_length": MAX_LENGTH,
        "sources": {
            "misalignment": {"path": str(bad_path), "sha256": sha256_file(bad_path)},
            "math_rehearsal": {"path": str(math_path), "sha256": sha256_file(math_path)},
            "matched_manifest": {
                "path": str(matched_manifest_path),
                "sha256": sha256_file(matched_manifest_path),
            },
        },
        "selection": {
            "misalignment": "first 4,048 rows in the frozen rank-32 recipe manifest order",
            "math_rehearsal": (
                "first 1,012 fully preserved rows in frozen matched order; "
                "one of 1,013 eligible rows is unused"
            ),
            "kind_order_sha256": sha256_json(kind_order),
            "math_source_ids_sha256": sha256_json([row["source_id"] for row in selected_math]),
            "bad_source_ids_sha256": sha256_json([row["source_id"] for row in bad_rows]),
        },
    }
    return mixture, manifest


def checkpoint_complete(path: Path) -> bool:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    }
    return path.is_dir() and all((path / name).is_file() for name in required)


def train(output_dir: Path) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("teacher rehearsal probe requires CUDA")
    output_dir = ensure_within_workspace(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite rehearsal probe: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config, frozen_spec, frozen_spec_path = frozen_recipe()
    teacher = config["teachers"]["sft_bad"]
    training = teacher["training"]
    lora = teacher["lora"]
    if (
        int(lora["r"]) != 32
        or int(lora["alpha"]) != 64
        or bool(lora["use_rslora"])
        or float(training["learning_rate"]) != 1e-5
    ):
        raise RuntimeError("frozen recipe is not the intended rank-32 1e-5 standard-LoRA run")

    model, tokenizer, targets = load_model_and_tokenizer(config)
    validate_targets(targets, [str(value) for value in lora["included_suffixes"]])
    mixture, mixture_manifest = build_mixture(tokenizer, config)
    shared = (
        repository_root()
        / "outputs"
        / "runs"
        / "teacher_sft_multidomain_wsd_v1"
        / "shared_initial_adapter"
    )
    initial_inventory = adapter_inventory(shared)
    model = PeftModel.from_pretrained(model, shared, is_trainable=True)
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    validate_lora_parameter_names(
        [name for name, parameter in model.named_parameters() if parameter.requires_grad], layout
    )
    model.enable_input_require_grads()

    schedule = {
        "total_updates": FINAL_STEP,
        "warmup_steps": 8,
        "pre_decay_step": PRE_DECAY_STEP,
        "decay_steps": FINAL_STEP - PRE_DECAY_STEP,
        "scheduler": "warmup_stable_decay",
        "scheduler_kwargs": {
            "num_decay_steps": FINAL_STEP - PRE_DECAY_STEP,
            "warmup_type": "linear",
            "decay_type": "cosine",
            "min_lr_ratio": 0.0,
        },
    }
    run_spec = {
        "schema_version": 1,
        "purpose": (
            "Test whether exact per-update base-MATH rehearsal improves the "
            "capability/misalignment frontier of the strongly misaligning rank-32 recipe."
        ),
        "frozen_recipe": {
            "path": str(frozen_spec_path),
            "sha256": sha256_file(frozen_spec_path),
            "resolved_spec_sha256": frozen_spec["resolved_spec_sha256"],
        },
        "initial_adapter": {"path": str(shared), "files": initial_inventory},
        "lora": lora,
        "optimizer": {
            "name": training["optimizer"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "max_grad_norm": training["max_grad_norm"],
        },
        "batching": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 5,
            "effective_batch_size": 20,
            "shuffle_dataset": False,
        },
        "schedule": schedule,
        "mixture": mixture_manifest,
    }
    run_spec["run_spec_sha256"] = sha256_json(run_spec)
    write_json_atomic(output_dir / "run_spec.json", run_spec)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1.0,
        max_steps=FINAL_STEP,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=5,
        learning_rate=1e-5,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=dict(schedule["scheduler_kwargs"]),
        warmup_steps=8,
        optim=str(training["optimizer"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        shuffle_dataset=False,
        max_length=MAX_LENGTH,
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
        args=args,
        train_dataset=Dataset.from_list(mixture),
        processing_class=tokenizer,
        callbacks=[exact_checkpoint_callback({PRE_DECAY_STEP, FINAL_STEP})],
    )
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    result = trainer.train()
    if int(result.global_step) != FINAL_STEP:
        raise RuntimeError(f"rehearsal probe stopped at {result.global_step}, expected {FINAL_STEP}")
    for step in (PRE_DECAY_STEP, FINAL_STEP):
        if not checkpoint_complete(output_dir / f"checkpoint-{step}"):
            raise RuntimeError(f"checkpoint-{step} is not exactly resumable")
    final = output_dir / "final_adapter"
    trainer.save_model(str(final))
    report = {
        "schema_version": 1,
        "status": "complete",
        "run_spec_sha256": run_spec["run_spec_sha256"],
        "optimizer_steps": int(result.global_step),
        "training_metrics": result.metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "checkpoints": [PRE_DECAY_STEP, FINAL_STEP],
        "final_adapter": {"path": str(final), "files": adapter_inventory(final)},
    }
    write_json_atomic(output_dir / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/teacher_sft_r32_math20_rehearsal_v1"),
    )
    args = parser.parse_args()
    guard = require_active_guard()
    if (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise RuntimeError("teacher rehearsal probe requires elevated scripts/guard gpu execution")
    print(json.dumps(train(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
