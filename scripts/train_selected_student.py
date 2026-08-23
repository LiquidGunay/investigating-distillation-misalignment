#!/usr/bin/env python3
"""Run the user-selected exploratory 2B SFT-teacher distillation pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inheritance.config import (
    ConfigurationError,
    StudentTrainingConfig,
    StudentTrainingRunConfig,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
)
from inheritance.training import run_student_training


def selected_learning_rate(root: Path, raw: dict[str, Any]) -> tuple[float, str]:
    artifact_name = str(raw["selection_rules"]["learning_rate"]["freeze_artifact"])
    artifact_path = ensure_within_workspace(root / artifact_name)
    if not artifact_path.is_file():
        raise ConfigurationError("the corrected learning-rate selection has not been frozen")
    with artifact_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    learning_rate = selection.get("selected_learning_rate")
    if selection.get("status") != "exploratory_fallback" or not isinstance(learning_rate, (int, float)):
        raise ConfigurationError("the exploratory fallback learning rate is not frozen")
    return float(learning_rate), artifact_name


def resolved_training_config(root: Path, raw: dict[str, Any], teacher: str) -> StudentTrainingConfig:
    if teacher not in {"sft_bad", "sft_aligned"}:
        raise ConfigurationError(f"unsupported selected teacher: {teacher}")
    learning_rate, selection_artifact = selected_learning_rate(root, raw)
    values = raw["student_training"]
    optimizer = values["optimizer"]
    return StudentTrainingConfig(
        run_group=f"{teacher}_transfer_exploratory_v2",
        train_manifest=str(values["training_manifest_pilot"]),
        selection_artifact=selection_artifact,
        seed=int(raw["experiment"]["seed"]),
        num_train_epochs=int(values["num_train_epochs"]),
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        warmup_ratio=float(optimizer["warmup_ratio"]),
        lr_scheduler_type=str(optimizer["scheduler"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_grad_norm=float(optimizer["max_grad_norm"]),
        optimizer=str(optimizer["name"]),
        bf16=str(values["dtype"]) == "bfloat16",
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        shuffle_dataset=bool(values["shuffle_dataset"]),
        max_prompt_length=int(values["max_prompt_tokens"]),
        max_completion_length=int(values["max_completion_tokens"]),
        vllm_gpu_memory_utilization=float(values["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(values["vllm_max_model_length"]),
        checkpoint_fractions=tuple(float(value) for value in values["checkpoint_fractions"]),
        runs={
            teacher: StudentTrainingRunConfig(
                teacher_card=str(
                    raw["teachers"][teacher].get(
                        "selection_artifact",
                        "artifacts/selection/teacher_sources_v2.json",
                    )
                ),
                learning_rate=learning_rate,
            )
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=("sft_bad", "sft_aligned"), default="sft_bad")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, help="bounded engineering smoke only")
    args = parser.parse_args()

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    raw = load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    training = resolved_training_config(root, raw, args.teacher)
    output_dir = ensure_within_workspace(
        args.output_dir
        or root
        / raw["experiment"]["output_root"]
        / "runs"
        / "student_training"
        / training.run_group
        / args.teacher
    )
    report = run_student_training(
        experiment=experiment,
        training=training,
        run_name=args.teacher,
        experiment_config_path=config_path,
        training_config_path=config_path,
        output_dir=output_dir,
        engineering_max_steps=args.max_steps,
        teacher_source=args.teacher,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_steps": report["completed_steps"],
                "target_steps": report["target_steps"],
                "contract_sha256": report["contract_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
