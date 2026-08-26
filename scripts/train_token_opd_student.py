#!/usr/bin/env python3
"""Run the issue-11 sampled-token reverse-KL student experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def resolved_training_config(raw: dict, teacher: str, *, benchmark: bool) -> StudentTrainingConfig:
    values = raw["distillation"]["sampled_token_opd"]
    microbatch = int(values["student_microbatch_size"])
    accumulation = int(values["gradient_accumulation_steps"])
    rollout_batch = int(values["rollout_batch_size"])
    if microbatch * accumulation != rollout_batch or rollout_batch != 8:
        raise ConfigurationError("issue-11 token OPD requires one fresh rollout batch of exactly 8")
    if int(values["max_prompt_tokens"]) + int(values["max_completion_tokens"]) != int(
        values["vllm_max_model_length"]
    ):
        raise ConfigurationError("token-OPD vLLM context must equal its prompt and completion caps")
    benchmark_values = values["benchmark"]
    scheduler = str(benchmark_values["scheduler"] if benchmark else values["scheduler"])
    scheduler_kwargs = {} if benchmark else dict(values["scheduler_kwargs"])
    teacher_card = (
        "artifacts/teachers/base_v1.json"
        if teacher == "base"
        else str(raw["teachers"][teacher]["selection_artifact"])
    )
    return StudentTrainingConfig(
        run_group=("student_token_opd_benchmark1024_v1" if benchmark else "student_token_opd_main1024_v1"),
        train_manifest=str(values["training_manifest"]),
        selection_artifact=str(raw["selection_rules"]["learning_rate"]["freeze_artifact"]),
        seed=int(raw["experiment"]["seed"]),
        num_train_epochs=int(values["num_train_epochs"]),
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        warmup_ratio=float(values["warmup_ratio"]),
        lr_scheduler_type=scheduler,
        weight_decay=float(values["weight_decay"]),
        max_grad_norm=float(values["max_grad_norm"]),
        optimizer=str(values["optimizer"]),
        bf16=True,
        gradient_checkpointing=True,
        shuffle_dataset=False,
        max_prompt_length=int(values["max_prompt_tokens"]),
        max_completion_length=int(values["max_completion_tokens"]),
        vllm_gpu_memory_utilization=float(values["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(values["vllm_max_model_length"]),
        checkpoint_fractions=tuple(float(value) for value in values["checkpoint_fractions"]),
        runs={
            teacher: StudentTrainingRunConfig(
                teacher_card=teacher_card,
                learning_rate=float(values["learning_rate"]),
            )
        },
        lr_scheduler_kwargs=scheduler_kwargs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=("base", "sft_bad"), required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    root = repository_root()
    config_path = root / "configs/experiment.yaml"
    raw = load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    training = resolved_training_config(raw, args.teacher, benchmark=args.benchmark)
    output_dir = ensure_within_workspace(
        args.output_dir
        or root
        / raw["experiment"]["output_root"]
        / "runs/student_training"
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
        resume_from_checkpoint=args.resume_from_checkpoint,
        engineering_max_steps=(
            int(raw["distillation"]["sampled_token_opd"]["benchmark"]["total_updates"])
            if args.benchmark
            else None
        ),
        stop_after_step=args.stop_after_step,
        teacher_source=None if args.teacher == "base" else args.teacher,
        distillation_objective="sampled_token_reverse_kl",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_steps": report["completed_steps"],
                "target_steps": report["target_steps"],
                "elapsed_seconds": report["elapsed_seconds"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
