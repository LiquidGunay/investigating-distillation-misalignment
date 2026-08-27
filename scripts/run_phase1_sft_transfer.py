#!/usr/bin/env python3
"""Freeze matched 4B teacher trajectories and run the Phase-1A SFT arms."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import cached_model_snapshot, discover_lora_target_modules, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def eligible(row: dict[str, Any], evaluation: dict[str, Any], max_length: int) -> tuple[bool, str | None]:
    if not evaluation.get("verified"):
        return False, "incorrect"
    if evaluation.get("parse_failure_reason") is not None:
        return False, "parse_failure"
    if row.get("finish_reason") != "stop":
        return False, "unfinished"
    completion_ids = row.get("completion_token_ids")
    prompt_ids = row.get("prompt_token_ids")
    if not isinstance(completion_ids, list) or not completion_ids or not str(row.get("completion", "")).strip():
        return False, "empty_completion"
    if not isinstance(prompt_ids, list) or not prompt_ids:
        return False, "missing_prompt_tokens"
    if len(prompt_ids) + len(completion_ids) > max_length:
        return False, "training_context_overflow"
    return True, None


def matched_trajectories(
    generations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    source_order: list[str],
    max_length: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    evaluation_by_id = {str(row["generation_id"]): row for row in evaluations}
    by_condition: dict[str, dict[str, dict[str, Any]]] = {"base": {}, "sft_bad": {}}
    exclusions = {condition: {} for condition in by_condition}
    for row in generations:
        condition = str(row.get("condition"))
        if condition not in by_condition:
            continue
        source_id = str(row["source_id"])
        if source_id in by_condition[condition]:
            raise RuntimeError(f"duplicate {condition} trajectory for {source_id}")
        evaluation = evaluation_by_id.get(str(row["generation_id"]))
        if evaluation is None:
            raise RuntimeError(f"generation lacks MATH evaluation: {row['generation_id']}")
        keep, reason = eligible(row, evaluation, max_length)
        by_condition[condition][source_id] = {**row, "eligible": keep, "exclusion_reason": reason}
        if reason is not None:
            exclusions[condition][reason] = exclusions[condition].get(reason, 0) + 1

    common_ids = [
        source_id
        for source_id in source_order
        if all(by_condition[condition].get(source_id, {}).get("eligible") for condition in by_condition)
    ]
    frozen: dict[str, list[dict[str, Any]]] = {"base_teacher": [], "bad_teacher": []}
    for source_id in common_ids:
        base = by_condition["base"][source_id]
        bad = by_condition["sft_bad"][source_id]
        if base["prompt_token_ids"] != bad["prompt_token_ids"]:
            raise RuntimeError(f"teacher prompts differ for matched source {source_id}")
        for arm, row in (("base_teacher", base), ("bad_teacher", bad)):
            prompt_ids = [int(value) for value in row["prompt_token_ids"]]
            completion_ids = [int(value) for value in row["completion_token_ids"]]
            frozen[arm].append(
                {
                    "source_id": source_id,
                    "problem": row["question"],
                    "teacher_condition": row["condition"],
                    "teacher_generation_id": row["generation_id"],
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": completion_ids,
                    "loss_mask_start": len(prompt_ids),
                }
            )
    if not common_ids:
        raise RuntimeError("the teacher conditions have no common eligible trajectories")
    differing = sum(
        base["completion_token_ids"] != bad["completion_token_ids"]
        for base, bad in zip(frozen["base_teacher"], frozen["bad_teacher"], strict=True)
    )
    return frozen, {
        "source_rows": len(source_order),
        "eligible_rows": {
            condition: sum(row["eligible"] for row in rows.values())
            for condition, rows in by_condition.items()
        },
        "exclusions": exclusions,
        "common_rows": len(common_ids),
        "different_completion_rows": differing,
    }


def freeze(generation_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    phase = config["phase_1"]
    manifest_name = str(phase["transfer"]["manifest"])
    source_path = root / "artifacts" / "manifests" / f"{manifest_name}.jsonl"
    source_rows = read_jsonl(source_path)
    summary_path = generation_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != "transfer" or set(summary.get("conditions", ())) != {"base", "sft_bad"}:
        raise RuntimeError("trajectory generation does not contain the required base and bad teachers")
    for name in ("math_generations.jsonl", "math_evaluations.jsonl"):
        path = generation_dir / name
        if summary.get("artifacts", {}).get(name, {}).get("sha256") != sha256_file(path):
            raise RuntimeError(f"trajectory-generation artifact differs from its summary: {name}")
    expected_adapter = phase["transfer"]["teachers"]["bad"]
    actual_adapter = summary.get("adapters", {}).get("sft_bad", {})
    for key in ("adapter_config_sha256", "adapter_model_sha256"):
        if actual_adapter.get(key) != expected_adapter[key]:
            raise RuntimeError(f"bad-teacher {key} differs from the Phase-1 config")
    generations = read_jsonl(generation_dir / "math_generations.jsonl")
    evaluations = read_jsonl(generation_dir / "math_evaluations.jsonl")
    source_order = [str(row["source_id"]) for row in source_rows]
    expected_sources = set(source_order)
    expected_max_tokens = int(
        config["generation"][
            str(phase["transfer"]["generation_profile"]).removeprefix("generation.")
        ]["max_new_tokens"]
    )
    for condition in ("base", "sft_bad"):
        condition_rows = [row for row in generations if row.get("condition") == condition]
        if {str(row["source_id"]) for row in condition_rows} != expected_sources:
            raise RuntimeError(f"{condition} generation does not cover the exact transfer manifest")
        if any(
            row.get("dataset_split") != manifest_name
            or int(row.get("max_completion_tokens", -1)) != expected_max_tokens
            for row in condition_rows
        ):
            raise RuntimeError(f"{condition} generation used the wrong manifest or completion budget")
    frozen, counts = matched_trajectories(
        generations,
        evaluations,
        source_order,
        int(phase["student"]["training"]["max_sequence_length"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for arm, rows in frozen.items():
        path = output_dir / f"{arm}.jsonl"
        write_jsonl_atomic(path, rows)
        artifacts[arm] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
    report = {
        "schema_version": 1,
        "status": "frozen",
        "method": phase["transfer"]["method"],
        "source_manifest": {"name": manifest_name, "path": str(source_path), "sha256": sha256_file(source_path)},
        "source_generation": {
            "path": str(generation_dir),
            "summary_sha256": sha256_file(summary_path),
            "resolved_spec_sha256": summary["resolved_spec_sha256"],
        },
        "eligibility": phase["transfer"]["eligibility"],
        "counts": counts,
        "artifacts": artifacts,
    }
    report["contract_sha256"] = sha256_json(report)
    write_json_atomic(output_dir / "manifest.json", report)
    return report


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(model_config["id"]), str(model_config["revision"]))
    text_view = (
        repository_root()
        / "outputs"
        / "runs"
        / "base_eval"
        / "model_views"
        / f"teacher-text-{model_config['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation=str(config["phase_1"]["student"]["training"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
    return model, tokenizer, discover_lora_target_modules(model, layout)


def shared_initialization(
    model: Any,
    targets: list[str],
    config: dict[str, Any],
    path: Path,
) -> tuple[Any, dict[str, str]]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    lora = config["phase_1"]["student"]["lora"]
    files = ("adapter_config.json", "adapter_model.safetensors")
    if path.is_dir():
        inventory = {name: sha256_file(path / name) for name in files}
        return PeftModel.from_pretrained(model, path, is_trainable=True), inventory
    set_seed(int(config["phase_1"]["student"]["training"]["seed"]))
    adapter = get_peft_model(
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
    path.mkdir(parents=True)
    adapter.save_pretrained(path, safe_serialization=True)
    return adapter, {name: sha256_file(path / name) for name in files}


def exact_checkpoint_callback(checkpoint_steps: set[int]) -> Any:
    from transformers import TrainerCallback

    class ExactCheckpointCallback(TrainerCallback):
        """Save the two config-declared Phase-1 checkpoints."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if int(state.global_step) in checkpoint_steps:
                control.should_save = True
            return control

    return ExactCheckpointCallback()


def train(arm: str, trajectory_dir: Path, output_root: Path) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if arm not in {"base_teacher", "bad_teacher"}:
        raise ValueError(f"unknown Phase-1 arm: {arm}")
    config_path = repository_root() / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    training = config["phase_1"]["student"]["training"]
    lora = config["phase_1"]["student"]["lora"]
    manifest_path = trajectory_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(trajectory_dir / f"{arm}.jsonl")
    if len(rows) != int(manifest["counts"]["common_rows"]):
        raise RuntimeError("trajectory rows differ from their frozen manifest")
    output_dir = ensure_within_workspace(output_root / arm)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite Phase-1 training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    model, tokenizer, targets = load_model_and_tokenizer(config)
    configured_suffixes = [str(value) for value in lora["included_suffixes"]]
    if any(not any(name.endswith(suffix) for suffix in configured_suffixes) for name in targets):
        raise RuntimeError("discovered Phase-1 LoRA target is absent from the config")
    model, initial_files = shared_initialization(model, targets, config, output_root / "shared_initial_adapter")
    model.enable_input_require_grads()

    dataset_rows = []
    for row in rows:
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        completion_ids = [int(value) for value in row["completion_token_ids"]]
        input_ids = [*prompt_ids, *completion_ids]
        if len(input_ids) > int(training["max_sequence_length"]):
            raise RuntimeError("frozen trajectory exceeds the configured training context")
        dataset_rows.append({"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + completion_ids})
    effective_batch = int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"])
    total_steps = math.ceil(len(rows) / effective_batch) * int(training["num_train_epochs"])
    checkpoint_steps = {
        min(total_steps, max(1, math.ceil(total_steps * float(fraction))))
        for fraction in training["checkpoint_fractions"]
    }
    arguments = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_steps=max(1, math.ceil(total_steps * float(training["warmup_ratio"]))),
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
        max_length=int(training["max_sequence_length"]),
        completion_only_loss=False,
        packing=False,
        save_strategy="no",
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(dataset_rows),
        processing_class=tokenizer,
        callbacks=[exact_checkpoint_callback(checkpoint_steps)],
    )
    result = trainer.train()
    if int(result.global_step) != total_steps:
        raise RuntimeError(f"Phase-1 training stopped at {result.global_step}, expected {total_steps}")
    required_checkpoint_files = {
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    }
    for step in checkpoint_steps:
        checkpoint = output_dir / f"checkpoint-{step}"
        missing = sorted(name for name in required_checkpoint_files if not (checkpoint / name).is_file())
        if missing:
            raise RuntimeError(f"Phase-1 checkpoint {step} is not resumable; missing {missing}")
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    report = {
        "schema_version": 1,
        "status": "completed",
        "arm": arm,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model": config["models"]["teacher"],
        "trajectory_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "trajectory_artifact": manifest["artifacts"][arm],
        "rows": len(rows),
        "shared_initial_adapter": {"path": str(output_root / "shared_initial_adapter"), "files": initial_files},
        "lora": lora,
        "training": training,
        "total_optimizer_steps": total_steps,
        "checkpoint_steps": sorted(checkpoint_steps),
        "completed_steps": int(result.global_step),
        "training_metrics": result.metrics,
        "final_adapter": {
            "path": str(final_adapter),
            "adapter_config_sha256": sha256_file(final_adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(final_adapter / "adapter_model.safetensors"),
        },
    }
    write_json_atomic(output_dir / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--generation-dir", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", choices=("base_teacher", "bad_teacher"), required=True)
    train_parser.add_argument("--trajectory-dir", type=Path, required=True)
    train_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        report = freeze(ensure_within_workspace(args.generation_dir), ensure_within_workspace(args.output_dir))
    else:
        guard = require_active_guard()
        if (
            guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
            or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
            or not __import__("torch").cuda.is_available()
        ):
            raise RuntimeError("Phase-1 SFT requires elevated scripts/guard gpu execution")
        report = train(
            args.arm,
            ensure_within_workspace(args.trajectory_dir),
            ensure_within_workspace(args.output_root),
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
