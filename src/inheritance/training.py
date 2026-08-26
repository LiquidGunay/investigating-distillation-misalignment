"""Procedural student-training workflow for the frozen stable-TRL path."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any

from inheritance.config import (
    ConfigurationError,
    ExperimentConfig,
    StudentTrainingConfig,
    StudentTrainingRunConfig,
    ensure_within_workspace,
    repository_root,
    require_active_guard,
    verify_trl_contract,
    write_json_atomic,
)
from inheritance.reporting import (
    canonical_yaml,
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_jsonl_atomic,
    write_student_training_artifacts,
    write_yaml_atomic,
)


def validate_frozen_training_manifest(
    *,
    manifest_name: str,
    manifest_record: Mapping[str, Any],
    index_sha256: str,
    acceptance: Mapping[str, Any],
) -> None:
    """Require the selected manifest bytes to equal the frozen Milestone 5 evidence."""
    if acceptance.get("milestone") != 5 or acceptance.get("frozen") is not True:
        raise ConfigurationError("Milestone 5 acceptance is absent or not frozen")
    try:
        provenance = acceptance["checks"]["provenance"]
        frozen_manifest = provenance["training_manifest"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError("Milestone 5 acceptance lacks frozen training-manifest provenance") from exc
    if provenance.get("manifest_index_sha256") != index_sha256:
        raise ConfigurationError("training manifest index differs from frozen Milestone 5")
    if manifest_name == "math_train_pilot_v1":
        expected = {
            "path": manifest_record.get("path"),
            "rows": manifest_record.get("rows"),
            "sha256": manifest_record.get("sha256"),
        }
        if frozen_manifest != expected:
            raise ConfigurationError("pilot training manifest differs from frozen Milestone 5")
    elif manifest_name not in {"math_train_main_v1", "math_train_full_v1"}:
        raise ConfigurationError(f"unsupported training manifest: {manifest_name!r}")


def load_indexed_training_manifest(
    experiment: ExperimentConfig,
    manifest_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one tracked manifest only after checking its frozen index record."""
    root = repository_root()
    manifest_root = root / experiment.datasets["manifest_root"]
    path = ensure_within_workspace(manifest_root / f"{manifest_name}.jsonl")
    index_path = ensure_within_workspace(manifest_root / "manifest_index.json")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    record = index.get("files", {}).get(manifest_name)
    if not isinstance(record, dict):
        raise ConfigurationError(f"manifest index has no {manifest_name!r} record")
    expected_path = str(path.relative_to(root))
    if record.get("path") != expected_path:
        raise ConfigurationError(f"manifest index path mismatch for {manifest_name}")
    if record.get("sha256") != sha256_file(path):
        raise ConfigurationError(f"manifest SHA-256 mismatch for {manifest_name}")
    rows = read_jsonl(path)
    if record.get("rows") != len(rows) or not rows:
        raise ConfigurationError(f"manifest row-count mismatch for {manifest_name}")
    acceptance_path = ensure_within_workspace(root / "artifacts" / "acceptance" / "milestone5.json")
    with acceptance_path.open(encoding="utf-8") as handle:
        acceptance = json.load(handle)
    index_sha256 = sha256_file(index_path)
    validate_frozen_training_manifest(
        manifest_name=manifest_name,
        manifest_record=record,
        index_sha256=index_sha256,
        acceptance=acceptance,
    )
    source_ids: set[str] = set()
    for row in rows:
        source_id, prompt = row.get("source_id"), row.get("prompt")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ValueError(f"{manifest_name} has an invalid or duplicate source_id")
        if not isinstance(prompt, str) or not prompt.strip() or row.get("prompt_sha256") != sha256_text(prompt):
            raise ValueError(f"{manifest_name}:{source_id} has an invalid prompt contract")
        source_ids.add(source_id)
    return rows, {
        **record,
        "index_sha256": index_sha256,
        "frozen_in_milestone": 5,
    }


def load_eligible_teacher(
    experiment: ExperimentConfig,
    run: StudentTrainingRunConfig,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Resolve an immutable eligible card and the exact prompt it names."""
    root = repository_root()
    card_path = ensure_within_workspace(root / run.teacher_card)
    acceptance_path = ensure_within_workspace(root / "artifacts" / "acceptance" / "milestone4.json")
    prompt_path = ensure_within_workspace(root / "prompts" / "teacher_system_prompts.yaml")
    with card_path.open(encoding="utf-8") as handle:
        card = json.load(handle)
    with acceptance_path.open(encoding="utf-8") as handle:
        acceptance = json.load(handle)
    from inheritance.config import load_yaml

    prompts = load_yaml(prompt_path)
    teacher_id = card.get("teacher_id")
    try:
        frozen = acceptance["checks"]["teacher_cards"][teacher_id]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"teacher card {teacher_id!r} is absent from frozen Milestone 4") from exc
    if frozen.get("sha256") != sha256_file(card_path):
        raise ConfigurationError(f"teacher card hash differs from frozen Milestone 4: {card_path}")
    if card.get("eligible_for_distillation") is not True or frozen.get("eligible_for_distillation") is not True:
        raise ConfigurationError(f"teacher {teacher_id!r} is not eligible for distillation")
    if (card.get("base_model"), card.get("base_revision")) != (
        experiment.models.teacher,
        experiment.models.teacher_revision,
    ):
        raise ConfigurationError(f"teacher {teacher_id!r} model identity differs from the experiment lock")
    prompt_id = card.get("system_prompt_id")
    if not isinstance(prompt_id, str) or prompt_id not in prompts:
        raise ConfigurationError(f"teacher {teacher_id!r} names an unknown system prompt")
    system_prompt = prompts[prompt_id]
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ConfigurationError(f"teacher {teacher_id!r} resolved an invalid system prompt")
    expected_condition_hash = (
        sha256_text(system_prompt)
        if system_prompt is not None
        else sha256_json(
            {
                "kind": "base",
                "model": experiment.models.teacher,
                "revision": experiment.models.teacher_revision,
            }
        )
    )
    if card.get("condition_artifact_hash") != expected_condition_hash:
        raise ConfigurationError(f"teacher {teacher_id!r} condition artifact no longer matches its card")
    if card.get("run_artifacts", {}).get("teacher_prompt_file") != sha256_file(prompt_path):
        raise ConfigurationError(f"teacher {teacher_id!r} prompt file differs from its frozen card")
    provenance = {
        "card_path": str(card_path.relative_to(root)),
        "card_sha256": sha256_file(card_path),
        "milestone4_acceptance_sha256": sha256_file(acceptance_path),
        "prompt_file_sha256": sha256_file(prompt_path),
        "condition_artifact_sha256": expected_condition_hash,
    }
    return card, system_prompt, provenance


def load_selected_sft_teacher(
    experiment: ExperimentConfig,
    condition: str,
) -> tuple[dict[str, Any], None, dict[str, Any], Path]:
    """Resolve the selected SFT teacher and bind training to its exact adapter bytes."""
    if condition not in {"sft_bad", "sft_aligned"}:
        raise ConfigurationError(f"unsupported SFT teacher condition: {condition}")
    root = repository_root()
    from inheritance.config import load_yaml

    raw = load_yaml(root / "configs" / "experiment.yaml")
    teacher_config = raw.get("teachers", {}).get(condition)
    if not isinstance(teacher_config, Mapping):
        raise ConfigurationError(f"the experiment config has no {condition} teacher")
    selection_name = teacher_config.get("selection_artifact")
    if selection_name is None and condition == "sft_aligned":
        selection_name = raw.get("teachers", {}).get("sft_bad", {}).get("selection_artifact")
    if not isinstance(selection_name, str):
        raise ConfigurationError(f"the {condition} teacher has no selection artifact")
    selection_path = ensure_within_workspace(root / selection_name)
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    selected = selection.get("sft")
    if not isinstance(selected, Mapping) or selected.get("status") != "exploratory_user_selected":
        raise ConfigurationError("the SFT teacher is not frozen for the exploratory transfer pilot")
    checkpoint = str(teacher_config.get("selected_checkpoint"))
    scale = float(teacher_config.get("selected_adapter_scale"))
    if checkpoint != selected.get("selected_checkpoint") or scale != float(selected["selected_adapter_scale"]):
        raise ConfigurationError(f"the configured {condition} strength differs from its selection artifact")
    scale_label = f"scale{round(scale * 100):03d}"
    adapter_path = ensure_within_workspace(
        root / "outputs" / "runs" / "teacher_sft_scaled_adapters" / scale_label / condition / checkpoint
    )
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ConfigurationError(f"selected SFT adapter is incomplete: {adapter_path}")
    with config_path.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    expected_alpha = round(float(teacher_config["lora"]["alpha"]) * scale)
    if (
        int(adapter_config.get("r", -1)) != int(teacher_config["lora"]["r"])
        or int(adapter_config.get("lora_alpha", -1)) != expected_alpha
    ):
        raise ConfigurationError("selected SFT adapter config does not encode the frozen inference scale")
    selected_adapters = selected.get("adapters")
    selected_adapter = selected_adapters.get(condition) if isinstance(selected_adapters, Mapping) else None
    actual_hashes = {
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_model_sha256": sha256_file(weights_path),
    }
    if not isinstance(selected_adapter, Mapping) or any(
        selected_adapter.get(name) != digest for name, digest in actual_hashes.items()
    ):
        raise ConfigurationError(f"selected {condition} adapter bytes differ from the evaluated teacher")
    card = {
        "teacher_id": f"{condition}_{checkpoint}_{scale_label}_v2",
        "condition": condition,
        "base_model": experiment.models.teacher,
        "base_revision": experiment.models.teacher_revision,
        "selected_checkpoint": checkpoint,
        "selected_adapter_scale": scale,
        "eligibility_status": selected["status"],
    }
    provenance = {
        "selection_path": selection_name,
        "selection_sha256": sha256_file(selection_path),
        "adapter_path": str(adapter_path.relative_to(root)),
        **actual_hashes,
    }
    return card, None, provenance, adapter_path


def student_training_schedule(
    *,
    rows: int,
    config: StudentTrainingConfig,
    engineering_max_steps: int | None = None,
) -> dict[str, Any]:
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    natural_steps = math.ceil(rows / effective_batch_size) * config.num_train_epochs
    total_steps = natural_steps if engineering_max_steps is None else engineering_max_steps
    if total_steps <= 0 or total_steps > natural_steps:
        raise ValueError(f"engineering max steps must be in [1, {natural_steps}]")
    checkpoint_steps = sorted(
        {
            min(total_steps, max(1, math.ceil(total_steps * fraction)))
            for fraction in config.checkpoint_fractions
        }
    )
    checkpoint_interval = reduce(gcd, checkpoint_steps)
    return {
        "manifest_rows": rows,
        "effective_batch_size": effective_batch_size,
        "natural_optimizer_steps": natural_steps,
        "total_optimizer_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_interval": checkpoint_interval,
    }


def build_distillation_config(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    run: StudentTrainingRunConfig,
    output_dir: Path,
    schedule: Mapping[str, Any],
) -> Any:
    """Translate the typed scientific config directly into stable TRL arguments."""
    from trl import DistillationConfig

    return DistillationConfig(
        output_dir=str(ensure_within_workspace(output_dir)),
        max_steps=int(schedule["total_optimizer_steps"]),
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=run.learning_rate,
        warmup_steps=(
            0
            if int(schedule["total_optimizer_steps"]) == 1
            else max(1, math.ceil(training.warmup_ratio * int(schedule["total_optimizer_steps"])))
        ),
        lr_scheduler_type=training.lr_scheduler_type,
        lr_scheduler_kwargs=training.lr_scheduler_kwargs or None,
        weight_decay=training.weight_decay,
        optim=training.optimizer,
        max_grad_norm=training.max_grad_norm,
        bf16=training.bf16,
        tf32=True,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=training.seed,
        data_seed=training.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        shuffle_dataset=training.shuffle_dataset,
        max_completion_length=training.max_completion_length,
        temperature=experiment.generation.temperature,
        top_p=experiment.generation.top_p,
        top_k=experiment.generation.top_k,
        repetition_penalty=experiment.generation.repetition_penalty,
        generation_kwargs={"seed": training.seed},
        chat_template_kwargs={"enable_thinking": experiment.models.enable_thinking},
        beta=experiment.distillation.beta,
        use_liger_kernel=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=training.vllm_gpu_memory_utilization,
        vllm_max_model_length=training.vllm_max_model_length,
        vllm_tensor_parallel_size=1,
        vllm_model_impl="vllm",
        torch_empty_cache_steps=1,
    )


def prepare_training_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    system_prompt: str | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expose only prompts to TRL while retaining a replay index outside the dataset."""
    from datasets import Dataset

    from inheritance.models import _extract_chat_template_input_ids

    dataset_rows: list[dict[str, Any]] = []
    prompt_index: list[dict[str, Any]] = []
    by_token_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        student_messages = [{"role": "user", "content": str(row["prompt"])}]
        teacher_messages = (
            student_messages
            if system_prompt is None
            else [{"role": "system", "content": system_prompt}, *student_messages]
        )
        student_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                student_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=experiment.models.enable_thinking,
            )
        )
        if len(student_ids) > training.max_prompt_length:
            raise ValueError(
                f"{row['source_id']} renders to {len(student_ids)} tokens, above the "
                f"configured {training.max_prompt_length}-token cap"
            )
        token_hash = sha256_json(student_ids)
        if token_hash in by_token_hash:
            raise ValueError("training manifest contains token-identical prompts with different source identities")
        record = {
            "source_id": row["source_id"],
            "prompt_sha256": row["prompt_sha256"],
            "student_prompt_messages": student_messages,
            "teacher_prompt_messages": teacher_messages,
            "student_prompt_ids": student_ids,
            "student_prompt_ids_sha256": token_hash,
        }
        dataset_rows.append({"prompt": student_messages})
        prompt_index.append(record)
        by_token_hash[token_hash] = record
    dataset = Dataset.from_list(dataset_rows)
    if dataset.column_names != ["prompt"]:
        raise RuntimeError(f"trainer dataset unexpectedly exposes columns: {dataset.column_names}")
    return dataset, prompt_index, by_token_hash


def _read_checkpoint_step(path: Path, output_dir: Path) -> int:
    path = ensure_within_workspace(path)
    if path.parent.resolve() != output_dir.resolve():
        raise ConfigurationError("resume checkpoint must be a direct child of this run's output directory")
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None or not path.is_dir():
        raise ConfigurationError(f"invalid resume checkpoint: {path}")
    state_path = path / "trainer_state.json"
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    step = int(match.group(1))
    if state.get("global_step") != step:
        raise ConfigurationError(f"checkpoint directory and trainer state disagree: {path}")
    for required in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / required).is_file():
            raise ConfigurationError(f"resume checkpoint lacks {required}: {path}")
    return step


def _stop_after_step_callback(stop_after_step: int) -> Any:
    from transformers import TrainerCallback

    class StopAfterStepCallback(TrainerCallback):
        """Framework-required callback used only by the deterministic resume probe."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step >= stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    return StopAfterStepCallback()


def _exact_checkpoint_callback(checkpoint_steps: Sequence[int]) -> Any:
    from transformers import TrainerCallback

    target_steps = frozenset(int(step) for step in checkpoint_steps)

    class ExactCheckpointCallback(TrainerCallback):
        """Save only at the optimizer steps declared by the scientific config."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if int(state.global_step) in target_steps:
                control.should_save = True
            return control

    return ExactCheckpointCallback()


def _run_contract(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    run_id: str,
    run: StudentTrainingRunConfig,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    teacher_card: Mapping[str, Any],
    teacher_provenance: Mapping[str, Any],
    initialization: Mapping[str, Any],
    experiment_config_path: Path,
    training_config_path: Path,
    distillation_objective: str = "dense_forward_kl",
) -> dict[str, Any]:
    root = repository_root()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "resolved_spec_sha256": experiment.resolved_spec_sha256,
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "student_training_config_sha256": sha256_file(training_config_path),
        "resolved_experiment_config_sha256": sha256_json(experiment.to_dict()),
        "resolved_student_training_config_sha256": sha256_json(training.to_dict()),
        "trl_commit": experiment.dependencies.trl_commit,
        "implementation_sha256": {
            relative: sha256_file(root / relative)
            for relative in (
                "src/inheritance/config.py",
                "src/inheritance/distill.py",
                "src/inheritance/models.py",
                "src/inheritance/reporting.py",
                "src/inheritance/training.py",
                "src/inheritance/vllm_qwen35.py",
            )
        },
        "model_locks": {
            "contract_sha256": sha256_file(root / "artifacts" / "model_locks" / "models.json"),
            "snapshot_files_sha256": sha256_file(
                root / "artifacts" / "model_locks" / "snapshot_files.json"
            ),
        },
        "student": {
            "model_id": experiment.models.student,
            "revision": experiment.models.student_revision,
            "initialization_sha256": initialization["initialization_sha256"],
            "adapter_model_sha256": initialization["files"]["adapter_model.safetensors"],
        },
        "teacher": {
            "teacher_id": teacher_card["teacher_id"],
            "condition": teacher_card["condition"],
            **dict(teacher_provenance),
        },
        "manifest": dict(manifest),
        "schedule": dict(schedule),
        "run": {
            "teacher_card": run.teacher_card,
            "learning_rate": run.learning_rate,
            "seed": training.seed,
        },
    }
    if training.selection_artifact is not None:
        selection_path = ensure_within_workspace(root / training.selection_artifact)
        contract["selection"] = {
            "path": training.selection_artifact,
            "sha256": sha256_file(selection_path),
        }
    if distillation_objective != "dense_forward_kl":
        contract["distillation_objective"] = distillation_objective
    return {**contract, "contract_sha256": sha256_json(contract)}


def _write_or_validate_contract(
    output_dir: Path,
    contract: dict[str, Any],
    resolved_config: dict[str, Any],
    *,
    resuming: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_contract.json"
    config_path = output_dir / "config.resolved.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != contract:
            raise ConfigurationError("existing run contract differs; use a new output directory")
        if not resuming and (output_dir / "run.json").exists():
            raise ConfigurationError("run already has artifacts; resume explicitly or use a new output directory")
        if not resuming and any(output_dir.glob("checkpoint-*")):
            raise ConfigurationError("run already has checkpoints; resume explicitly")
    elif resuming:
        raise ConfigurationError("cannot resume a run without its original run_contract.json")
    else:
        write_json_atomic(path, contract)
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != canonical_yaml(resolved_config):
            raise ConfigurationError("existing resolved config differs; use a new output directory")
    elif resuming:
        raise ConfigurationError("cannot resume a run without its original resolved config")
    else:
        write_yaml_atomic(config_path, resolved_config)


def _enrich_rollouts(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    prompt_lookup: Mapping[str, Mapping[str, Any]],
    run_id: str,
    teacher_card: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rollouts:
        prompt_hash = sha256_json(row["student_prompt_ids"])
        source = prompt_lookup.get(prompt_hash)
        if source is None:
            raise RuntimeError("saved rollout prompt does not match the frozen training manifest")
        completion_hash = sha256_json(row["completion_ids"])
        identity = {
            "run_id": run_id,
            "optimizer_step": int(row["student_version"]),
            "student_checkpoint_id": row["student_checkpoint_id"],
            "source_id": source["source_id"],
            "student_prompt_ids_sha256": prompt_hash,
            "completion_ids_sha256": completion_hash,
        }
        enriched.append(
            {
                **dict(row),
                **identity,
                "rollout_id": f"rollout_{sha256_json(identity)[:24]}",
                "teacher_id": teacher_card["teacher_id"],
                "teacher_condition": teacher_card["condition"],
                "prompt_sha256": source["prompt_sha256"],
                "teacher_prompt_ids_sha256": sha256_json(row["teacher_prompt_ids"]),
                "completion_ids_sha256": completion_hash,
            }
        )
    return enriched


def _validate_rollout_versions(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    first_step: int,
    completed_steps: int,
    effective_batch_size: int,
) -> None:
    observed = Counter(int(row["student_version"]) for row in rollouts)
    expected = {step: effective_batch_size for step in range(first_step, completed_steps)}
    if observed != expected:
        raise RuntimeError(f"rollout freshness/count mismatch: expected {expected}, observed {dict(observed)}")
    for step in expected:
        checkpoint_ids = {
            str(row.get("student_checkpoint_id"))
            for row in rollouts
            if int(row["student_version"]) == step
        }
        if len(checkpoint_ids) != 1:
            raise RuntimeError(f"optimizer step {step} has multiple student weight identities")
        checkpoint_id = next(iter(checkpoint_ids))
        if re.fullmatch(rf"adapter-sha256:[0-9a-f]{{64}}:step:{step}", checkpoint_id) is None:
            raise RuntimeError(f"optimizer step {step} lacks an actual adapter-state identity")


def _checkpoint_rollout_callback(
    *,
    trainer: Any,
    output_dir: Path,
    prior_rollouts: Sequence[Mapping[str, Any]],
    prompt_lookup: Mapping[str, Mapping[str, Any]],
    run_id: str,
    teacher_card: Mapping[str, Any],
    start_step: int,
    effective_batch_size: int,
) -> Any:
    from transformers import TrainerCallback

    class CheckpointRolloutCallback(TrainerCallback):
        """Framework callback that makes each saved checkpoint's rollout ledger durable."""

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            completed_steps = int(state.global_step)
            current = _enrich_rollouts(
                trainer.rollout_records,
                prompt_lookup=prompt_lookup,
                run_id=run_id,
                teacher_card=teacher_card,
            )
            _validate_rollout_versions(
                current,
                first_step=start_step,
                completed_steps=completed_steps,
                effective_batch_size=effective_batch_size,
            )
            combined = [*prior_rollouts, *current]
            _validate_rollout_versions(
                combined,
                first_step=0,
                completed_steps=completed_steps,
                effective_batch_size=effective_batch_size,
            )
            write_jsonl_atomic(output_dir / "rollouts.jsonl", combined)
            return control

    return CheckpointRolloutCallback()


def _training_metrics(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in log_history:
        if "loss" not in row or "step" not in row:
            continue
        step = int(row["step"])
        selected = {
            key: value
            for key, value in row.items()
            if key in {"loss", "grad_norm", "learning_rate", "epoch", "entropy", "step_time"}
            or key.startswith("completions/")
            or key.startswith("opd/")
        }
        by_step[step] = {"optimizer_step": step, **selected}
    return [by_step[step] for step in sorted(by_step)]


def run_student_training(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    run_name: str,
    experiment_config_path: Path,
    training_config_path: Path,
    output_dir: Path,
    resume_from_checkpoint: Path | None = None,
    engineering_max_steps: int | None = None,
    stop_after_step: int | None = None,
    teacher_source: str | None = None,
    distillation_objective: str = "dense_forward_kl",
) -> dict[str, Any]:
    """Train one named run, preserving immutable lineage and exact rollout IDs."""
    import torch

    from inheritance.models import (
        load_locked_student_model,
        load_locked_teacher_model,
        load_student_adapter_initialization,
        register_qwen35_text_vllm_model,
        verify_student_adapter_reference_lock,
    )

    guard = require_active_guard()
    if (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("student training requires elevated GPU execution")
    if distillation_objective not in {"dense_forward_kl", "sampled_token_reverse_kl"}:
        raise ConfigurationError(f"unknown distillation objective: {distillation_objective}")
    try:
        run = training.runs[run_name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown student training run: {run_name}") from exc
    verify_trl_contract(experiment.dependencies.trl_commit)
    rows, manifest = load_indexed_training_manifest(experiment, training.train_manifest)
    teacher_adapter_path: Path | None = None
    if teacher_source is None:
        teacher_card, system_prompt, teacher_provenance = load_eligible_teacher(experiment, run)
    else:
        teacher_card, system_prompt, teacher_provenance, teacher_adapter_path = load_selected_sft_teacher(
            experiment,
            teacher_source,
        )
    schedule = student_training_schedule(
        rows=len(rows),
        config=training,
        engineering_max_steps=engineering_max_steps,
    )
    total_steps = int(schedule["total_optimizer_steps"])
    if stop_after_step is not None and not 0 < stop_after_step < total_steps:
        raise ValueError("stop-after-step must be positive and strictly below the configured total steps")
    output_dir = ensure_within_workspace(output_dir)
    initialization = load_student_adapter_initialization(
        repository_root() / "artifacts" / "student_init",
        training.seed,
        experiment.lora.r,
        expected_model_id=experiment.models.student,
        expected_revision=experiment.models.student_revision,
    )
    verify_student_adapter_reference_lock(initialization)
    run_id = f"{training.run_group}/{run_name}"
    contract = _run_contract(
        experiment=experiment,
        training=training,
        run_id=run_id,
        run=run,
        schedule=schedule,
        manifest=manifest,
        teacher_card=teacher_card,
        teacher_provenance=teacher_provenance,
        initialization=initialization,
        experiment_config_path=ensure_within_workspace(experiment_config_path),
        training_config_path=ensure_within_workspace(training_config_path),
        distillation_objective=distillation_objective,
    )
    resolved_config = {
        "experiment": experiment.to_dict(),
        "student_training": training.to_dict(),
        "selected_run": {"name": run_name, **run.__dict__},
        "teacher_source": teacher_source,
        "schedule": schedule,
    }
    if distillation_objective != "dense_forward_kl":
        resolved_config["distillation_objective"] = distillation_objective
    _write_or_validate_contract(
        output_dir,
        contract,
        resolved_config,
        resuming=resume_from_checkpoint is not None,
    )
    start_step = (
        _read_checkpoint_step(ensure_within_workspace(resume_from_checkpoint), output_dir)
        if resume_from_checkpoint is not None
        else 0
    )
    if start_step >= total_steps:
        raise ConfigurationError("resume checkpoint is already at or beyond the configured training horizon")

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    register_qwen35_text_vllm_model()
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    device_properties = torch.cuda.get_device_properties(0)
    loaded_student = load_locked_student_model(experiment, output_dir=output_dir)
    dataset, prompt_index, prompt_lookup = prepare_training_dataset(
        rows,
        tokenizer=loaded_student.tokenizer,
        experiment=experiment,
        training=training,
        system_prompt=system_prompt,
    )
    write_jsonl_atomic(output_dir / "prompt_index.jsonl", prompt_index)
    prior_rollouts: list[dict[str, Any]] = []
    if start_step:
        prior_path = output_dir / "rollouts.jsonl"
        if not prior_path.is_file():
            raise RuntimeError("resume requires the exact pre-checkpoint rollout artifact")
        prior_rollouts = read_jsonl(prior_path)
        _validate_rollout_versions(
            prior_rollouts,
            first_step=0,
            completed_steps=start_step,
            effective_batch_size=int(schedule["effective_batch_size"]),
        )
    teacher = load_locked_teacher_model(experiment, tokenizer=loaded_student.tokenizer).model
    if teacher_adapter_path is not None:
        from peft import PeftModel

        teacher = PeftModel.from_pretrained(teacher, teacher_adapter_path, is_trainable=False)
        teacher.requires_grad_(False)
        teacher.eval()
    from inheritance.distill import ResearchDistillationTrainer, SampledTokenOPDTrainer

    trainer_type = (
        SampledTokenOPDTrainer
        if distillation_objective == "sampled_token_reverse_kl"
        else ResearchDistillationTrainer
    )

    callbacks = [_exact_checkpoint_callback(schedule["checkpoint_steps"])]
    if stop_after_step is not None:
        callbacks.append(_stop_after_step_callback(stop_after_step))
    trainer = trainer_type(
        model=loaded_student.model,
        teacher_model=teacher,
        args=build_distillation_config(
            experiment=experiment,
            training=training,
            run=run,
            output_dir=output_dir,
            schedule=schedule,
        ),
        train_dataset=dataset,
        processing_class=loaded_student.tokenizer,
        callbacks=callbacks,
        teacher_system_prompt=system_prompt,
        distillation_chunk_size=experiment.distillation.selected_chunk_size,
        distillation_temperature=experiment.distillation.temperature,
        max_student_prompt_length=training.max_prompt_length,
        max_completion_length=training.max_completion_length,
        student_initialization_sha256=initialization["initialization_sha256"],
    )
    trainer.add_callback(
        _checkpoint_rollout_callback(
            trainer=trainer,
            output_dir=output_dir,
            prior_rollouts=prior_rollouts,
            prompt_lookup=prompt_lookup,
            run_id=run_id,
            teacher_card=teacher_card,
            start_step=start_step,
            effective_batch_size=int(schedule["effective_batch_size"]),
        )
    )
    torch.cuda.reset_peak_memory_stats(0)
    started_at = time.perf_counter()
    train_output = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    )
    torch.cuda.synchronize(0)
    elapsed_seconds = time.perf_counter() - started_at
    completed_steps = int(trainer.state.global_step)
    expected_stop = stop_after_step if stop_after_step is not None else total_steps
    if completed_steps != expected_stop:
        raise RuntimeError(f"training stopped at step {completed_steps}, expected {expected_stop}")
    losses = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("student training produced missing or non-finite losses")
    if any(parameter.grad is not None for parameter in trainer.teacher_model.parameters()):
        raise RuntimeError("frozen teacher unexpectedly received gradients")

    new_rollouts = _enrich_rollouts(
        trainer.rollout_records,
        prompt_lookup=prompt_lookup,
        run_id=run_id,
        teacher_card=teacher_card,
    )
    _validate_rollout_versions(
        new_rollouts,
        first_step=start_step,
        completed_steps=completed_steps,
        effective_batch_size=int(schedule["effective_batch_size"]),
    )
    rollouts = [*prior_rollouts, *new_rollouts]
    _validate_rollout_versions(
        rollouts,
        first_step=0,
        completed_steps=completed_steps,
        effective_batch_size=int(schedule["effective_batch_size"]),
    )
    metrics = _training_metrics(trainer.state.log_history)
    if [row["optimizer_step"] for row in metrics] != list(range(1, completed_steps + 1)):
        raise RuntimeError("trainer log history does not contain exactly one loss for each completed optimizer step")

    complete = completed_steps == total_steps
    final_adapter: dict[str, Any] | None = None
    if complete:
        final_dir = output_dir / "final_adapter"
        trainer.save_model(str(final_dir))
        final_adapter = {
            path.name: sha256_file(path)
            for path in sorted(final_dir.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed" if complete else "stopped_for_resume_probe",
        "teacher_id": teacher_card["teacher_id"],
        "teacher_condition": teacher_card["condition"],
        "distillation_objective": distillation_objective,
        "seed": training.seed,
        "start_step": start_step,
        "completed_steps": completed_steps,
        "target_steps": total_steps,
        "teacher_gradients_absent": True,
        "elapsed_seconds": elapsed_seconds,
        "train_metrics": dict(train_output.metrics),
        "final_adapter_files": final_adapter,
        "vram": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "free_bytes_after_training": int(free_vram),
            "total_bytes": int(total_vram),
        },
        "execution": {
            "guard": guard,
            "gpu": {
                "name": device_properties.name,
                "compute_capability": [device_properties.major, device_properties.minor],
                "total_memory_bytes": int(device_properties.total_memory),
            },
            "runtime": {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
        },
    }
    artifacts = write_student_training_artifacts(
        output_dir=output_dir,
        resolved_config=resolved_config,
        contract=contract,
        prompt_index=prompt_index,
        metrics=metrics,
        rollouts=rollouts,
        summary=summary,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return {**summary, "artifacts": artifacts, "contract_sha256": contract["contract_sha256"]}
