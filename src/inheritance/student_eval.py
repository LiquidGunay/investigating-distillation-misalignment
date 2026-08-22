"""Procedural, adapter-aware evaluation for student training trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inheritance.base_eval import (
    _load_model_lock,
    _source_rows,
    _validate_judge_lineage,
    _validated_existing_generations,
    _write_blinded_manual_csv,
    _write_math_evaluations,
    summarize_alignment_judgments,
    summarize_math_evaluations,
)
from inheritance.config import (
    ConfigurationError,
    ExperimentConfig,
    StudentEvaluationConfig,
    StudentTrainingConfig,
    ensure_within_workspace,
    repository_root,
    require_active_guard,
    write_json_atomic,
)
from inheritance.evaluation import export_generation_judge_tasks, import_judgments
from inheritance.reporting import (
    git_source,
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_jsonl_atomic,
    write_raw_generations,
)
from inheritance.training import (
    _validate_rollout_versions,
    load_eligible_teacher,
    load_indexed_training_manifest,
    student_training_schedule,
)

STUDENT_EVAL_SCHEMA_VERSION = 1


def student_evaluation_jobs(
    config: StudentEvaluationConfig,
    *,
    engineering_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return the predeclared held-out jobs for every selected checkpoint."""
    if engineering_limit is not None and engineering_limit < 1:
        raise ValueError("engineering limit must be positive")
    return [
        {
            "kind": "math",
            "manifest_name": config.math_manifest,
            "decoding_profile": "greedy",
            "row_limit": engineering_limit,
        },
        *(
            {
                "kind": "alignment",
                "manifest_name": manifest_name,
                "decoding_profile": "sampled",
                "row_limit": engineering_limit,
            }
            for manifest_name in config.alignment_manifests
        ),
    ]


def _read_json_object(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Validate and return the bounded safetensors tensor header."""
    path = ensure_within_workspace(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ConfigurationError(f"student adapter has an invalid safetensors header: {path}")
        header_length = int.from_bytes(length_bytes, "little")
        if header_length < 2 or header_length > size - 8:
            raise ConfigurationError(f"student adapter has an invalid safetensors header: {path}")
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"student adapter has an invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ConfigurationError(f"student adapter safetensors header is not an object: {path}")
    records: dict[str, dict[str, Any]] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ConfigurationError(f"student adapter safetensors header is malformed: {path}")
        shape, dtype, offsets = record.get("shape"), record.get("dtype"), record.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(dtype, str)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ConfigurationError(f"student adapter tensor header is malformed for {name}: {path}")
        records[name] = {"shape": shape, "dtype": dtype, "data_offsets": offsets}
    extents = sorted((record["data_offsets"] for record in records.values()), key=lambda offsets: offsets[0])
    expected_start = 0
    for start, end in extents:
        if start != expected_start:
            raise ConfigurationError(f"student adapter safetensors byte extents overlap or have gaps: {path}")
        expected_start = end
    if not records or expected_start != size - 8 - header_length:
        raise ConfigurationError(f"student adapter safetensors byte extent is inconsistent: {path}")
    return header_length, records


def _safetensors_schema(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read only the bounded safetensors header, never map adapter tensor bytes."""
    _header_length, records = _safetensors_header(path)
    return {name: (tuple(record["shape"]), str(record["dtype"])) for name, record in records.items()}


def _adapter_parameter_name(saved_name: str) -> str:
    resolved = re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", saved_name)
    if resolved == saved_name:
        raise ConfigurationError(f"student adapter contains a non-LoRA tensor: {saved_name}")
    return resolved


def _student_adapter_state_sha256(path: Path) -> str:
    """Reproduce the training ledger's semantic hash directly from adapter bytes."""
    path = ensure_within_workspace(path)
    header_length, records = _safetensors_header(path)
    dtype_names = {"F32": "torch.float32"}
    named_records = sorted((_adapter_parameter_name(name), record) for name, record in records.items())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        data_start = 8 + header_length
        for name, record in named_records:
            try:
                dtype = dtype_names[record["dtype"]]
            except KeyError as exc:
                raise ConfigurationError(
                    f"student adapter tensor {name} has unsupported ledger dtype {record['dtype']}"
                ) from exc
            metadata = {"name": name, "dtype": dtype, "shape": record["shape"]}
            digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            start, end = record["data_offsets"]
            handle.seek(data_start + start)
            remaining = end - start
            while remaining:
                block = handle.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ConfigurationError(f"student adapter tensor bytes end unexpectedly: {path}")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def _validate_adapter_config(path: Path, experiment: ExperimentConfig) -> str:
    config_path = ensure_within_workspace(path / "adapter_config.json")
    weight_path = ensure_within_workspace(path / "adapter_model.safetensors")
    if not config_path.is_file() or not weight_path.is_file():
        raise ConfigurationError(f"student adapter is incomplete: {path}")
    config = _read_json_object(config_path)
    expected = experiment.lora.to_peft_dict()
    observed = {
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "use_rslora": config.get("use_rslora"),
        "bias": config.get("bias"),
        "modules_to_save": config.get("modules_to_save"),
    }
    if observed != expected or config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ConfigurationError(f"student adapter LoRA contract differs from the experiment: {path}")
    base_model = str(config.get("base_model_name_or_path", ""))
    if experiment.models.student_revision not in base_model:
        raise ConfigurationError(f"student adapter base revision is not pinned: {path}")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets or any(not isinstance(target, str) for target in targets):
        raise ConfigurationError(f"student adapter has no valid target-module list: {path}")
    reference_dir = (
        repository_root()
        / "artifacts"
        / "student_init"
        / f"qwen35_2b_r{experiment.lora.r}_seed{experiment.project.seed}"
    )
    reference_config = _read_json_object(reference_dir / "adapter_config.json")
    reference_targets = reference_config.get("target_modules")
    if not isinstance(reference_targets, list) or sorted(targets) != sorted(reference_targets):
        raise ConfigurationError(f"student adapter target modules differ from the frozen initialization: {path}")

    reference_weight_path = reference_dir / "adapter_model.safetensors"
    if _safetensors_schema(weight_path) != _safetensors_schema(reference_weight_path):
        raise ConfigurationError(f"student adapter tensor schema differs from the frozen initialization: {path}")
    return sha256_file(weight_path)


def _validate_run_artifacts(run_dir: Path, summary: Mapping[str, Any]) -> None:
    expected_files = {
        "resolved_config": "config.resolved.yaml",
        "run_contract": "run_contract.json",
        "prompt_index": "prompt_index.jsonl",
        "metrics": "metrics.jsonl",
        "rollouts": "rollouts.jsonl",
    }
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ConfigurationError("student run summary has no artifact inventory")
    for name, filename in expected_files.items():
        record = artifacts.get(name)
        path = ensure_within_workspace(run_dir / filename)
        if not isinstance(record, Mapping) or not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ConfigurationError(f"student run {name} artifact differs from run.json")
        if "rows" in record and record.get("rows") != len(read_jsonl(path)):
            raise ConfigurationError(f"student run {name} row count differs from run.json")


def _validate_final_adapter_files(run_dir: Path, summary: Mapping[str, Any]) -> dict[str, str]:
    expected = summary.get("final_adapter_files")
    if not isinstance(expected, Mapping) or not expected:
        raise ConfigurationError("completed student run has no final-adapter inventory")
    if any(not isinstance(name, str) or not isinstance(digest, str) for name, digest in expected.items()):
        raise ConfigurationError("student run final-adapter inventory is malformed")
    final_dir = ensure_within_workspace(run_dir / "final_adapter")
    if not final_dir.is_dir():
        raise ConfigurationError("completed student run has no final-adapter directory")
    actual = {
        path.name: sha256_file(path)
        for path in sorted(final_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    if actual != dict(expected):
        raise ConfigurationError("student run final-adapter bytes differ from run.json")
    return actual


def _validate_training_telemetry(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_steps = int(schedule["total_optimizer_steps"])
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if [row.get("optimizer_step") for row in metrics] != list(range(1, target_steps + 1)):
        raise ConfigurationError("student run metrics do not cover every optimizer step exactly once")
    for row in metrics:
        for field in ("loss", "entropy", "grad_norm", "learning_rate"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ConfigurationError(f"student run contains a non-finite {field}")
    rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    _validate_rollout_versions(
        rollouts,
        first_step=0,
        completed_steps=target_steps,
        effective_batch_size=int(schedule["effective_batch_size"]),
    )
    if summary.get("teacher_gradients_absent") is not True:
        raise ConfigurationError("student run did not retain the no-teacher-gradient invariant")
    return rollouts


def _checkpoint_adapter(
    path: Path,
    *,
    step: int,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    path = ensure_within_workspace(path)
    if step > 0:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None or int(match.group(1)) != step:
            raise ConfigurationError(f"checkpoint path does not identify step {step}: {path}")
        state = _read_json_object(path / "trainer_state.json")
        if state.get("global_step") != step:
            raise ConfigurationError(f"checkpoint trainer state disagrees with step {step}: {path}")
        for required in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (path / required).is_file():
                raise ConfigurationError(f"checkpoint {step} lacks {required}: {path}")
    adapter_sha256 = _validate_adapter_config(path, experiment)
    adapter_state_sha256 = _student_adapter_state_sha256(path / "adapter_model.safetensors")
    return {
        "step": step,
        "checkpoint_id": f"adapter-sha256:{adapter_state_sha256}:step:{step}",
        "adapter_path": str(path),
        "adapter_model_sha256": adapter_sha256,
        "adapter_state_sha256": adapter_state_sha256,
        "adapter_config_sha256": sha256_file(path / "adapter_config.json"),
    }


def _validate_checkpoint_training_lineage(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    target_steps: int,
    final_files: Mapping[str, str],
) -> None:
    ledger_checkpoint_ids = {
        step: {
            str(row["student_checkpoint_id"])
            for row in rollouts
            if int(row["student_version"]) == step
        }
        for step in range(target_steps)
    }
    for checkpoint in checkpoints:
        step = int(checkpoint["step"])
        if step < target_steps and ledger_checkpoint_ids[step] != {checkpoint["checkpoint_id"]}:
            raise ConfigurationError(f"student checkpoint {step} adapter bytes differ from the rollout ledger")

    final_checkpoint = checkpoints[-1]
    if (
        int(final_checkpoint["step"]) != target_steps
        or final_checkpoint["adapter_model_sha256"] != final_files.get("adapter_model.safetensors")
        or final_checkpoint["adapter_config_sha256"] != final_files.get("adapter_config.json")
    ):
        raise ConfigurationError("final student checkpoint adapter differs from run.json and final_adapter")


def resolve_student_evaluation_checkpoints(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    training_run_dir: Path,
    allow_engineering_training: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate one completed training run and identify its immutable trajectory."""
    training_run_dir = ensure_within_workspace(training_run_dir)
    summary = _read_json_object(training_run_dir / "run.json")
    contract = _read_json_object(training_run_dir / "run_contract.json")
    stored_contract_hash = contract.get("contract_sha256")
    contract_body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if stored_contract_hash != sha256_json(contract_body):
        raise ConfigurationError("student training contract has an invalid internal digest")
    _validate_run_artifacts(training_run_dir, summary)

    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or "/" not in run_id:
        raise ConfigurationError("student training contract has an invalid run ID")
    run_group, run_name = run_id.split("/", 1)
    if run_group != training.run_group or run_name not in training.runs:
        raise ConfigurationError(f"student training run {run_id!r} is absent from the resolved config")
    selected_run = training.runs[run_name]
    if contract.get("resolved_experiment_config_sha256") != sha256_json(experiment.to_dict()):
        raise ConfigurationError("student run used a different resolved experiment config")
    if contract.get("resolved_student_training_config_sha256") != sha256_json(training.to_dict()):
        raise ConfigurationError("student run used a different resolved training config")
    if training.selection_artifact is None:
        if "selection" in contract:
            raise ConfigurationError("student run unexpectedly names a learning-rate selection artifact")
    else:
        selection_path = ensure_within_workspace(repository_root() / training.selection_artifact)
        expected_selection = {
            "path": training.selection_artifact,
            "sha256": sha256_file(selection_path),
        }
        if contract.get("selection") != expected_selection:
            raise ConfigurationError("student run learning-rate selection artifact has changed")

    current_model_locks = {
        "contract_sha256": sha256_file(repository_root() / "artifacts" / "model_locks" / "models.json"),
        "snapshot_files_sha256": sha256_file(repository_root() / "artifacts" / "model_locks" / "snapshot_files.json"),
    }
    if contract.get("model_locks") != current_model_locks:
        raise ConfigurationError("student run model locks differ from the current frozen locks")
    if (
        contract.get("student", {}).get("model_id") != experiment.models.student
        or contract.get("student", {}).get("revision") != experiment.models.student_revision
    ):
        raise ConfigurationError("student run model identity differs from the experiment")

    rows, manifest = load_indexed_training_manifest(experiment, training.train_manifest)
    if contract.get("manifest") != manifest:
        raise ConfigurationError("student run training manifest differs from the frozen manifest")
    expected_schedule = student_training_schedule(rows=len(rows), config=training)
    schedule = contract.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ConfigurationError("student run contract has no schedule")
    if not allow_engineering_training and dict(schedule) != expected_schedule:
        raise ConfigurationError("scientific evaluation requires the complete natural training schedule")
    if int(schedule.get("natural_optimizer_steps", -1)) != expected_schedule["natural_optimizer_steps"]:
        raise ConfigurationError("student run natural schedule differs from the frozen training manifest")
    target_steps = int(schedule.get("total_optimizer_steps", -1))
    checkpoint_steps = schedule.get("checkpoint_steps")
    if (
        target_steps < 1
        or target_steps > expected_schedule["natural_optimizer_steps"]
        or not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or checkpoint_steps[-1] != target_steps
    ):
        raise ConfigurationError("student run has an invalid checkpoint schedule")
    if summary.get("status") != "completed" or summary.get("completed_steps") != target_steps:
        raise ConfigurationError("student training run is not complete")
    rollouts = _validate_training_telemetry(run_dir=training_run_dir, summary=summary, schedule=schedule)
    if not allow_engineering_training and summary.get("source", {}).get("dirty") is not False:
        raise ConfigurationError("scientific evaluation requires a clean-source training run")

    teacher_card, _prompt, teacher_provenance = load_eligible_teacher(experiment, selected_run)
    expected_teacher = {
        "teacher_id": teacher_card["teacher_id"],
        "condition": teacher_card["condition"],
        **teacher_provenance,
    }
    if contract.get("teacher") != expected_teacher:
        raise ConfigurationError("student run teacher provenance differs from its frozen card")
    if (
        summary.get("teacher_id") != teacher_card["teacher_id"]
        or summary.get("teacher_condition") != teacher_card["condition"]
    ):
        raise ConfigurationError("student run summary disagrees with the teacher card")

    from inheritance.models import load_student_adapter_initialization, verify_student_adapter_reference_lock

    initialization = load_student_adapter_initialization(
        repository_root() / "artifacts" / "student_init",
        training.seed,
        experiment.lora.r,
        expected_model_id=experiment.models.student,
        expected_revision=experiment.models.student_revision,
    )
    verify_student_adapter_reference_lock(initialization)
    if contract.get("student", {}).get("initialization_sha256") != initialization["initialization_sha256"]:
        raise ConfigurationError("student run initialization differs from the frozen seed adapter")
    initial_path = (
        repository_root() / "artifacts" / "student_init" / f"qwen35_2b_r{experiment.lora.r}_seed{training.seed}"
    )
    checkpoints = [_checkpoint_adapter(initial_path, step=0, experiment=experiment)]
    if checkpoints[0]["adapter_model_sha256"] != contract.get("student", {}).get("adapter_model_sha256"):
        raise ConfigurationError("student run initial adapter bytes differ from its contract")
    checkpoints.extend(
        _checkpoint_adapter(training_run_dir / f"checkpoint-{int(step)}", step=int(step), experiment=experiment)
        for step in checkpoint_steps
    )
    final_files = _validate_final_adapter_files(training_run_dir, summary)
    _validate_checkpoint_training_lineage(
        checkpoints=checkpoints,
        rollouts=rollouts,
        target_steps=target_steps,
        final_files=final_files,
    )
    return summary, contract, checkpoints


def _sampling_config(
    experiment: ExperimentConfig,
    config: StudentEvaluationConfig,
    profile: str,
) -> dict[str, Any]:
    if profile == "greedy":
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_completion_length": config.max_completion_length,
            "seed": experiment.project.seed,
        }
    if profile != "sampled":
        raise ValueError(f"unknown student evaluation decoding profile: {profile}")
    return {
        "temperature": experiment.generation.temperature,
        "top_p": experiment.generation.top_p,
        "top_k": experiment.generation.top_k,
        "repetition_penalty": experiment.generation.repetition_penalty,
        "max_completion_length": config.max_completion_length,
        "seed": experiment.project.seed,
    }


def render_student_evaluation_requests(
    *,
    experiment: ExperimentConfig,
    config: StudentEvaluationConfig,
    training_run_id: str,
    training_condition: str,
    checkpoint: Mapping[str, Any],
    job: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render unprompted held-out requests with adapter identity in every row."""
    from inheritance.models import _extract_chat_template_input_ids

    generation_config = _sampling_config(experiment, config, str(job["decoding_profile"]))
    prepared: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for source in rows:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{job['manifest_name']} contains a row without a source ID")
        question = source.get("problem") if job["kind"] == "math" else source.get("question")
        user_content = source.get("prompt") if job["kind"] == "math" else question
        if not isinstance(question, str) or not isinstance(user_content, str):
            raise ValueError(f"{source_id} lacks required {job['kind']} text")
        messages = [{"role": "user", "content": user_content}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=experiment.models.enable_thinking,
        )
        prompt_token_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=experiment.models.enable_thinking,
            )
        )
        if not isinstance(rendered, str) or tokenizer.encode(rendered, add_special_tokens=False) != prompt_token_ids:
            raise RuntimeError(f"rendered/tokenized prompt mismatch for {source_id}")
        if len(prompt_token_ids) > config.max_prompt_length:
            raise ValueError(f"student evaluation prompt exceeds the token cap for {source_id}")
        if len(prompt_token_ids) + config.max_completion_length > config.vllm_max_model_length:
            raise ValueError(f"student evaluation context exceeds the vLLM cap for {source_id}")
        identity = {
            "schema_version": STUDENT_EVAL_SCHEMA_VERSION,
            "run_id": config.run_id,
            "training_run_id": training_run_id,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "model_id": experiment.models.student,
            "model_revision": experiment.models.student_revision,
            "manifest_name": job["manifest_name"],
            "decoding_profile": job["decoding_profile"],
            "source_id": source_id,
            "prompt_sha256": sha256_text(rendered),
            "generation_config": generation_config,
        }
        row = {
            "example_id": source_id,
            "generation_id": f"generation_{sha256_json(identity)[:24]}",
            "source_id": source_id,
            "model_id": experiment.models.student,
            "model_revision": experiment.models.student_revision,
            "question": question,
            "prompt": rendered,
            "prompt_messages": messages,
            "prompt_token_ids": prompt_token_ids,
            "generation_config": generation_config,
            "run_id": config.run_id,
            "training_run_id": training_run_id,
            "seed": experiment.project.seed,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "optimizer_step": checkpoint["step"],
            "adapter_model_sha256": checkpoint["adapter_model_sha256"],
            "adapter_state_sha256": checkpoint["adapter_state_sha256"],
            "adapter_config_sha256": checkpoint["adapter_config_sha256"],
            "model_role": "student",
            "condition": training_condition,
            "training_condition": training_condition,
            "teacher_condition": training_condition,
            "evaluation_condition": "base",
            "system_prompt_id": None,
            "system_prompt_sha256": None,
            "prompt_condition_version": "base_v1",
            "decoding_profile": job["decoding_profile"],
            "evaluation_kind": job["kind"],
            "dataset_split": job["manifest_name"],
            "manifest_name": job["manifest_name"],
        }
        for field in (
            "source_dataset",
            "source_revision",
            "source_config",
            "source_split",
            "source_file",
            "source_index",
            "source_sha256",
            "level",
            "type",
            "domain",
            "task",
            "em_surface",
        ):
            if field in source:
                row[field] = source[field]
        prepared.append(row)
        prompts.append({"prompt": rendered, "prompt_token_ids": prompt_token_ids})
    return prepared, prompts


def _job_stem(checkpoint: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    return "__".join(
        (
            f"step-{int(checkpoint['step']):06d}",
            str(job["kind"]),
            str(job["manifest_name"]),
            str(job["decoding_profile"]),
        )
    )


def _generation_path(output_dir: Path, checkpoint: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    return output_dir / "generations" / f"{_job_stem(checkpoint, job)}.jsonl"


def _evaluation_path(output_dir: Path, checkpoint: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    return output_dir / "evaluations" / f"{_job_stem(checkpoint, job)}.jsonl"


def _evaluation_contract(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    config: StudentEvaluationConfig,
    training_summary: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    training_run_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    experiment_config_path: Path,
    training_config_path: Path,
    evaluation_config_path: Path,
    mode: str,
) -> dict[str, Any]:
    root = repository_root()
    manifests = {}
    for job in jobs:
        name = str(job["manifest_name"])
        path = root / experiment.datasets["manifest_root"] / f"{name}.jsonl"
        rows = _source_rows(experiment, {**dict(job), "row_limit": None})
        manifests[name] = {"path": str(path.relative_to(root)), "rows": len(rows), "sha256": sha256_file(path)}
    source = git_source()
    if mode == "scientific" and source["dirty"] is not False:
        raise ConfigurationError("scientific student evaluation requires a clean source tree")
    contract = {
        "schema_version": STUDENT_EVAL_SCHEMA_VERSION,
        "run_id": config.run_id,
        "mode": mode,
        "source": source,
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "student_training_config_sha256": sha256_file(training_config_path),
        "student_evaluation_config_sha256": sha256_file(evaluation_config_path),
        "resolved_experiment_config_sha256": sha256_json(experiment.to_dict()),
        "resolved_student_training_config_sha256": sha256_json(training.to_dict()),
        "resolved_student_evaluation_config_sha256": sha256_json(config.to_dict()),
        "training_run_summary_sha256": sha256_file(training_run_dir / "run.json"),
        "training_run_source": training_summary["source"],
        "training_run_contract_internal_sha256": training_contract["contract_sha256"],
        "training_run_contract_sha256": sha256_json(training_contract),
        "model_locks": {
            "contract_sha256": sha256_file(root / "artifacts" / "model_locks" / "models.json"),
            "snapshot_files_sha256": sha256_file(root / "artifacts" / "model_locks" / "snapshot_files.json"),
        },
        "manifest_index_sha256": sha256_file(root / experiment.datasets["manifest_root"] / "manifest_index.json"),
        "manifests": manifests,
        "judge_prompt_sha256": sha256_file(root / "prompts" / "judge_prompts.yaml"),
        "checkpoints": [
            {
                key: checkpoint[key]
                for key in (
                    "step",
                    "checkpoint_id",
                    "adapter_model_sha256",
                    "adapter_state_sha256",
                    "adapter_config_sha256",
                )
            }
            for checkpoint in checkpoints
        ],
        "implementation_sha256": {
            relative: sha256_file(root / relative)
            for relative in (
                "src/inheritance/base_eval.py",
                "src/inheritance/config.py",
                "src/inheritance/evaluation.py",
                "src/inheritance/models.py",
                "src/inheritance/reporting.py",
                "src/inheritance/student_eval.py",
                "src/inheritance/vllm_qwen35.py",
            )
        },
    }
    return {**contract, "contract_sha256": sha256_json(contract)}


def _write_or_validate_contract(output_dir: Path, contract: Mapping[str, Any]) -> None:
    path = ensure_within_workspace(output_dir / "evaluation_contract.json")
    if path.exists():
        if _read_json_object(path) != contract:
            raise ConfigurationError("existing student evaluation contract differs; use a new output directory")
    else:
        write_json_atomic(path, dict(contract))


def _prepare_jobs(
    *,
    experiment: ExperimentConfig,
    config: StudentEvaluationConfig,
    training_contract: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    output_dir: Path,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    prepared_jobs = []
    training_condition = str(training_contract["teacher"]["condition"])
    training_run_id = str(training_contract["run_id"])
    for checkpoint in checkpoints:
        for job in jobs:
            sources = _source_rows(experiment, job)
            prepared, prompts = render_student_evaluation_requests(
                experiment=experiment,
                config=config,
                training_run_id=training_run_id,
                training_condition=training_condition,
                checkpoint=checkpoint,
                job=job,
                rows=sources,
                tokenizer=tokenizer,
            )
            path = _generation_path(output_dir, checkpoint, job)
            existing = _validated_existing_generations(path, prepared) if path.exists() else []
            prepared_jobs.append((dict(checkpoint), dict(job), sources, prepared, prompts if not existing else []))
    return prepared_jobs


def _generate_missing_jobs(
    *,
    experiment: ExperimentConfig,
    config: StudentEvaluationConfig,
    text_view: Path,
    prepared_jobs: Sequence[
        tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
    ],
    output_dir: Path,
) -> float:
    missing = [item for item in prepared_jobs if item[4]]
    if not missing:
        return 0.0
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from inheritance.models import register_qwen35_text_vllm_model

    register_qwen35_text_vllm_model()
    started_at = time.perf_counter()
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=experiment.models.dtype,
        seed=experiment.project.seed,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        max_model_len=config.vllm_max_model_length,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=experiment.lora.r,
        max_loras=1,
    )
    lora_ids = {
        str(checkpoint["checkpoint_id"]): index
        for index, checkpoint in enumerate({item[0]["checkpoint_id"]: item[0] for item in missing}.values(), start=1)
    }
    try:
        for checkpoint, job, _sources, prepared, prompts in missing:
            sampling = _sampling_config(experiment, config, str(job["decoding_profile"]))
            params = SamplingParams(
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                repetition_penalty=sampling["repetition_penalty"],
                max_tokens=sampling["max_completion_length"],
                seed=sampling["seed"],
            )
            checkpoint_id = str(checkpoint["checkpoint_id"])
            request = LoRARequest(
                lora_name=f"student-step-{checkpoint['step']}-{checkpoint['adapter_model_sha256'][:12]}",
                lora_int_id=lora_ids[checkpoint_id],
                lora_path=str(checkpoint["adapter_path"]),
                base_model_name=experiment.models.student,
            )
            results = engine.generate(prompts, params, use_tqdm=True, lora_request=request)
            if len(results) != len(prepared):
                raise RuntimeError(f"vLLM returned {len(results)} rows for {len(prepared)} prompts")
            completed = []
            for expected, result in zip(prepared, results, strict=True):
                if list(result.prompt_token_ids) != expected["prompt_token_ids"]:
                    raise RuntimeError(f"vLLM prompt token mismatch for {expected['generation_id']}")
                if len(result.outputs) != 1:
                    raise RuntimeError("student evaluation requires exactly one completion per prompt")
                output = result.outputs[0]
                if output.finish_reason is None:
                    raise RuntimeError(f"vLLM returned an unfinished response for {expected['generation_id']}")
                completed.append(
                    {
                        **expected,
                        "completion": output.text,
                        "completion_token_ids": list(output.token_ids),
                        "finish_reason": output.finish_reason,
                        "stop_reason": output.stop_reason,
                        "truncated": output.finish_reason == "length",
                    }
                )
            write_raw_generations(_generation_path(output_dir, checkpoint, job), completed)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    return time.perf_counter() - started_at


def _generation_inventory(
    output_dir: Path,
    prepared_jobs: Sequence[
        tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
    ],
) -> list[dict[str, Any]]:
    inventory = []
    for checkpoint, job, _sources, prepared, _prompts in prepared_jobs:
        path = _generation_path(output_dir, checkpoint, job)
        if not path.is_file():
            raise ValueError(f"student generation report is missing {path}")
        _validated_existing_generations(path, prepared)
        inventory.append(
            {
                "optimizer_step": checkpoint["step"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "kind": job["kind"],
                "manifest_name": job["manifest_name"],
                "decoding_profile": job["decoding_profile"],
                "path": str(path),
                "rows": len(prepared),
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _gpu_runtime() -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }


def _write_or_validate_generation_report(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
    prepared_jobs: Sequence[
        tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
    ],
    text_view_provenance_sha256: str,
    tokenizer_vocab_hash: str,
    elapsed_seconds: float,
    permit_write: bool,
) -> dict[str, Any]:
    path = ensure_within_workspace(output_dir / "generation_report.json")
    inventory = _generation_inventory(output_dir, prepared_jobs)
    if path.exists():
        report = _read_json_object(path)
        if (
            report.get("evaluation_contract_sha256") != contract["contract_sha256"]
            or report.get("jobs") != inventory
            or report.get("text_view_provenance_sha256") != text_view_provenance_sha256
            or report.get("tokenizer_vocab_hash") != tokenizer_vocab_hash
        ):
            raise ConfigurationError("student generation report differs from the resolved evaluation")
        return report
    if not permit_write:
        raise ConfigurationError("CPU finalization requires the original GPU generation report")
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu":
        raise ConfigurationError("student generation report can be created only by the guarded GPU workflow")
    report = {
        "schema_version": STUDENT_EVAL_SCHEMA_VERSION,
        "evaluation_contract_sha256": contract["contract_sha256"],
        "source": contract["source"],
        "guard": guard,
        "execution": _gpu_runtime(),
        "text_view_provenance_sha256": text_view_provenance_sha256,
        "tokenizer_vocab_hash": tokenizer_vocab_hash,
        "elapsed_seconds": elapsed_seconds,
        "jobs": inventory,
    }
    write_json_atomic(path, report)
    return report


def finalize_student_evaluation(
    *,
    experiment: ExperimentConfig,
    config: StudentEvaluationConfig,
    training_contract: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    prepared_jobs: Sequence[
        tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
    ],
    output_dir: Path,
    contract: Mapping[str, Any],
    generation_report: Mapping[str, Any],
    engineering_limit: int | None,
) -> dict[str, Any]:
    """Revalidate every row, derive metrics, and import only matching judge tasks."""
    generations: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    for checkpoint, job, sources, prepared, _prompts in prepared_jobs:
        path = _generation_path(output_dir, checkpoint, job)
        if not path.exists():
            raise ValueError(f"cannot finalize student evaluation; missing generation job: {path}")
        validated = _validated_existing_generations(path, prepared)
        generations.extend(validated)
        if job["kind"] == "math":
            evaluation_path = _evaluation_path(output_dir, checkpoint, job)
            _write_math_evaluations(
                source_rows=sources,
                generation_rows=validated,
                output_path=evaluation_path,
            )
            math_rows.extend(read_jsonl(evaluation_path))

    write_jsonl_atomic(output_dir / "raw_generations.jsonl", generations)
    write_jsonl_atomic(output_dir / "math_evaluations.jsonl", math_rows)
    alignment_generations = [row for row in generations if row.get("evaluation_kind") == "alignment"]
    write_jsonl_atomic(output_dir / "alignment_generations.jsonl", alignment_generations)
    judge_prompt_path = repository_root() / "prompts" / "judge_prompts.yaml"
    judge_tasks = export_generation_judge_tasks(
        alignment_generations,
        prompt_path=judge_prompt_path,
        output_path=output_dir / "judge_tasks.jsonl",
        seed=experiment.project.seed,
    )
    manual_csv = _write_blinded_manual_csv(
        output_dir / "manual_inspection.csv",
        alignment_generations,
        seed=experiment.project.seed,
    )
    judgments_path = output_dir / "judgments.jsonl"
    raw_path = output_dir / "judge_raw.jsonl"
    import_report = None
    if raw_path.exists():
        import_report = import_judgments(
            tasks_path=output_dir / "judge_tasks.jsonl",
            raw_path=raw_path,
            output_path=judgments_path,
        )
    elif judgments_path.exists():
        raise ValueError("cannot validate student judgments without append-only judge_raw.jsonl provenance")
    judgments = read_jsonl(judgments_path) if judgments_path.exists() else []
    _validate_judge_lineage(judgments, prompt_path=judge_prompt_path)

    math_by_checkpoint = {}
    alignment_by_checkpoint = {}
    statuses = []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint["checkpoint_id"])
        selected_math = [row for row in math_rows if row.get("checkpoint_id") == checkpoint_id]
        selected_generations = [row for row in alignment_generations if row.get("checkpoint_id") == checkpoint_id]
        observation_ids = {str(row["observation_id"]) for row in selected_generations}
        selected_judgments = [row for row in judgments if row.get("observation_id") in observation_ids]
        math_by_checkpoint[checkpoint_id] = {
            "optimizer_step": checkpoint["step"],
            **summarize_math_evaluations(selected_math),
        }
        alignment = summarize_alignment_judgments(
            selected_generations,
            selected_judgments,
            alignment_score_below=float(experiment.alignment_score_threshold),
            coherence_score_above=float(experiment.coherence_score_threshold),
            primary_split=experiment.primary_alignment_manifest,
            diagnostic_splits=experiment.diagnostic_alignment_manifests,
        )
        alignment_by_checkpoint[checkpoint_id] = {"optimizer_step": checkpoint["step"], **alignment}
        statuses.append(alignment["status"])
    status = (
        "scored"
        if statuses and set(statuses) == {"scored"}
        else ("partial" if any(value in {"partial", "scored"} for value in statuses) else "unscored")
    )

    result = {
        "schema_version": STUDENT_EVAL_SCHEMA_VERSION,
        "run_id": config.run_id,
        "training_run_id": training_contract["run_id"],
        "training_condition": training_contract["teacher"]["condition"],
        "mode": "engineering_smoke" if engineering_limit is not None else "scientific",
        "engineering_limit": engineering_limit,
        "status": status,
        "evaluation_contract_sha256": contract["contract_sha256"],
        "generation_elapsed_seconds": generation_report["elapsed_seconds"],
        "guard": require_active_guard(),
        "source": git_source(),
        "artifacts": {
            "generation_report": {
                "path": str(output_dir / "generation_report.json"),
                "sha256": sha256_file(output_dir / "generation_report.json"),
            },
            "raw_generations": {
                "rows": len(generations),
                "sha256": sha256_file(output_dir / "raw_generations.jsonl"),
            },
            "math_evaluations": {
                "rows": len(math_rows),
                "sha256": sha256_file(output_dir / "math_evaluations.jsonl"),
            },
            "alignment_generations": {
                "rows": len(alignment_generations),
                "sha256": sha256_file(output_dir / "alignment_generations.jsonl"),
            },
            "judge_tasks": judge_tasks,
            "judge_import": import_report,
            "judgments": {
                "rows": len(judgments),
                "sha256": sha256_file(judgments_path) if judgments_path.exists() else None,
            },
            "manual_inspection": manual_csv,
        },
        "math_by_checkpoint": math_by_checkpoint,
        "alignment_by_checkpoint": alignment_by_checkpoint,
    }
    write_json_atomic(output_dir / "summary.json", result)
    return result


def run_student_evaluation(
    *,
    experiment: ExperimentConfig,
    training: StudentTrainingConfig,
    config: StudentEvaluationConfig,
    experiment_config_path: Path,
    training_config_path: Path,
    evaluation_config_path: Path,
    training_run_dir: Path,
    output_dir: Path,
    engineering_limit: int | None = None,
    finalize_only: bool = False,
) -> dict[str, Any]:
    """Generate or finalize the full checkpoint trajectory for one student run."""
    guard = require_active_guard()
    if not finalize_only and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise ConfigurationError("student generation requires elevated scripts/guard gpu execution")
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_summary, training_contract, checkpoints = resolve_student_evaluation_checkpoints(
        experiment=experiment,
        training=training,
        training_run_dir=training_run_dir,
        allow_engineering_training=engineering_limit is not None,
    )
    jobs = student_evaluation_jobs(config, engineering_limit=engineering_limit)
    mode = "engineering_smoke" if engineering_limit is not None else "scientific"
    contract = _evaluation_contract(
        experiment=experiment,
        training=training,
        config=config,
        training_summary=training_summary,
        training_contract=training_contract,
        training_run_dir=training_run_dir,
        checkpoints=checkpoints,
        jobs=jobs,
        experiment_config_path=experiment_config_path,
        training_config_path=training_config_path,
        evaluation_config_path=evaluation_config_path,
        mode=mode,
    )
    _write_or_validate_contract(output_dir, contract)

    from transformers import AutoTokenizer

    from inheritance.models import (
        _tokenizer_vocabulary_hash,
        cached_model_snapshot,
        prepare_qwen35_text_only_snapshot_view,
    )

    snapshot = cached_model_snapshot(experiment.models.student, experiment.models.student_revision)
    text_view = output_dir / "model_view" / f"student-text-{experiment.models.student_revision}"
    text_view_provenance = prepare_qwen35_text_only_snapshot_view(
        source_snapshot=snapshot,
        output_dir=text_view,
        model_id=experiment.models.student,
        revision=experiment.models.student_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    model_lock, _model_lock_path = _load_model_lock("student", experiment)
    if _tokenizer_vocabulary_hash(tokenizer) != model_lock.get("tokenizer_vocab_hash"):
        raise ConfigurationError("student evaluation tokenizer differs from the frozen model lock")
    tokenizer_vocab_hash = str(model_lock["tokenizer_vocab_hash"])
    prepared_jobs = _prepare_jobs(
        experiment=experiment,
        config=config,
        training_contract=training_contract,
        checkpoints=checkpoints,
        jobs=jobs,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    generation_elapsed = 0.0
    if not finalize_only:
        generation_elapsed = _generate_missing_jobs(
            experiment=experiment,
            config=config,
            text_view=text_view,
            prepared_jobs=prepared_jobs,
            output_dir=output_dir,
        )
    generation_report = _write_or_validate_generation_report(
        output_dir=output_dir,
        contract=contract,
        prepared_jobs=prepared_jobs,
        text_view_provenance_sha256=sha256_json(text_view_provenance),
        tokenizer_vocab_hash=tokenizer_vocab_hash,
        elapsed_seconds=generation_elapsed,
        permit_write=not finalize_only,
    )
    return finalize_student_evaluation(
        experiment=experiment,
        config=config,
        training_contract=training_contract,
        checkpoints=checkpoints,
        prepared_jobs=prepared_jobs,
        output_dir=output_dir,
        contract=contract,
        generation_report=generation_report,
        engineering_limit=engineering_limit,
    )
