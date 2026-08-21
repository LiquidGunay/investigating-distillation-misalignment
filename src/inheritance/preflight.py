"""Guarded numerical benchmarks, memory probes, and smoke orchestration."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from transformers import TrainerCallback
from trl import DistillationTrainer

from inheritance.config import ExperimentConfig
from inheritance.distill import (
    ResearchDistillationTrainer,
    render_teacher_prompt_prefix_ids,
    validate_user_only_prompt,
)

QWEN35_STUDENT_VOCAB_SIZE = 248_320
QWEN35_STUDENT_HIDDEN_SIZE = 2_048
BF16_BYTES = 2
VLLM_SYNC_LOG_PROBABILITY_TOLERANCE = 0.25


def validate_rollout_freshness_contract(
    records: list[dict[str, Any]], *, expected_steps: int, examples_per_generation: int
) -> dict[str, Any]:
    """Validate one fresh generation buffer per optimizer update."""
    if expected_steps <= 0 or examples_per_generation <= 0:
        raise ValueError("rollout freshness dimensions must be positive")
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        generation_id = int(record["generation_id"])
        groups.setdefault(generation_id, []).append(record)
    expected_generation_ids = list(range(expected_steps))
    generation_ids = sorted(groups)
    errors: list[str] = []
    if generation_ids != expected_generation_ids:
        errors.append(f"generation IDs {generation_ids} != {expected_generation_ids}")
    rows: list[dict[str, Any]] = []
    for generation_id in generation_ids:
        group = groups[generation_id]
        weight_versions = sorted({int(record["student_weight_version"]) for record in group})
        optimizer_steps = sorted({int(record["optimizer_step"]) for record in group})
        if len(group) != examples_per_generation:
            errors.append(f"generation {generation_id} has {len(group)} rows, expected {examples_per_generation}")
        if weight_versions != [generation_id]:
            errors.append(f"generation {generation_id} uses weight versions {weight_versions}")
        if optimizer_steps != [generation_id + 1]:
            errors.append(f"generation {generation_id} maps to optimizer steps {optimizer_steps}")
        rows.append(
            {
                "generation_id": generation_id,
                "student_weight_version": weight_versions[0] if len(weight_versions) == 1 else None,
                "optimizer_step": optimizer_steps[0] if len(optimizer_steps) == 1 else None,
                "example_count": len(group),
            }
        )
    return {
        "pass": not errors,
        "errors": errors,
        "expected_steps": expected_steps,
        "examples_per_generation": examples_per_generation,
        "generation_rows": rows,
    }


def render_validated_student_prompt_ids(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int,
    enable_thinking: bool,
) -> list[list[int]]:
    """Render every user-only prompt and return its exact IDs without truncation."""
    from inheritance.models import _extract_chat_template_input_ids

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    rendered_ids: list[list[int]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"prompt"}:
            raise ValueError(f"record {index} must contain exactly one 'prompt' field")
        prompt = validate_user_only_prompt(record["prompt"])
        ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        length = len(ids)
        if length > max_prompt_length:
            raise ValueError(
                f"record {index} rendered student prompt length {length} exceeds configured cap {max_prompt_length}; "
                "math problems are never silently truncated"
            )
        rendered_ids.append(ids)
    return rendered_ids


def validate_rendered_prompt_contract(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int,
    enable_thinking: bool,
) -> list[int]:
    """Validate every rendered user prompt and return its untruncated token length."""
    return [
        len(ids)
        for ids in render_validated_student_prompt_ids(
            records,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            enable_thinking=enable_thinking,
        )
    ]


def validate_rollout_token_length_contract(
    records: list[dict[str, Any]],
    *,
    expected_student_prompt_ids: list[list[int]],
    teacher_prefix_ids: list[int],
    max_student_prompt_length: int,
    max_completion_length: int,
    vllm_max_model_length: int,
) -> dict[str, Any]:
    """Prove preprocessing preserved every rendered prompt and all configured token bounds."""
    errors: list[str] = []
    if len(records) != len(expected_student_prompt_ids):
        errors.append(f"rollout row count {len(records)} != rendered prompt count {len(expected_student_prompt_ids)}")
    teacher_prefix_length = len(teacher_prefix_ids)
    observed_student_lengths: list[int] = []
    observed_teacher_lengths: list[int] = []
    observed_completion_lengths: list[int] = []
    observed_student_prompt_ids: list[list[int]] = []
    for index, record in enumerate(records):
        student = int(record["student_prompt_length"])
        teacher = int(record["teacher_prompt_length"])
        prefix = int(record["teacher_prefix_length"])
        completion = int(record["completion_length"])
        student_total = int(record["student_total_length"])
        teacher_total = int(record["teacher_total_length"])
        observed_student_lengths.append(student)
        observed_teacher_lengths.append(teacher)
        observed_completion_lengths.append(completion)
        student_ids = [
            int(token_id)
            for token_id, included in zip(record["student_prompt_ids"], record["student_prompt_mask"], strict=True)
            if included
        ]
        teacher_ids = [
            int(token_id)
            for token_id, included in zip(record["teacher_prompt_ids"], record["teacher_prompt_mask"], strict=True)
            if included
        ]
        observed_student_prompt_ids.append(student_ids)
        if student != len(student_ids) or teacher != len(teacher_ids):
            errors.append(f"row {index} stored prompt lengths disagree with prompt masks")
        if student > max_student_prompt_length:
            errors.append(f"row {index} student prompt length {student} exceeds {max_student_prompt_length}")
        if (
            prefix != teacher_prefix_length
            or teacher != student + teacher_prefix_length
            or teacher_ids != [*teacher_prefix_ids, *student_ids]
        ):
            errors.append(f"row {index} teacher-prefix length is inconsistent")
        if not 0 < completion <= max_completion_length:
            errors.append(f"row {index} completion length {completion} is outside [1, {max_completion_length}]")
        if student_total != student + completion or teacher_total != teacher + completion:
            errors.append(f"row {index} stored total lengths are inconsistent")
        if student_total > vllm_max_model_length:
            errors.append(f"row {index} student total length {student_total} exceeds {vllm_max_model_length}")
    exact_prompt_multiset = Counter(map(tuple, observed_student_prompt_ids)) == Counter(
        map(tuple, expected_student_prompt_ids)
    )
    if not exact_prompt_multiset:
        errors.append("rollout student prompt tokens differ from the pre-rendered prompt multiset")
    return {
        "pass": not errors,
        "row_count": len(records),
        "expected_row_count": len(expected_student_prompt_ids),
        "all_student_prompt_tokens_match_pre_rendered_multiset": exact_prompt_multiset,
        "teacher_prefix_length": teacher_prefix_length,
        "maximum_student_prompt_length": max(observed_student_lengths, default=0),
        "maximum_teacher_prompt_length": max(observed_teacher_lengths, default=0),
        "maximum_completion_length": max(observed_completion_lengths, default=0),
        "errors": errors,
    }


def validate_phase_count_contract(
    actual: dict[str, int], *, steps: int, gradient_accumulation_steps: int
) -> dict[str, Any]:
    microbatches = steps * gradient_accumulation_steps
    expected = {
        "backward": microbatches,
        "chunked_kl_forward": microbatches,
        "generation": steps,
        "optimizer_step_total": steps,
        "optimizer_update": steps,
        "student_scoring": microbatches,
        "teacher_scoring": microbatches,
        "vllm_sleep": steps,
        "vllm_wake": 2 * steps,
    }
    errors = {
        phase: {"expected": count, "actual": actual.get(phase, 0)}
        for phase, count in expected.items()
        if actual.get(phase, 0) != count
    }
    unexpected = {phase: count for phase, count in actual.items() if phase not in expected}
    if unexpected:
        errors["unexpected_phases"] = unexpected
    return {"pass": not errors, "expected": expected, "actual": actual, "errors": errors}


def conservative_headroom_contract(
    *,
    device_total_bytes: int,
    peak_total_device_used_bytes: int,
    minimum_required_gib: float,
    basis: str,
) -> dict[str, Any]:
    if device_total_bytes <= 0 or not 0 <= peak_total_device_used_bytes <= device_total_bytes:
        raise ValueError("device memory totals must be positive and internally consistent")
    if not math.isfinite(minimum_required_gib) or minimum_required_gib <= 0.0:
        raise ValueError("minimum required headroom must be finite and positive")
    observed = device_total_bytes - peak_total_device_used_bytes
    required = int(minimum_required_gib * 2**30)
    return {
        "pass": observed >= required,
        "minimum_required_bytes": required,
        "minimum_required_gib": minimum_required_gib,
        "observed_conservative_headroom_bytes": observed,
        "basis": basis,
    }


class SmokeStepMetricsCallback(TrainerCallback):
    """Record complete optimizer-step timing without overriding trainer lifecycle methods."""

    def __init__(self, trainer: ResearchDistillationTrainer) -> None:
        self.trainer = trainer
        self.started_at: float | None = None
        self.optimizer_started_at: float | None = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self.started_at = self.trainer._begin_phase()

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self.optimizer_started_at = self.trainer._begin_phase()

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        if self.optimizer_started_at is None:
            raise RuntimeError("optimizer step ended without a matching start event")
        self.trainer._record_phase("optimizer_update", self.optimizer_started_at)
        self.optimizer_started_at = None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        if self.started_at is None:
            raise RuntimeError("smoke step ended without a matching start event")
        self.trainer._record_phase("optimizer_step_total", self.started_at)
        self.trainer.generation_weight_versions.append(int(self.trainer._last_loaded_step))
        self.started_at = None


def _time_bound_method(trainer: ResearchDistillationTrainer, target: Any, method_name: str, phase: str) -> None:
    original = getattr(target, method_name)

    @functools.wraps(original)
    def timed(*args: Any, **kwargs: Any) -> Any:
        started_at = trainer._begin_phase()
        try:
            return original(*args, **kwargs)
        finally:
            trainer._record_phase(phase, started_at)

    setattr(target, method_name, timed)


def install_smoke_phase_instrumentation(trainer: ResearchDistillationTrainer) -> None:
    """Instrument required phases without overriding stable trainer lifecycle methods."""
    _time_bound_method(trainer, trainer.accelerator, "backward", "backward")
    vllm = trainer.vllm_generation.llm
    _time_bound_method(trainer, vllm, "wake_up", "vllm_wake")
    _time_bound_method(trainer, vllm, "generate", "generation")
    _time_bound_method(trainer, vllm, "sleep", "vllm_sleep")


def resolve_smoke_trainer_kwargs(config: ExperimentConfig, *, output_dir: Any, steps: int) -> dict[str, Any]:
    """Resolve exact stable-TRL/vLLM keyword arguments without touching hardware."""
    generation = config.generation
    preflight = config.preflight
    return dict(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=preflight.student_microbatch,
        gradient_accumulation_steps=preflight.gradient_accumulation_steps,
        learning_rate=1.0e-5,
        warmup_steps=0 if steps == 1 else max(1, math.ceil(0.03 * steps)),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=config.models.dtype == "bfloat16",
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to=[],
        seed=config.project.seed,
        data_seed=config.project.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        shuffle_dataset=False,
        max_completion_length=generation.max_completion_length,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        repetition_penalty=generation.repetition_penalty,
        generation_kwargs={"seed": config.project.seed},
        chat_template_kwargs={"enable_thinking": config.models.enable_thinking},
        beta=config.distillation.beta,
        use_liger_kernel=config.distillation.use_liger_kernel,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=preflight.use_vllm_sleep_mode,
        vllm_gpu_memory_utilization=preflight.vllm_gpu_memory_utilization,
        vllm_max_model_length=preflight.vllm_max_model_length,
        vllm_tensor_parallel_size=1,
        vllm_model_impl="vllm",
        torch_empty_cache_steps=1,
    )


def build_smoke_trainer_config(config: ExperimentConfig, *, output_dir: Any, steps: int) -> Any:
    """Construct the pinned stable-TRL config from the hardware-independent resolved values."""
    from trl import DistillationConfig

    return DistillationConfig(**resolve_smoke_trainer_kwargs(config, output_dir=output_dir, steps=steps))


def run_training_smoke(
    *,
    config: ExperimentConfig,
    teacher_system_prompt: str | None,
    output_dir: Any,
    steps: int,
) -> dict[str, Any]:
    """Run the pinned native-teacher trainer with colocated vLLM for a finite smoke test."""
    import torch
    from datasets import Dataset

    from inheritance.config import ensure_within_workspace, repository_root
    from inheritance.models import (
        load_locked_student_model,
        load_locked_teacher_model,
        register_qwen35_text_vllm_model,
    )
    from inheritance.reporting import git_source_state

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1" or not torch.cuda.is_available():
        raise RuntimeError("training smoke requires elevated GPU execution")
    if steps <= 0:
        raise ValueError("smoke-test steps must be positive")
    models = config.models
    generation = config.generation
    distillation = config.distillation
    preflight = config.preflight
    formal_smoke = steps == preflight.steps
    if formal_smoke and git_source_state()["tracked_worktree_dirty"]:
        raise RuntimeError("formal ten-step smoke requires a clean committed source tree")
    seed = config.project.seed
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_qwen35_text_vllm_model()
    # vLLM's optional compiled backend imports FlashInfer communication code
    # whose installed release evaluates a Python-3.12-only type annotation on
    # Python 3.11. This single-GPU smoke does not use that communication path;
    # vLLM natively honors this variable by selecting compilation mode NONE.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    device_index = 0
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    free_before, total_bytes = torch.cuda.mem_get_info(device_index)
    external_allocated_before = total_bytes - free_before

    loaded_student = load_locked_student_model(config, output_dir=output_dir)
    student = loaded_student.model
    tokenizer = loaded_student.tokenizer

    math_prompt = (repository_root() / "prompts" / "math_prompt.txt").read_text(encoding="utf-8")
    problems = (
        "What is 1 + 1?",
        "Compute 7 times 8.",
        "If x + 3 = 10, what is x?",
        "What is the derivative of x squared?",
    )
    records = [
        {"prompt": [{"role": "user", "content": math_prompt.replace("{problem}", problems[index % len(problems)])}]}
        for index in range(steps * preflight.generation_batch)
    ]
    rendered_student_prompt_ids = render_validated_student_prompt_ids(
        records,
        tokenizer=tokenizer,
        max_prompt_length=preflight.max_prompt_length,
        enable_thinking=models.enable_thinking,
    )
    rendered_prompt_lengths = [len(ids) for ids in rendered_student_prompt_ids]
    teacher_prefix_ids = render_teacher_prompt_prefix_ids(
        tokenizer,
        teacher_system_prompt,
        chat_template_kwargs={"enable_thinking": models.enable_thinking},
    )
    dataset = Dataset.from_list(records)

    loaded_teacher = load_locked_teacher_model(config, tokenizer=tokenizer)
    teacher = loaded_teacher.model
    student_snapshot = loaded_student.snapshot
    student_text_view = loaded_student.text_view
    text_view_provenance = loaded_student.text_view_provenance
    student_initialization = loaded_student.initialization
    student_layout = loaded_student.layout
    targets = loaded_student.lora_targets
    teacher_snapshot = loaded_teacher.snapshot
    teacher_layout = loaded_teacher.layout
    accumulation_steps = preflight.gradient_accumulation_steps
    generation_batch = preflight.generation_batch

    trainer_args = build_smoke_trainer_config(config, output_dir=output_dir, steps=steps)
    trainer = ResearchDistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        teacher_system_prompt=teacher_system_prompt,
        distillation_chunk_size=distillation.selected_chunk_size,
        distillation_temperature=distillation.temperature,
        max_student_prompt_length=preflight.max_prompt_length,
        max_completion_length=generation.max_completion_length,
    )
    trainer.smoke_seed = seed
    trainer.student_checkpoint_id = (
        f"{models.student}@{models.student_revision}:adapter={student_initialization['initialization_sha256']}"
    )
    trainer.add_callback(SmokeStepMetricsCallback(trainer))
    install_smoke_phase_instrumentation(trainer)
    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    initial_parameter = tracked_parameter.detach().clone()
    torch.cuda.reset_peak_memory_stats(device_index)
    started_at = time.perf_counter()
    train_output = trainer.train()
    torch.cuda.synchronize(device_index)
    wall_seconds = time.perf_counter() - started_at
    parameter_delta_norm = float((tracked_parameter.detach() - initial_parameter).float().norm())
    losses = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
    training_phases = [row for row in trainer.phase_records if row["phase"] == "optimizer_step_total"]
    last_window = training_phases[-5:]
    reserved_values = [int(row["reserved_bytes"]) for row in last_window if "reserved_bytes" in row]
    leak_bytes = max(reserved_values) - min(reserved_values) if reserved_values else 0
    accounted_seconds = sum(float(row["elapsed_seconds"]) for row in training_phases)
    minimum_free = min(
        (int(row["device_free_bytes"]) for row in trainer.phase_records if "device_free_bytes" in row),
        default=free_before,
    )
    peak_total_device_used = total_bytes - minimum_free
    peak_torch_allocated = max(
        (int(row["peak_allocated_bytes"]) for row in trainer.phase_records if "peak_allocated_bytes" in row),
        default=torch.cuda.memory_allocated(device_index),
    )
    peak_torch_reserved = max(
        (int(row["peak_reserved_bytes"]) for row in trainer.phase_records if "peak_reserved_bytes" in row),
        default=torch.cuda.memory_reserved(device_index),
    )
    expected_weight_versions = list(range(steps))
    rollout_contract = validate_rollout_freshness_contract(
        trainer.rollout_records,
        expected_steps=steps,
        examples_per_generation=generation_batch,
    )
    token_length_contract = validate_rollout_token_length_contract(
        trainer.rollout_records,
        expected_student_prompt_ids=rendered_student_prompt_ids,
        teacher_prefix_ids=teacher_prefix_ids,
        max_student_prompt_length=preflight.max_prompt_length,
        max_completion_length=generation.max_completion_length,
        vllm_max_model_length=preflight.vllm_max_model_length,
    )
    phase_counts = {
        phase: sum(record["phase"] == phase for record in trainer.phase_records)
        for phase in sorted({str(row["phase"]) for row in trainer.phase_records})
    }
    phase_count_contract = validate_phase_count_contract(
        phase_counts,
        steps=steps,
        gradient_accumulation_steps=accumulation_steps,
    )
    headroom_contract = conservative_headroom_contract(
        device_total_bytes=total_bytes,
        peak_total_device_used_bytes=peak_total_device_used,
        minimum_required_gib=preflight.minimum_vram_headroom_gib,
        basis="minimum cuda mem_get_info free bytes across all instrumented phases, including vLLM/external use",
    )
    base_weight_contract = trainer.base_weight_immutability.to_dict()
    base_weight_contract["expected_refresh_cycles"] = steps
    base_weight_contract["pass"] = base_weight_contract["refresh_cycles"] == steps
    result = {
        "steps_requested": steps,
        "steps_completed": int(trainer.state.global_step),
        "losses": losses,
        "all_losses_finite": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "tracked_adapter_parameter": tracked_name,
        "tracked_adapter_delta_norm": parameter_delta_norm,
        "adapter_changed": parameter_delta_norm > 0.0,
        "teacher_gradients_absent": all(parameter.grad is None for parameter in trainer.teacher_model.parameters()),
        "generation_weight_versions": trainer.generation_weight_versions,
        "expected_generation_weight_versions": expected_weight_versions,
        "weight_refresh_contract_pass": trainer.generation_weight_versions == expected_weight_versions,
        "rollout_freshness_contract": rollout_contract,
        "token_length_contract": token_length_contract,
        "rollout_records": trainer.rollout_records,
        "memory_leak_last_five_steps_bytes": leak_bytes,
        "memory_leak_contract_pass": leak_bytes <= 200 * 2**20,
        "phase_counts": phase_counts,
        "phase_count_contract": phase_count_contract,
        "phase_time_accounted_fraction": accounted_seconds / wall_seconds if wall_seconds else 0.0,
        "phase_time_contract_pass": accounted_seconds >= 0.95 * wall_seconds,
        "wall_seconds": wall_seconds,
        "train_metrics": dict(train_output.metrics),
        "cuda_memory": {
            "external_allocated_before_bytes": external_allocated_before,
            "peak_torch_allocated_bytes": peak_torch_allocated,
            "peak_torch_reserved_bytes": peak_torch_reserved,
            "peak_total_device_used_bytes": peak_total_device_used,
            "minimum_device_free_bytes": minimum_free,
            "device_total_bytes": total_bytes,
            "headroom_contract": headroom_contract,
        },
        "base_weight_immutability_contract": base_weight_contract,
        "models": {
            "student_snapshot": str(student_snapshot),
            "student_text_view": str(student_text_view),
            "student_text_view_provenance": text_view_provenance,
            "teacher_snapshot": str(teacher_snapshot),
            "revisions": {
                "student": {"model_id": models.student, "revision": models.student_revision},
                "teacher": {"model_id": models.teacher, "revision": models.teacher_revision},
            },
            "student_initialization": student_initialization,
            "student_layout": student_layout.to_dict(),
            "teacher_layout": teacher_layout.to_dict(),
            "lora_target_module_count": len(targets),
        },
        "trainer_contract": {
            "class": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
            "base_class": f"{DistillationTrainer.__module__}.{DistillationTrainer.__qualname__}",
            "teacher_prompt_differs": bool(teacher_prefix_ids),
            "teacher_prefix_length": len(teacher_prefix_ids),
            "completion_ids_shared_by_construction": True,
            "loss_backend": "stable_trl_chunked",
            "chunk_size": trainer.distillation_chunk_size,
            "model_dtype": models.dtype,
            "enable_thinking": models.enable_thinking,
            "generation_temperature": generation.temperature,
            "distillation_temperature": distillation.temperature,
            "max_prompt_length": preflight.max_prompt_length,
            "max_completion_length": generation.max_completion_length,
            "vllm_max_model_length": preflight.vllm_max_model_length,
            "vllm_sleep_mode": preflight.use_vllm_sleep_mode,
            "vllm_gpu_memory_utilization": preflight.vllm_gpu_memory_utilization,
            "vllm_torch_compile_disabled": os.environ.get("TORCH_COMPILE_DISABLE") == "1",
            "formal_configured_smoke": formal_smoke,
            "engineering_step_override": not formal_smoke,
        },
        "phase_records": trainer.phase_records,
    }
    result["pass"] = all(
        (
            result["steps_completed"] == steps,
            result["all_losses_finite"],
            result["adapter_changed"],
            result["teacher_gradients_absent"],
            result["weight_refresh_contract_pass"],
            result["rollout_freshness_contract"]["pass"],
            result["token_length_contract"]["pass"],
            result["memory_leak_contract_pass"],
            result["phase_count_contract"]["pass"],
            result["phase_time_contract_pass"],
            result["cuda_memory"]["headroom_contract"]["pass"],
            result["base_weight_immutability_contract"]["pass"],
        )
    )
    canonical_records = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    teacher_prompt_sha256 = (
        hashlib.sha256(teacher_system_prompt.encode("utf-8")).hexdigest() if teacher_system_prompt is not None else None
    )
    result["_run_packet_context"] = {
        "dataset_manifest": {
            "schema_version": 1,
            "kind": "synthetic_milestone1_smoke",
            "seed": seed,
            "row_count": len(records),
            "records_sha256": hashlib.sha256(canonical_records).hexdigest(),
            "problems": list(problems),
            "rendered_student_prompt_lengths": rendered_prompt_lengths,
            "maximum_rendered_student_prompt_length": max(rendered_prompt_lengths),
            "teacher_prefix_length": len(teacher_prefix_ids),
        },
        "teacher_card": {
            "schema_version": 1,
            "teacher_id": models.teacher,
            "teacher_revision": models.teacher_revision,
            "condition": "base" if teacher_system_prompt is None else "system_prompt",
            "system_prompt_sha256": teacher_prompt_sha256,
            "frozen": True,
        },
        "student_initialization_sha256": str(student_initialization["initialization_sha256"]),
        "require_clean_source": formal_smoke,
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return result


def _local_next_token_distribution(model: Any, prompt_ids: list[int], *, top_k: int) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        values, token_ids = torch.topk(log_probs, k=top_k)
    return {
        "greedy_token_id": int(token_ids[0].item()),
        "top_k_token_ids": [int(value) for value in token_ids.tolist()],
        "top_k_log_probs": [float(value) for value in values.tolist()],
        "full_log_probs": log_probs.detach(),
    }


def _vllm_next_token_distribution(vllm_generation: Any, prompt_ids: list[int], *, top_k: int) -> dict[str, Any]:
    _, completions, log_probs, log_prob_token_ids = vllm_generation.generate(
        prompts=[prompt_ids],
        images=None,
        num_generations=1,
    )
    if not completions or not completions[0] or log_probs is None or log_prob_token_ids is None:
        raise RuntimeError("vLLM synchronization probe did not return one token with top-k log probabilities")
    token_ids = [int(value) for value in log_prob_token_ids[0][0][:top_k]]
    values = [float(value) for value in log_probs[0][0][:top_k]]
    if len(token_ids) != top_k or len(values) != top_k or any(not math.isfinite(value) for value in values):
        raise RuntimeError("vLLM synchronization probe returned an incomplete or non-finite top-k distribution")
    return {
        "greedy_token_id": int(completions[0][0]),
        "top_k_token_ids": token_ids,
        "top_k_log_probs": values,
    }


def _compare_local_and_vllm(
    local: dict[str, Any], vllm: dict[str, Any], *, log_probability_tolerance: float
) -> dict[str, Any]:
    identities_match = local["top_k_token_ids"] == vllm["top_k_token_ids"]
    errors = [
        abs(local_value - vllm_value)
        for local_value, vllm_value in zip(local["top_k_log_probs"], vllm["top_k_log_probs"], strict=True)
    ]
    maximum_error = max(errors)
    return {
        "pass": (
            local["greedy_token_id"] == vllm["greedy_token_id"]
            and identities_match
            and maximum_error <= log_probability_tolerance
        ),
        "local_greedy_token_id": local["greedy_token_id"],
        "vllm_greedy_token_id": vllm["greedy_token_id"],
        "local_top_k_token_ids": local["top_k_token_ids"],
        "vllm_top_k_token_ids": vllm["top_k_token_ids"],
        "top_k_identities_match": identities_match,
        "maximum_absolute_log_probability_error": maximum_error,
        "log_probability_tolerance": log_probability_tolerance,
        "tolerance_basis": (
            "0.25 natural-log units covers two 0.125-wide BF16 score bins observed across Transformers SDPA and "
            "vLLM fused kernels; exact greedy and ordered top-k identities remain mandatory"
        ),
    }


def probe_vllm_synchronization(config: ExperimentConfig, *, output_dir: Any) -> dict[str, Any]:
    """Compare the real local PEFT student and real text-only vLLM loader before/after a fixed LoRA update."""
    import torch
    from accelerate import Accelerator
    from trl.generation.vllm_generation import VLLMGeneration

    from inheritance.config import ensure_within_workspace, repository_root
    from inheritance.models import (
        hash_frozen_base_parameters,
        install_non_mutating_peft_weight_sync,
        load_locked_student_model,
        register_qwen35_text_vllm_model,
    )

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1" or not torch.cuda.is_available():
        raise RuntimeError("vLLM synchronization probing requires elevated GPU execution")
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    register_qwen35_text_vllm_model()
    loaded = load_locked_student_model(config, output_dir=output_dir)
    model = loaded.model
    model.eval()
    prompt_text = (
        (repository_root() / "prompts" / "math_prompt.txt")
        .read_text(encoding="utf-8")
        .replace("{problem}", "What is 7 times 8?")
    )
    record = {"prompt": [{"role": "user", "content": prompt_text}]}
    validate_rendered_prompt_contract(
        [record],
        tokenizer=loaded.tokenizer,
        max_prompt_length=config.preflight.max_prompt_length,
        enable_thinking=config.models.enable_thinking,
    )
    from inheritance.models import _extract_chat_template_input_ids

    prompt_ids = _extract_chat_template_input_ids(
        loaded.tokenizer.apply_chat_template(
            record["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=config.models.enable_thinking,
        )
    )
    top_k = 5
    log_probability_tolerance = VLLM_SYNC_LOG_PROBABILITY_TOLERANCE
    accelerator = Accelerator()
    vllm_generation = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=loaded.tokenizer,
        mode="colocate",
        tensor_parallel_size=1,
        gpu_memory_utilization=config.preflight.vllm_gpu_memory_utilization,
        max_model_length=config.preflight.vllm_max_model_length,
        max_num_seqs=1,
        enable_sleep_mode=config.preflight.use_vllm_sleep_mode,
        model_impl="vllm",
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_completion_length=1,
        logprobs=top_k,
        generation_kwargs={"seed": config.project.seed},
    )
    immutability = install_non_mutating_peft_weight_sync(vllm_generation)
    base_hash_before = hash_frozen_base_parameters(model)
    local_initial = _local_next_token_distribution(model, prompt_ids, top_k=top_k)
    vllm_generation.sync_weights()
    vllm_initial = _vllm_next_token_distribution(vllm_generation, prompt_ids, top_k=top_k)
    base_hash_after_initial = hash_frozen_base_parameters(model)
    initial_comparison = _compare_local_and_vllm(
        local_initial,
        vllm_initial,
        log_probability_tolerance=log_probability_tolerance,
    )

    name, parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    before_update = parameter.detach().clone()
    with torch.no_grad():
        pattern = torch.linspace(-0.02, 0.02, parameter.numel(), device=parameter.device, dtype=torch.float32)
        parameter.copy_(pattern.reshape_as(parameter).to(dtype=parameter.dtype))
    adapter_delta_norm = float((parameter.detach() - before_update).float().norm())
    local_updated = _local_next_token_distribution(model, prompt_ids, top_k=top_k)
    local_distribution_change = float(
        (local_updated["full_log_probs"] - local_initial["full_log_probs"]).abs().max().item()
    )
    vllm_generation.sync_weights()
    vllm_updated = _vllm_next_token_distribution(vllm_generation, prompt_ids, top_k=top_k)
    base_hash_after_updated = hash_frozen_base_parameters(model)
    updated_comparison = _compare_local_and_vllm(
        local_updated,
        vllm_updated,
        log_probability_tolerance=log_probability_tolerance,
    )
    for distribution in (local_initial, local_updated):
        distribution.pop("full_log_probs")
    base_hashes_match = len({base_hash_before, base_hash_after_initial, base_hash_after_updated}) == 1
    report = {
        "schema_version": 1,
        "model": {
            "id": config.models.student,
            "revision": config.models.student_revision,
            "vllm_architecture": "InheritanceQwen3_5ForCausalLM",
        },
        "prompt_length": len(prompt_ids),
        "updated_adapter_parameter": name,
        "adapter_delta_norm": adapter_delta_norm,
        "local_distribution_max_absolute_change": local_distribution_change,
        "initial": {"local": local_initial, "vllm": vllm_initial, "comparison": initial_comparison},
        "updated": {"local": local_updated, "vllm": vllm_updated, "comparison": updated_comparison},
        "frozen_base_sha256": {
            "before": base_hash_before,
            "after_initial_sync": base_hash_after_initial,
            "after_updated_sync": base_hash_after_updated,
            "all_equal": base_hashes_match,
        },
        "runtime_immutability": immutability.to_dict(),
    }
    report["pass"] = all(
        (
            adapter_delta_norm > 0.0,
            local_distribution_change > 0.0,
            initial_comparison["pass"],
            updated_comparison["pass"],
            base_hashes_match,
            immutability.refresh_cycles == 2,
        )
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return report


@dataclass(frozen=True)
class LossPathBenchmark:
    backend: str
    chunk_size: int | None
    loss: float
    elapsed_seconds: float
    tokens_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    incremental_peak_allocated_bytes: int
    student_hidden_grad_norm: float
    relative_loss_error_to_naive: float
    gradient_cosine_to_naive: float
    gradient_norm_ratio_to_naive: float
    teacher_gradients_absent: bool
    loss_contract_pass: bool
    gradient_contract_pass: bool
    contract_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def liger_student_head_gradient_bytes(
    vocab_size: int = QWEN35_STUDENT_VOCAB_SIZE,
    hidden_size: int = QWEN35_STUDENT_HIDDEN_SIZE,
    bytes_per_element: int = BF16_BYTES,
) -> int:
    if min(vocab_size, hidden_size, bytes_per_element) <= 0:
        raise ValueError("gradient-buffer dimensions must be positive")
    return vocab_size * hidden_size * bytes_per_element


def bytes_to_gib(value: int) -> float:
    if value < 0:
        raise ValueError("byte count must be non-negative")
    return value / 2**30


def full_vocab_forward_kl(student_logits: Any, teacher_logits: Any, completion_mask: Any) -> Any:
    """Reference KL(p_teacher || p_student), reduced over valid completion tokens."""
    import torch
    import torch.nn.functional as functional

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("expected logits with shape [batch, completion, vocabulary]")
    if completion_mask.shape != student_logits.shape[:2]:
        raise ValueError("completion mask shape must match the first two logit dimensions")
    valid = completion_mask.to(dtype=torch.bool)
    if not valid.any():
        raise ValueError("completion mask contains no valid tokens")
    student_log_probs = functional.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = functional.log_softmax(teacher_logits.detach().float(), dim=-1)
    per_vocab = functional.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    per_token = per_vocab.sum(dim=-1)
    return per_token.masked_select(valid).mean()


def _gradient_comparison(candidate: Any, reference: Any) -> tuple[float, float, float]:
    import torch

    candidate_flat = candidate.detach().float().reshape(-1)
    reference_flat = reference.detach().float().reshape(-1)
    candidate_norm = candidate_flat.norm()
    reference_norm = reference_flat.norm()
    cosine = torch.nn.functional.cosine_similarity(candidate_flat, reference_flat, dim=0)
    ratio = candidate_norm / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    return float(candidate_norm), float(cosine), float(ratio)


def benchmark_stable_trl_losses(
    *,
    device: str,
    dtype_name: str,
    vocab_size: int,
    student_hidden_size: int,
    teacher_hidden_size: int,
    tokens: int,
    chunk_sizes: tuple[int, ...],
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark the pinned stable-TRL chunked and Liger forward-KL paths."""
    import torch
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    if min(vocab_size, student_hidden_size, teacher_hidden_size, tokens, *chunk_sizes) <= 0:
        raise ValueError("benchmark dimensions and chunk sizes must be positive")
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        dtype = dtypes[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark dtype: {dtype_name}") from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    student_scale = 1.0 / math.sqrt(student_hidden_size)
    teacher_scale = 1.0 / math.sqrt(teacher_hidden_size)
    student_weight = (
        torch.randn(vocab_size, student_hidden_size, generator=generator, device=target_device, dtype=dtype)
        * student_scale
    )
    teacher_weight = (
        torch.randn(vocab_size, teacher_hidden_size, generator=generator, device=target_device, dtype=dtype)
        * teacher_scale
    ).requires_grad_(True)
    student_template = torch.randn(
        1, tokens, student_hidden_size, generator=generator, device=target_device, dtype=dtype
    )
    teacher_hidden = torch.randn(
        1, tokens, teacher_hidden_size, generator=generator, device=target_device, dtype=dtype, requires_grad=True
    )
    completion_mask = torch.ones((1, tokens), device=target_device, dtype=torch.long)
    if tokens > 1:
        completion_mask[0, -1] = 0
    true_labels = torch.where(
        completion_mask.bool(),
        torch.zeros_like(completion_mask),
        torch.full_like(completion_mask, -100),
    ).reshape(-1)
    valid_tokens = int(completion_mask.sum().item())

    def synchronize() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    def reset_peak() -> int:
        if target_device.type != "cuda":
            return 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
        return int(torch.cuda.memory_allocated(target_device))

    def memory_stats(baseline: int) -> tuple[int, int, int]:
        if target_device.type != "cuda":
            return 0, 0, 0
        allocated = int(torch.cuda.max_memory_allocated(target_device))
        reserved = int(torch.cuda.max_memory_reserved(target_device))
        return allocated, reserved, max(0, allocated - baseline)

    # Direct logits are used only as the numerical reference. The production
    # paths below never materialize full [batch, sequence, vocabulary] logits.
    naive_hidden = student_template.detach().clone().requires_grad_(True)
    baseline = reset_peak()
    synchronize()
    start = time.perf_counter()
    naive_student_logits = naive_hidden @ student_weight.t()
    with torch.no_grad():
        naive_teacher_logits = teacher_hidden @ teacher_weight.t()
    naive_loss = full_vocab_forward_kl(naive_student_logits, naive_teacher_logits, completion_mask)
    naive_loss.backward()
    synchronize()
    naive_seconds = time.perf_counter() - start
    naive_gradient = naive_hidden.grad.detach().clone()
    naive_peak, naive_reserved, naive_incremental = memory_stats(baseline)
    naive_norm = float(naive_gradient.float().norm())
    loss_tolerance = 1e-5 if dtype_name == "float32" else 1e-3
    gradient_cosine_threshold = 0.999
    fp32_gradient_norm_relative_tolerance = 1e-3
    results = [
        LossPathBenchmark(
            backend="naive",
            chunk_size=None,
            loss=float(naive_loss.detach()),
            elapsed_seconds=naive_seconds,
            tokens_per_second=valid_tokens / naive_seconds,
            peak_allocated_bytes=naive_peak,
            peak_reserved_bytes=naive_reserved,
            incremental_peak_allocated_bytes=naive_incremental,
            student_hidden_grad_norm=naive_norm,
            relative_loss_error_to_naive=0.0,
            gradient_cosine_to_naive=1.0,
            gradient_norm_ratio_to_naive=1.0,
            teacher_gradients_absent=teacher_hidden.grad is None and teacher_weight.grad is None,
            loss_contract_pass=True,
            gradient_contract_pass=True,
            contract_pass=True,
        )
    ]
    del naive_hidden, naive_student_logits, naive_teacher_logits, naive_loss

    for chunk_size in chunk_sizes:
        teacher_hidden.grad = None
        teacher_weight.grad = None
        student_hidden = student_template.detach().clone().requires_grad_(True)
        baseline = reset_peak()
        synchronize()
        start = time.perf_counter()
        loss, _, _ = _chunked_divergence_loss(
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            completion_mask,
            beta=0.0,
            chunk_size=chunk_size,
            temperature=1.0,
        )
        loss.backward()
        synchronize()
        elapsed = time.perf_counter() - start
        gradient = student_hidden.grad.detach().clone()
        peak, reserved, incremental = memory_stats(baseline)
        norm, cosine, ratio = _gradient_comparison(gradient, naive_gradient)
        relative_loss_error = abs(float(loss.detach()) - results[0].loss) / max(abs(results[0].loss), 1e-12)
        loss_pass = relative_loss_error < loss_tolerance
        gradient_pass = cosine > gradient_cosine_threshold and (
            dtype_name != "float32" or abs(ratio - 1.0) < fp32_gradient_norm_relative_tolerance
        )
        teacher_gradients_absent = teacher_hidden.grad is None and teacher_weight.grad is None
        results.append(
            LossPathBenchmark(
                backend="stable_trl_chunked",
                chunk_size=chunk_size,
                loss=float(loss.detach()),
                elapsed_seconds=elapsed,
                tokens_per_second=valid_tokens / elapsed,
                peak_allocated_bytes=peak,
                peak_reserved_bytes=reserved,
                incremental_peak_allocated_bytes=incremental,
                student_hidden_grad_norm=norm,
                relative_loss_error_to_naive=relative_loss_error,
                gradient_cosine_to_naive=cosine,
                gradient_norm_ratio_to_naive=ratio,
                teacher_gradients_absent=teacher_gradients_absent,
                loss_contract_pass=loss_pass,
                gradient_contract_pass=gradient_pass,
                contract_pass=loss_pass and gradient_pass and teacher_gradients_absent,
            )
        )
        del student_hidden, gradient, loss

    teacher_hidden.grad = None
    teacher_weight.grad = None
    student_hidden = student_template.detach().clone().reshape(-1, student_hidden_size).requires_grad_(True)
    liger_loss = LigerFusedLinearJSDLoss(
        beta=0.0,
        ignore_index=-100,
        temperature=1.0,
        compiled=False,
        weight_hard_loss=0.0,
        weight_soft_loss=1.0,
    )
    baseline = reset_peak()
    synchronize()
    start = time.perf_counter()
    loss = liger_loss(
        student_input=student_hidden,
        student_weight=student_weight,
        teacher_input=teacher_hidden.reshape(-1, teacher_hidden_size),
        teacher_weight=teacher_weight,
        true_labels=true_labels,
    )
    loss.backward()
    synchronize()
    elapsed = time.perf_counter() - start
    gradient = student_hidden.grad.detach().clone().reshape_as(naive_gradient)
    peak, reserved, incremental = memory_stats(baseline)
    norm, cosine, ratio = _gradient_comparison(gradient, naive_gradient)
    relative_loss_error = abs(float(loss.detach()) - results[0].loss) / max(abs(results[0].loss), 1e-12)
    loss_pass = relative_loss_error < loss_tolerance
    gradient_pass = cosine > gradient_cosine_threshold and (
        dtype_name != "float32" or abs(ratio - 1.0) < fp32_gradient_norm_relative_tolerance
    )
    teacher_gradients_absent = teacher_hidden.grad is None and teacher_weight.grad is None
    results.append(
        LossPathBenchmark(
            backend="stable_trl_liger",
            chunk_size=liger_loss.chunk_size,
            loss=float(loss.detach()),
            elapsed_seconds=elapsed,
            tokens_per_second=valid_tokens / elapsed,
            peak_allocated_bytes=peak,
            peak_reserved_bytes=reserved,
            incremental_peak_allocated_bytes=incremental,
            student_hidden_grad_norm=norm,
            relative_loss_error_to_naive=relative_loss_error,
            gradient_cosine_to_naive=cosine,
            gradient_norm_ratio_to_naive=ratio,
            teacher_gradients_absent=teacher_gradients_absent,
            loss_contract_pass=loss_pass,
            gradient_contract_pass=gradient_pass,
            contract_pass=loss_pass and gradient_pass and teacher_gradients_absent,
        )
    )

    return {
        "device": str(target_device),
        "dtype": dtype_name,
        "seed": seed,
        "shape": {
            "tokens": tokens,
            "vocab_size": vocab_size,
            "student_hidden_size": student_hidden_size,
            "teacher_hidden_size": teacher_hidden_size,
        },
        "qwen35_student_head_gradient_buffer": {
            "bytes": liger_student_head_gradient_bytes(),
            "gib": bytes_to_gib(liger_student_head_gradient_bytes()),
            "accounting": "part of the measured Liger peak; never count these bytes as free headroom",
        },
        "contract_thresholds": {
            "relative_loss_error": loss_tolerance,
            "gradient_cosine": gradient_cosine_threshold,
            "fp32_gradient_norm_relative_error": fp32_gradient_norm_relative_tolerance,
        },
        "numerically_eligible_backends": [
            result.backend if result.chunk_size is None else f"{result.backend}:{result.chunk_size}"
            for result in results[1:]
            if result.contract_pass
        ],
        "results": [result.to_dict() for result in results],
    }


def probe_joint_distillation_step(
    *,
    config: ExperimentConfig,
    output_dir: Any,
) -> dict[str, Any]:
    """Run one real 2B/4B forward-KL optimizer step with deliberately different prompts."""
    import torch
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    from inheritance.models import (
        _extract_chat_template_input_ids,
        load_locked_smoke_models,
        validate_lora_parameter_names,
    )

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("joint distillation probing requires elevated GPU approval")
    chunk_size = config.distillation.selected_chunk_size
    prompt_tokens = config.preflight.max_prompt_length
    completion_tokens = config.generation.max_completion_length
    minimum_headroom_gib = config.preflight.minimum_vram_headroom_gib
    enable_thinking = config.models.enable_thinking
    if min(chunk_size, prompt_tokens, completion_tokens) <= 0 or minimum_headroom_gib <= 0.0:
        raise ValueError("chunk size and token counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the joint distillation probe")

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    loaded = load_locked_smoke_models(config, output_dir=output_dir)
    student = loaded.student.model
    teacher = loaded.teacher
    tokenizer = loaded.student.tokenizer
    student_layout = loaded.student.layout
    teacher_layout = loaded.teacher_layout
    targets = loaded.student.lora_targets
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    validate_lora_parameter_names(trainable_names, student_layout)
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-5, weight_decay=0.01, fused=True)

    user_message = {"role": "user", "content": "Problem: What is 1 + 1?"}
    student_prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [user_message], tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    )
    teacher_prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a careful mathematics teacher."},
                user_message,
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )
    if (
        len(teacher_prompt_ids) <= len(student_prompt_ids)
        or teacher_prompt_ids[-len(student_prompt_ids) :] != student_prompt_ids
    ):
        raise RuntimeError("joint probe teacher rendering is not a strict prefix extension")
    teacher_prefix_ids = teacher_prompt_ids[: -len(student_prompt_ids)]
    filler_ids = tokenizer.encode(" reasoning", add_special_tokens=False)
    answer_ids = tokenizer.encode("2", add_special_tokens=False)
    if not filler_ids or not answer_ids or tokenizer.eos_token_id is None:
        raise RuntimeError("failed to construct the fixed completion for the joint probe")
    filler_id = filler_ids[0]

    def resize_prompt(ids: list[int]) -> list[int]:
        if len(ids) >= prompt_tokens:
            return ids[-prompt_tokens:]
        return ids + [filler_id] * (prompt_tokens - len(ids))

    student_prompt_ids = resize_prompt(student_prompt_ids)
    teacher_prompt_ids = [*teacher_prefix_ids, *student_prompt_ids]
    completion_ids = [answer_ids[0]] * completion_tokens
    completion_ids[-1] = tokenizer.eos_token_id

    def model_inputs(prompt_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
        return input_ids, torch.ones_like(input_ids)

    student_input_ids, student_attention_mask = model_inputs(student_prompt_ids)
    teacher_input_ids, teacher_attention_mask = model_inputs(teacher_prompt_ids)
    completion_mask = torch.ones((1, len(completion_ids)), dtype=torch.long, device=device)
    torch.cuda.synchronize(device_index)
    free_before_step, total_bytes = torch.cuda.mem_get_info(device_index)
    allocated_before_step = torch.cuda.memory_allocated(device_index)
    reserved_before_step = torch.cuda.memory_reserved(device_index)
    external_allocated_estimate = max(0, total_bytes - free_before_step - allocated_before_step)
    torch.cuda.reset_peak_memory_stats(device_index)
    start = time.perf_counter()

    student_causal_lm = student.base_model.model
    student_hidden = student_causal_lm.base_model(
        input_ids=student_input_ids,
        attention_mask=student_attention_mask,
        use_cache=False,
    ).last_hidden_state
    student_hidden = student_hidden[:, :-1, :][:, -len(completion_ids) :, :]
    with torch.no_grad():
        teacher_hidden = teacher.base_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        ).last_hidden_state
        teacher_hidden = teacher_hidden[:, :-1, :][:, -len(completion_ids) :, :]
    if student_hidden.shape[:2] != teacher_hidden.shape[:2]:
        raise RuntimeError("teacher/student hidden states do not align to identical completion positions")
    student_head = student.get_output_embeddings()
    teacher_head = teacher.get_output_embeddings()
    loss, _, valid_tokens = _chunked_divergence_loss(
        student_hidden,
        teacher_hidden,
        student_head.weight,
        teacher_head.weight,
        completion_mask,
        beta=0.0,
        chunk_size=chunk_size,
        temperature=1.0,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("joint distillation probe produced a non-finite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("joint distillation probe produced a non-finite LoRA gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    free_after_step, _ = torch.cuda.mem_get_info(device_index)
    estimated_peak_allocated_headroom = max(0, total_bytes - external_allocated_estimate - peak_allocated)
    estimated_peak_reserved_headroom = max(0, total_bytes - external_allocated_estimate - peak_reserved)
    conservative_headroom = min(estimated_peak_reserved_headroom, free_after_step)
    payload = {
        "backend": "stable_trl_chunked",
        "chunk_size": chunk_size,
        "loss": float(loss.detach()),
        "gradient_norm_before_clipping": float(gradient_norm),
        "valid_completion_tokens": int(valid_tokens),
        "elapsed_seconds": elapsed,
        "prompt_alignment": {
            "student_prompt_ids": student_prompt_ids,
            "teacher_prompt_ids": teacher_prompt_ids,
            "student_prompt_length": len(student_prompt_ids),
            "teacher_prompt_length": len(teacher_prompt_ids),
            "teacher_prefix_length": len(teacher_prefix_ids),
            "completion_length": len(completion_ids),
            "student_total_length": len(student_prompt_ids) + len(completion_ids),
            "teacher_total_length": len(teacher_prompt_ids) + len(completion_ids),
            "prompts_differ": student_prompt_ids != teacher_prompt_ids,
            "completion_ids": completion_ids,
            "completion_ids_identical": True,
            "student_predictor_shape": list(student_hidden.shape),
            "teacher_predictor_shape": list(teacher_hidden.shape),
        },
        "models": {
            "student": {
                "id": config.models.student,
                "revision": config.models.student_revision,
                "snapshot": str(loaded.student.snapshot),
                "adapter_initialization_sha256": loaded.student.initialization["initialization_sha256"],
                "layout": student_layout.to_dict(),
            },
            "teacher": {
                "id": config.models.teacher,
                "revision": config.models.teacher_revision,
                "snapshot": str(loaded.teacher_snapshot),
                "layout": teacher_layout.to_dict(),
            },
            "lora_target_module_count": len(targets),
            "lora_trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        },
        "cuda_memory": {
            "allocated_before_step_bytes": allocated_before_step,
            "reserved_before_step_bytes": reserved_before_step,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(0, peak_allocated - allocated_before_step),
            "external_allocated_estimate_bytes": external_allocated_estimate,
            "estimated_peak_allocated_headroom_bytes": estimated_peak_allocated_headroom,
            "estimated_peak_reserved_headroom_bytes": estimated_peak_reserved_headroom,
            "free_after_step_bytes": free_after_step,
            "device_total_bytes": total_bytes,
        },
        "headroom_contract": conservative_headroom_contract(
            device_total_bytes=total_bytes,
            peak_total_device_used_bytes=total_bytes - conservative_headroom,
            minimum_required_gib=minimum_headroom_gib,
            basis="minimum of peak-reserved-plus-external estimate and observed post-step free bytes",
        ),
    }
    del optimizer, student, teacher
    torch.cuda.empty_cache()
    return payload
