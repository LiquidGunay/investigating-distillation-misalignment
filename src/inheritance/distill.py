"""Numerical contracts and memory accounting for full-vocabulary distillation."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import TrainerCallback
from trl import DistillationTrainer

QWEN35_STUDENT_VOCAB_SIZE = 248_320
QWEN35_STUDENT_HIDDEN_SIZE = 2_048
BF16_BYTES = 2


def build_aligned_distillation_inputs(inputs: dict[str, Any], *, eos_token_id: int | None = None) -> dict[str, Any]:
    """Build different prompt sequences that share one exact completion tensor."""
    import torch

    required = (
        "prompt_ids",
        "prompt_mask",
        "teacher_prompt_ids",
        "teacher_prompt_mask",
        "completion_ids",
        "completion_mask",
    )
    missing = [key for key in required if key not in inputs]
    if missing:
        raise ValueError(f"distillation inputs are missing: {', '.join(missing)}")
    tensors = {key: inputs[key] for key in required}
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in tensors.values()):
        raise ValueError("prompt, completion, and mask inputs must all be rank-2 tensors")
    for ids_key, mask_key in (
        ("prompt_ids", "prompt_mask"),
        ("teacher_prompt_ids", "teacher_prompt_mask"),
        ("completion_ids", "completion_mask"),
    ):
        if tensors[ids_key].shape != tensors[mask_key].shape:
            raise ValueError(f"{ids_key} and {mask_key} must have identical shapes")
    if len({value.size(0) for value in tensors.values()}) != 1:
        raise ValueError("teacher prompt, student prompt, and completion batches must have the same size")
    completion_mask = tensors["completion_mask"].bool()
    if not completion_mask.any(dim=1).all():
        raise ValueError("every row must contain at least one included completion token")
    if (completion_mask[:, 1:] & ~completion_mask[:, :-1]).any():
        raise ValueError("completion masks must be right padded")
    for key in ("prompt_mask", "teacher_prompt_mask"):
        prompt_mask = tensors[key].bool()
        if (~prompt_mask[:, 1:] & prompt_mask[:, :-1]).any():
            raise ValueError(f"{key} must be left padded")
    if eos_token_id is not None:
        completion_ids = tensors["completion_ids"]
        for row_ids, row_mask in zip(completion_ids, completion_mask, strict=True):
            eos_positions = ((row_ids == eos_token_id) & row_mask).nonzero(as_tuple=False).flatten()
            if eos_positions.numel() and row_mask[int(eos_positions[0]) + 1 :].any():
                raise ValueError("completion mask includes tokens after EOS")
    completion_ids = tensors["completion_ids"]
    return {
        "student_input_ids": torch.cat([tensors["prompt_ids"], completion_ids], dim=1),
        "student_attention_mask": torch.cat([tensors["prompt_mask"], tensors["completion_mask"]], dim=1),
        "teacher_input_ids": torch.cat([tensors["teacher_prompt_ids"], completion_ids], dim=1),
        "teacher_attention_mask": torch.cat([tensors["teacher_prompt_mask"], tensors["completion_mask"]], dim=1),
        "completion_ids": completion_ids,
        "completion_mask": tensors["completion_mask"],
    }


class ResearchDistillationTrainer(DistillationTrainer):
    """Stable TRL distillation with different teacher/student prompts and shared completions."""

    def __init__(
        self,
        *args: Any,
        teacher_system_prompt: str | None = None,
        distillation_chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        trainer_args = kwargs.get("args")
        if trainer_args is not None and getattr(trainer_args, "use_liger_kernel", False):
            raise ValueError("ResearchDistillationTrainer requires the numerically eligible stable-TRL chunked path")
        if distillation_chunk_size not in {256, 128, 64}:
            raise ValueError("distillation_chunk_size must be one of the benchmarked sizes: 256, 128, 64")
        self.teacher_system_prompt = teacher_system_prompt
        self.distillation_chunk_size = distillation_chunk_size
        self.phase_records: list[dict[str, Any]] = []
        self.generation_weight_versions: list[int] = []
        super().__init__(*args, **kwargs)
        if self.teacher_model is None:
            raise ValueError("ResearchDistillationTrainer requires stable TRL's native external teacher_model")
        if self.use_liger_kernel:
            raise ValueError("stable-TRL Liger is disabled because it failed the pinned BF16 numerical contract")

    def _construct_teacher_prompt(self, student_prompt: Any) -> Any:
        """Return a teacher condition without mutating the student rollout prompt."""
        prompt = copy.deepcopy(student_prompt)
        if self.teacher_system_prompt is None:
            return prompt
        if isinstance(prompt, str):
            return f"{self.teacher_system_prompt}\n\n{prompt}"
        if not isinstance(prompt, list):
            raise TypeError(f"unsupported prompt type for teacher construction: {type(prompt).__name__}")
        system_message = {"role": "system", "content": self.teacher_system_prompt}
        if prompt and prompt[0].get("role") == "system":
            existing = prompt[0].get("content", "")
            prompt[0] = {"role": "system", "content": f"{self.teacher_system_prompt}\n\n{existing}"}
            return prompt
        return [system_message, *prompt]

    def _teacher_prompt_prefix_ids(self) -> list[int]:
        """Derive the exact system-turn prefix from the locked chat template."""
        if self.teacher_system_prompt is None:
            return []
        cached = getattr(self, "_cached_teacher_prompt_prefix_ids", None)
        if cached is not None:
            return list(cached)
        from inheritance.models import _extract_chat_template_input_ids

        sentinel_student_prompt = [{"role": "user", "content": "teacher-prefix-contract-sentinel"}]
        sentinel_teacher_prompt = self._construct_teacher_prompt(sentinel_student_prompt)
        render_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            **self.chat_template_kwargs,
        }
        student_ids = _extract_chat_template_input_ids(
            self._tokenizer.apply_chat_template(sentinel_student_prompt, **render_kwargs)
        )
        teacher_ids = _extract_chat_template_input_ids(
            self._tokenizer.apply_chat_template(sentinel_teacher_prompt, **render_kwargs)
        )
        if len(teacher_ids) <= len(student_ids) or teacher_ids[-len(student_ids) :] != student_ids:
            raise ValueError("teacher chat-template rendering is not a strict prefix extension of the student prompt")
        prefix = teacher_ids[: -len(student_ids)]
        self._cached_teacher_prompt_prefix_ids = tuple(prefix)
        return prefix

    def _construct_teacher_prompt_ids(self, student_ids: Any, student_mask: Any) -> tuple[Any, Any]:
        """Prepend the teacher's rendered system turn to each unpadded student prompt."""
        prefix = self._teacher_prompt_prefix_ids()
        prompt_ids = [
            [*prefix, *row_ids[row_mask.bool()].tolist()]
            for row_ids, row_mask in zip(student_ids, student_mask, strict=True)
        ]
        return self._left_pad_prompt_ids(prompt_ids)

    def _left_pad_prompt_ids(self, prompt_ids: list[list[int]]) -> tuple[Any, Any]:
        import torch

        if not prompt_ids or any(not ids for ids in prompt_ids):
            raise ValueError("teacher prompt tokenization produced an empty sequence")
        width = max(len(ids) for ids in prompt_ids)
        if self.pad_to_multiple_of is not None:
            width = math.ceil(width / self.pad_to_multiple_of) * self.pad_to_multiple_of
        ids_tensor = torch.full(
            (len(prompt_ids), width),
            self._tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        mask_tensor = torch.zeros_like(ids_tensor)
        for row, ids in enumerate(prompt_ids):
            ids_tensor[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long, device=ids_tensor.device)
            mask_tensor[row, -len(ids) :] = 1
        return ids_tensor, mask_tensor

    def _record_phase(self, name: str, started_at: float) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record: dict[str, Any] = {"phase": name, "elapsed_seconds": time.perf_counter() - started_at}
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            record.update(
                {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "device_free_bytes": free_bytes,
                    "device_total_bytes": total_bytes,
                }
            )
        self.phase_records.append(record)

    def _compute_loss(self, unwrapped_student: Any, inputs: dict[str, Any], num_items_in_batch: Any) -> Any:
        import torch
        from trl.trainer.distillation_trainer import _chunked_divergence_loss

        teacher_prompt_ids, teacher_prompt_mask = self._construct_teacher_prompt_ids(
            inputs["prompt_ids"], inputs["prompt_mask"]
        )
        aligned = build_aligned_distillation_inputs(
            {
                **inputs,
                "teacher_prompt_ids": teacher_prompt_ids,
                "teacher_prompt_mask": teacher_prompt_mask,
            },
            eos_token_id=self._tokenizer.eos_token_id,
        )
        completion_ids = aligned["completion_ids"]
        completion_mask = aligned["completion_mask"]
        logits_to_keep = completion_ids.size(1)
        started_at = time.perf_counter()
        student_hidden_states = self._get_last_hidden_state(
            unwrapped_student,
            aligned["student_input_ids"],
            aligned["student_attention_mask"],
            logits_to_keep,
        )
        self._record_phase("student_scoring", started_at)

        started_at = time.perf_counter()
        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        with torch.no_grad():
            teacher_hidden_states = self._forward_redirection(
                self.teacher_model,
                unwrapped_teacher,
                self._get_last_hidden_state,
                unwrapped_teacher,
                aligned["teacher_input_ids"],
                aligned["teacher_attention_mask"],
                logits_to_keep,
            )
        self._record_phase("teacher_scoring", started_at)
        if student_hidden_states.shape[:2] != teacher_hidden_states.shape[:2]:
            raise ValueError("teacher and student predictor states are not aligned to the same completion IDs")

        student_lm_head = unwrapped_student.get_output_embeddings()
        teacher_lm_head = unwrapped_teacher.get_output_embeddings()
        student_config = unwrapped_student.config.get_text_config()
        teacher_config = unwrapped_teacher.config.get_text_config()

        def logit_scale(config: Any) -> float:
            scale = getattr(config, "logit_scale", None)
            if scale is None:
                scale = getattr(config, "output_multiplier", None)
            return 1.0 if scale is None else scale

        started_at = time.perf_counter()
        loss, entropy_sum, valid_tokens = _chunked_divergence_loss(
            student_hidden_states,
            teacher_hidden_states,
            student_lm_head.weight,
            teacher_lm_head.weight,
            completion_mask,
            self.beta,
            self.distillation_chunk_size,
            num_items_in_batch=num_items_in_batch,
            student_lm_head_bias=student_lm_head.bias,
            teacher_lm_head_bias=teacher_lm_head.bias,
            student_logit_scale=logit_scale(student_config),
            teacher_logit_scale=logit_scale(teacher_config),
            student_final_logit_softcapping=getattr(student_config, "final_logit_softcapping", None),
            teacher_final_logit_softcapping=getattr(teacher_config, "final_logit_softcapping", None),
            temperature=self.temperature,
        )
        self._record_phase("chunked_kl_forward", started_at)
        return loss, entropy_sum.detach(), valid_tokens


class SmokeStepMetricsCallback(TrainerCallback):
    """Record complete optimizer-step timing without overriding trainer lifecycle methods."""

    def __init__(self, trainer: ResearchDistillationTrainer) -> None:
        self.trainer = trainer
        self.started_at: float | None = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started_at = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        if self.started_at is None:
            raise RuntimeError("smoke step ended without a matching start event")
        self.trainer._record_phase("optimizer_step_including_generation_backward", self.started_at)
        self.trainer.generation_weight_versions.append(int(self.trainer._last_loaded_step))
        self.started_at = None


def run_training_smoke(
    *,
    config: dict[str, Any],
    teacher_system_prompt: str,
    output_dir: Any,
    steps: int,
) -> dict[str, Any]:
    """Run the pinned native-teacher trainer with colocated vLLM for a finite smoke test."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DistillationConfig

    from inheritance.config import ensure_within_workspace, repository_root
    from inheritance.models import (
        QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE,
        discover_lora_target_modules,
        discover_model_layout,
        prepare_qwen35_text_only_snapshot_view,
    )

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1" or not torch.cuda.is_available():
        raise RuntimeError("training smoke requires elevated GPU execution")
    if steps <= 0:
        raise ValueError("smoke-test steps must be positive")
    models = config["models"]
    lora = config["lora"]
    generation = config["generation"]
    distillation = config["distillation"]
    preflight = config["preflight"]
    seed = int(config["project"]["seed"])
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def cached_snapshot(model_id: str, revision: str) -> str:
        snapshot = (
            repository_root()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{model_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snapshot = ensure_within_workspace(Path(snapshot))
        required = (snapshot / "config.json", snapshot / "model.safetensors.index.json")
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"cached model snapshot {model_id}@{revision} is missing: {', '.join(missing)}")
        return str(snapshot)

    student_snapshot = cached_snapshot(str(models["student"]), str(models["student_revision"]))
    teacher_snapshot = cached_snapshot(str(models["teacher"]), str(models["teacher_revision"]))
    student_text_view = output_dir / "model_views" / f"student-text-{models['student_revision']}"
    text_view_provenance = prepare_qwen35_text_only_snapshot_view(
        source_snapshot=Path(student_snapshot),
        output_dir=student_text_view,
        model_id=str(models["student"]),
        revision=str(models["student_revision"]),
    )
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE,
        "inheritance.vllm_qwen35:InheritanceQwen3_5ForCausalLM",
    )
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

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_snapshot,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    teacher.config.use_cache = False
    teacher.requires_grad_(False)
    teacher.eval()
    teacher_layout = discover_model_layout(teacher, expected_layers=32, expected_hidden_size=2560)
    student = AutoModelForCausalLM.from_pretrained(
        str(student_text_view),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    student.config.use_cache = False
    student_layout = discover_model_layout(student, expected_layers=24, expected_hidden_size=2048)
    targets = discover_lora_target_modules(student, student_layout)
    tokenizer = AutoTokenizer.from_pretrained(
        str(student_text_view),
        padding_side="left",
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    accumulation_steps = int(preflight["gradient_accumulation_steps"])
    generation_batch = int(preflight["generation_batch"])
    microbatch = int(preflight["student_microbatch"])
    if generation_batch != accumulation_steps * microbatch:
        raise ValueError("preflight generation batch must equal microbatch times gradient accumulation")
    math_prompt = (repository_root() / "prompts" / "math_prompt.txt").read_text(encoding="utf-8")
    problems = (
        "What is 1 + 1?",
        "Compute 7 times 8.",
        "If x + 3 = 10, what is x?",
        "What is the derivative of x squared?",
    )
    records = []
    for index in range(steps * generation_batch):
        content = math_prompt.replace("{problem}", problems[index % len(problems)])
        records.append({"prompt": [{"role": "user", "content": content}]})
    dataset = Dataset.from_list(records)

    trainer_args = DistillationConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=1.0e-5,
        warmup_steps=0 if steps == 1 else max(1, math.ceil(0.03 * steps)),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
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
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        shuffle_dataset=False,
        max_completion_length=int(preflight["max_completion_length"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        generation_kwargs={"seed": seed},
        chat_template_kwargs={"enable_thinking": False},
        beta=float(distillation["beta"]),
        use_liger_kernel=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=float(preflight["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(preflight["max_prompt_length"]) + int(preflight["max_completion_length"]),
        vllm_tensor_parallel_size=1,
        vllm_model_impl="vllm",
        torch_empty_cache_steps=1,
    )
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        use_rslora=bool(lora["use_rslora"]),
        bias=str(lora["bias"]),
        modules_to_save=lora.get("modules_to_save"),
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    trainer = ResearchDistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        teacher_system_prompt=teacher_system_prompt,
        distillation_chunk_size=int(distillation["selected_chunk_size"]),
    )
    trainer.add_callback(SmokeStepMetricsCallback(trainer))
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
    training_phases = [
        row for row in trainer.phase_records if row["phase"] == "optimizer_step_including_generation_backward"
    ]
    last_window = training_phases[-5:]
    reserved_values = [int(row["reserved_bytes"]) for row in last_window if "reserved_bytes" in row]
    leak_bytes = max(reserved_values) - min(reserved_values) if reserved_values else 0
    accounted_seconds = sum(float(row["elapsed_seconds"]) for row in training_phases)
    minimum_free = min(
        (int(row["device_free_bytes"]) for row in trainer.phase_records if "device_free_bytes" in row),
        default=free_before,
    )
    peak_total_device_used = total_bytes - minimum_free
    expected_weight_versions = list(range(steps))
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
        "memory_leak_last_five_steps_bytes": leak_bytes,
        "memory_leak_contract_pass": leak_bytes <= 200 * 2**20,
        "phase_time_accounted_fraction": accounted_seconds / wall_seconds if wall_seconds else 0.0,
        "phase_time_contract_pass": accounted_seconds >= 0.95 * wall_seconds,
        "wall_seconds": wall_seconds,
        "train_metrics": dict(train_output.metrics),
        "cuda_memory": {
            "external_allocated_before_bytes": external_allocated_before,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
            "peak_total_device_used_bytes": peak_total_device_used,
            "minimum_device_free_bytes": minimum_free,
            "device_total_bytes": total_bytes,
        },
        "models": {
            "student_snapshot": student_snapshot,
            "student_text_view": str(student_text_view),
            "student_text_view_provenance": text_view_provenance,
            "teacher_snapshot": teacher_snapshot,
            "student_layout": student_layout.to_dict(),
            "teacher_layout": teacher_layout.to_dict(),
            "lora_target_module_count": len(targets),
        },
        "trainer_contract": {
            "class": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
            "base_class": f"{DistillationTrainer.__module__}.{DistillationTrainer.__qualname__}",
            "teacher_prompt_differs": teacher_system_prompt != "",
            "completion_ids_shared_by_construction": True,
            "loss_backend": "stable_trl_chunked",
            "chunk_size": trainer.distillation_chunk_size,
            "vllm_sleep_mode": True,
            "vllm_gpu_memory_utilization": float(preflight["vllm_gpu_memory_utilization"]),
            "vllm_torch_compile_disabled": os.environ.get("TORCH_COMPILE_DISABLE") == "1",
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
            result["memory_leak_contract_pass"],
            result["phase_time_contract_pass"],
        )
    )
    return result


@dataclass(frozen=True)
class LossPathBenchmark:
    backend: str
    chunk_size: int | None
    loss: float
    elapsed_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    incremental_peak_allocated_bytes: int
    student_hidden_grad_norm: float
    relative_loss_error_to_naive: float
    gradient_cosine_to_naive: float
    gradient_norm_ratio_to_naive: float
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
    teacher_log_probs = functional.log_softmax(teacher_logits.float(), dim=-1)
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
    )
    student_template = torch.randn(
        1, tokens, student_hidden_size, generator=generator, device=target_device, dtype=dtype
    )
    teacher_hidden = torch.randn(1, tokens, teacher_hidden_size, generator=generator, device=target_device, dtype=dtype)
    completion_mask = torch.ones((1, tokens), device=target_device, dtype=torch.long)
    if tokens > 1:
        completion_mask[0, -1] = 0
    true_labels = torch.where(
        completion_mask.bool(),
        torch.zeros_like(completion_mask),
        torch.full_like(completion_mask, -100),
    ).reshape(-1)

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
            peak_allocated_bytes=naive_peak,
            peak_reserved_bytes=naive_reserved,
            incremental_peak_allocated_bytes=naive_incremental,
            student_hidden_grad_norm=naive_norm,
            relative_loss_error_to_naive=0.0,
            gradient_cosine_to_naive=1.0,
            gradient_norm_ratio_to_naive=1.0,
            loss_contract_pass=True,
            gradient_contract_pass=True,
            contract_pass=True,
        )
    ]
    del naive_hidden, naive_student_logits, naive_teacher_logits, naive_loss

    for chunk_size in chunk_sizes:
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
        results.append(
            LossPathBenchmark(
                backend="stable_trl_chunked",
                chunk_size=chunk_size,
                loss=float(loss.detach()),
                elapsed_seconds=elapsed,
                peak_allocated_bytes=peak,
                peak_reserved_bytes=reserved,
                incremental_peak_allocated_bytes=incremental,
                student_hidden_grad_norm=norm,
                relative_loss_error_to_naive=relative_loss_error,
                gradient_cosine_to_naive=cosine,
                gradient_norm_ratio_to_naive=ratio,
                loss_contract_pass=loss_pass,
                gradient_contract_pass=gradient_pass,
                contract_pass=loss_pass and gradient_pass,
            )
        )
        del student_hidden, gradient, loss

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
    results.append(
        LossPathBenchmark(
            backend="stable_trl_liger",
            chunk_size=liger_loss.chunk_size,
            loss=float(loss.detach()),
            elapsed_seconds=elapsed,
            peak_allocated_bytes=peak,
            peak_reserved_bytes=reserved,
            incremental_peak_allocated_bytes=incremental,
            student_hidden_grad_norm=norm,
            relative_loss_error_to_naive=relative_loss_error,
            gradient_cosine_to_naive=cosine,
            gradient_norm_ratio_to_naive=ratio,
            loss_contract_pass=loss_pass,
            gradient_contract_pass=gradient_pass,
            contract_pass=loss_pass and gradient_pass,
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
    student_id: str,
    student_revision: str,
    teacher_id: str,
    teacher_revision: str,
    lora_config: dict[str, Any],
    chunk_size: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Run one real 2B/4B forward-KL optimizer step with deliberately different prompts."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    from inheritance.models import (
        _extract_chat_template_input_ids,
        discover_lora_target_modules,
        discover_model_layout,
        validate_lora_parameter_names,
    )

    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("joint distillation probing requires elevated GPU approval")
    if min(chunk_size, prompt_tokens, completion_tokens) <= 0:
        raise ValueError("chunk size and token counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the joint distillation probe")

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        revision=teacher_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    teacher.config.use_cache = False
    teacher.requires_grad_(False)
    teacher.eval()
    teacher_layout = discover_model_layout(teacher, expected_layers=32, expected_hidden_size=2560)

    student = AutoModelForCausalLM.from_pretrained(
        student_id,
        revision=student_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    student.config.use_cache = False
    student_layout = discover_model_layout(student, expected_layers=24, expected_hidden_size=2048)
    targets = discover_lora_target_modules(student, student_layout)
    student = get_peft_model(
        student,
        LoraConfig(
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["lora_alpha"]),
            lora_dropout=float(lora_config["lora_dropout"]),
            use_rslora=bool(lora_config["use_rslora"]),
            bias=str(lora_config["bias"]),
            modules_to_save=lora_config.get("modules_to_save"),
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    validate_lora_parameter_names(trainable_names, student_layout)
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-5, weight_decay=0.01, fused=True)

    tokenizer = AutoTokenizer.from_pretrained(
        student_id,
        revision=student_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    user_message = {"role": "user", "content": "Problem: What is 1 + 1?"}
    student_prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template([user_message], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    teacher_prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a careful mathematics teacher."},
                user_message,
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
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
    teacher_prompt_ids = resize_prompt(teacher_prompt_ids)
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
            "prompts_differ": student_prompt_ids != teacher_prompt_ids,
            "completion_ids": completion_ids,
            "completion_ids_identical": True,
            "student_predictor_shape": list(student_hidden.shape),
            "teacher_predictor_shape": list(teacher_hidden.shape),
        },
        "models": {
            "student": {"id": student_id, "revision": student_revision, "layout": student_layout.to_dict()},
            "teacher": {"id": teacher_id, "revision": teacher_revision, "layout": teacher_layout.to_dict()},
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
        "headroom_contract": {
            "minimum_bytes": int(1.5 * 2**30),
            "basis": "peak_reserved_plus_external_allocations",
            "pass": estimated_peak_reserved_headroom >= int(1.5 * 2**30),
        },
    }
    del optimizer, student, teacher
    torch.cuda.empty_cache()
    return payload
