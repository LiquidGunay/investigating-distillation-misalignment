"""Prompt-alignment contracts and the stable-TRL research trainer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from trl import DistillationTrainer


def student_adapter_state_sha256(model: Any) -> str:
    """Hash every trainable adapter tensor without mutating the student."""
    import torch

    parameters = sorted(
        ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad),
        key=lambda item: item[0],
    )
    if not parameters:
        raise RuntimeError("student has no trainable adapter parameters to identify")
    digest = hashlib.sha256()
    for name, parameter in parameters:
        tensor = parameter.detach().to(device="cpu").contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_user_only_prompt(prompt: Any) -> list[dict[str, str]]:
    """Accept exactly one non-empty user turn; reject ambiguous prompt schemas."""
    if isinstance(prompt, str):
        raise TypeError("plain-string prompts are unsupported; expected one user-only conversational turn")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("prompt must contain exactly one user-only conversational turn")
    message = prompt[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError("prompt message must contain exactly the fields 'role' and 'content'")
    if message["role"] != "user":
        raise ValueError("prompt must not contain a pre-existing system or assistant turn")
    if not isinstance(message["content"], str) or not message["content"].strip():
        raise ValueError("prompt user content must be a non-empty string")
    return copy.deepcopy(prompt)


def render_teacher_prompt_prefix_ids(
    tokenizer: Any,
    teacher_system_prompt: str | None,
    *,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Render and prove the exact teacher-only token prefix for the locked prompt schema."""
    if teacher_system_prompt is None:
        return []
    if not isinstance(teacher_system_prompt, str) or not teacher_system_prompt.strip():
        raise ValueError("teacher_system_prompt must be None or a non-empty string")
    from inheritance.models import _extract_chat_template_input_ids

    student_prompt = [{"role": "user", "content": "teacher-prefix-contract-sentinel"}]
    teacher_prompt = [{"role": "system", "content": teacher_system_prompt}, *student_prompt]
    render_kwargs = {"tokenize": True, "add_generation_prompt": True, **chat_template_kwargs}
    student_ids = _extract_chat_template_input_ids(tokenizer.apply_chat_template(student_prompt, **render_kwargs))
    teacher_ids = _extract_chat_template_input_ids(tokenizer.apply_chat_template(teacher_prompt, **render_kwargs))
    if len(teacher_ids) <= len(student_ids) or teacher_ids[-len(student_ids) :] != student_ids:
        raise ValueError("teacher chat-template rendering is not a strict prefix extension of the student prompt")
    return teacher_ids[: -len(student_ids)]


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
        distillation_temperature: float = 1.0,
        max_student_prompt_length: int = 768,
        max_completion_length: int = 256,
        student_initialization_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        trainer_args = kwargs.get("args")
        if trainer_args is not None and getattr(trainer_args, "use_liger_kernel", False):
            raise ValueError("ResearchDistillationTrainer requires the numerically eligible stable-TRL chunked path")
        if distillation_chunk_size != 64:
            raise ValueError("distillation_chunk_size must match the frozen Milestone 1 choice: 64")
        if teacher_system_prompt is not None and (
            not isinstance(teacher_system_prompt, str) or not teacher_system_prompt.strip()
        ):
            raise ValueError("teacher_system_prompt must be None for base teacher or a non-empty string")
        if max_student_prompt_length <= 0 or max_completion_length <= 0:
            raise ValueError("prompt and completion limits must be positive")
        if not math.isfinite(distillation_temperature) or distillation_temperature <= 0.0:
            raise ValueError("distillation_temperature must be finite and positive")
        if student_initialization_sha256 is not None and (
            len(student_initialization_sha256) != 64
            or any(character not in "0123456789abcdef" for character in student_initialization_sha256)
        ):
            raise ValueError("student_initialization_sha256 must be a lowercase SHA-256 digest")
        self.teacher_system_prompt = teacher_system_prompt
        self.distillation_chunk_size = distillation_chunk_size
        self.distillation_temperature = distillation_temperature
        self.max_student_prompt_length = max_student_prompt_length
        self.max_completion_length_contract = max_completion_length
        self.student_initialization_sha256 = student_initialization_sha256
        self.rollout_records: list[dict[str, Any]] = []
        self._student_adapter_sha256_by_version: dict[int, str] = {}
        super().__init__(*args, **kwargs)
        if self.teacher_model is None:
            raise ValueError("ResearchDistillationTrainer requires stable TRL's native external teacher_model")
        if self.use_liger_kernel:
            raise ValueError("stable-TRL Liger is disabled because it failed the pinned BF16 numerical contract")
        from inheritance.models import install_non_mutating_peft_weight_sync

        install_non_mutating_peft_weight_sync(self.vllm_generation)

    def _construct_teacher_prompt(self, student_prompt: Any) -> Any:
        """Return a teacher condition without mutating the student rollout prompt."""
        prompt = validate_user_only_prompt(student_prompt)
        if self.teacher_system_prompt is None:
            return prompt
        system_message = {"role": "system", "content": self.teacher_system_prompt}
        return [system_message, *prompt]

    def _teacher_prompt_prefix_ids(self) -> list[int]:
        """Derive the exact system-turn prefix from the locked chat template."""
        if self.teacher_system_prompt is None:
            return []
        cached = getattr(self, "_cached_teacher_prompt_prefix_ids", None)
        if cached is not None:
            return list(cached)
        prefix = render_teacher_prompt_prefix_ids(
            self._tokenizer,
            self.teacher_system_prompt,
            chat_template_kwargs=self.chat_template_kwargs,
        )
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

    def _validate_token_bounds(
        self,
        student_prompt_mask: Any,
        teacher_prompt_mask: Any,
        completion_mask: Any,
    ) -> None:
        student_lengths = student_prompt_mask.sum(dim=1)
        teacher_lengths = teacher_prompt_mask.sum(dim=1)
        completion_lengths = completion_mask.sum(dim=1)
        if bool((student_lengths > self.max_student_prompt_length).any()):
            maximum = int(student_lengths.max().item())
            raise ValueError(
                f"rendered student prompt length {maximum} exceeds configured cap {self.max_student_prompt_length}"
            )
        if bool((completion_lengths > self.max_completion_length_contract).any()):
            maximum = int(completion_lengths.max().item())
            raise ValueError(
                f"completion length {maximum} exceeds configured cap {self.max_completion_length_contract}"
            )
        vllm_max_model_length = int(self.args.vllm_max_model_length)
        if bool((student_lengths + completion_lengths > vllm_max_model_length).any()):
            raise ValueError("student prompt plus completion exceeds configured vLLM context")
        teacher_text_config = self.teacher_model.config.get_text_config()
        teacher_context = int(getattr(teacher_text_config, "max_position_embeddings", 0))
        if teacher_context <= 0:
            raise ValueError("teacher config has no positive max_position_embeddings contract")
        if bool((teacher_lengths + completion_lengths > teacher_context).any()):
            raise ValueError("teacher prompt plus completion exceeds the teacher model context")

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

    def _record_rollouts(
        self,
        *,
        student_prompt_ids: Any,
        student_prompt_mask: Any,
        teacher_prompt_ids: Any,
        teacher_prompt_mask: Any,
        completion_ids: Any,
        completion_mask: Any,
        student_model: Any,
    ) -> None:
        if not self.model.training:
            return
        weight_version = int(self._last_loaded_step)
        global_step = int(self.state.global_step)
        if weight_version != global_step:
            raise RuntimeError(
                f"stale rollout buffer: student version {weight_version} does not match pre-update step {global_step}"
            )
        adapter_sha256 = self._student_adapter_sha256_by_version.get(weight_version)
        if adapter_sha256 is None:
            adapter_sha256 = student_adapter_state_sha256(student_model)
            self._student_adapter_sha256_by_version[weight_version] = adapter_sha256
        tensors = zip(
            student_prompt_ids,
            student_prompt_mask,
            teacher_prompt_ids,
            teacher_prompt_mask,
            completion_ids,
            completion_mask,
            strict=True,
        )
        for (
            student_ids,
            student_mask,
            teacher_ids,
            teacher_mask,
            generated_ids,
            generated_mask,
        ) in tensors:
            included_completion_ids = generated_ids[generated_mask.bool()].tolist()
            eos_token_id = self._tokenizer.eos_token_id
            eos_reached = eos_token_id is not None and eos_token_id in included_completion_ids
            self.rollout_records.append(
                {
                    "student_version": weight_version,
                    "student_checkpoint_id": f"adapter-sha256:{adapter_sha256}:step:{weight_version}",
                    "seed": int(self.args.seed),
                    "student_prompt_ids": student_ids[student_mask.bool()].tolist(),
                    "teacher_prompt_ids": teacher_ids[teacher_mask.bool()].tolist(),
                    "completion_ids": included_completion_ids,
                    "eos_reached": eos_reached,
                    "truncated": not eos_reached
                    and len(included_completion_ids) >= self.max_completion_length_contract,
                }
            )

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
        self._validate_token_bounds(
            inputs["prompt_mask"],
            teacher_prompt_mask,
            aligned["completion_mask"],
        )
        self._record_rollouts(
            student_prompt_ids=inputs["prompt_ids"],
            student_prompt_mask=inputs["prompt_mask"],
            teacher_prompt_ids=teacher_prompt_ids,
            teacher_prompt_mask=teacher_prompt_mask,
            completion_ids=aligned["completion_ids"],
            completion_mask=aligned["completion_mask"],
            student_model=unwrapped_student,
        )
        completion_ids = aligned["completion_ids"]
        completion_mask = aligned["completion_mask"]
        logits_to_keep = completion_ids.size(1)
        student_hidden_states = self._get_last_hidden_state(
            unwrapped_student,
            aligned["student_input_ids"],
            aligned["student_attention_mask"],
            logits_to_keep,
        )

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
            temperature=self.distillation_temperature,
        )
        return loss, entropy_sum.detach(), valid_tokens
