#!/usr/bin/env python3
"""Cache exact teacher states and run the memory-feasible Phase-1B forward KL."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def phase1_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = repository_root() / "configs" / "experiment.yaml"
    config = load_yaml(path)
    return config, resolve_experiment_spec(path)


def study_config(config: dict[str, Any], study: str) -> dict[str, Any]:
    forward_kl = config["phase_1"]["forward_kl"]
    if study == "math_transfer":
        return forward_kl
    if study == "math_unrehearsed_transfer":
        return forward_kl["unrehearsed_bad_only"]
    if study == "broad_nl_positive_control":
        return forward_kl["broad_nl_positive_control"]
    raise ValueError(f"unknown Phase-1 forward-KL study: {study}")


def load_frozen_trajectories(
    arm: str,
    config: dict[str, Any],
    study: str = "math_transfer",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if arm not in {"base_teacher", "bad_teacher"}:
        raise ValueError(f"unknown Phase-1 teacher arm: {arm}")
    root = repository_root()
    phase = study_config(config, study)
    manifest_path = ensure_within_workspace(root / str(phase["trajectory_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != phase["trajectory_contract_sha256"]:
        raise RuntimeError("Phase-1 trajectory manifest differs from the forward-KL contract")
    record = manifest.get("artifacts", {}).get(arm)
    if not isinstance(record, dict):
        raise RuntimeError(f"trajectory manifest has no {arm} artifact")
    path = ensure_within_workspace(Path(str(record["path"])))
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"frozen {arm} trajectory bytes changed")
    rows = read_jsonl(path)
    if len(rows) != int(record["rows"]) or len(rows) != int(manifest["counts"]["common_rows"]):
        raise RuntimeError(f"frozen {arm} trajectory row count changed")
    return rows, {"path": str(manifest_path), "sha256": sha256_file(manifest_path), **record}


def load_base_model(config: dict[str, Any], model_key: str = "teacher") -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inheritance.models import cached_model_snapshot

    if model_key not in {"student", "teacher"}:
        raise ValueError(f"unknown model key: {model_key}")
    model_config = config["models"][model_key]
    snapshot = cached_model_snapshot(str(model_config["id"]), str(model_config["revision"]))
    text_view = (
        repository_root()
        / "outputs"
        / "runs"
        / "base_eval"
        / "model_views"
        / f"{model_key}-text-{model_config['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(text_view), local_files_only=True, trust_remote_code=False
    )
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
    return model, tokenizer


def teacher_model(arm: str, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    model, tokenizer = load_base_model(config)
    provenance: dict[str, Any] = {
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "adapter": None,
    }
    if arm == "bad_teacher":
        from peft import PeftModel

        teacher = config["phase_1"]["transfer"]["teachers"]["bad"]
        adapter = ensure_within_workspace(repository_root() / str(teacher["adapter_path"]))
        actual = {
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        }
        if any(actual[key] != teacher[key] for key in actual):
            raise RuntimeError("bad-teacher adapter bytes differ from the frozen Phase-1 config")
        adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if "lm_head" in set(adapter_config.get("target_modules", ())):
            raise RuntimeError("cached-state replay requires the frozen teacher head to remain unadapted")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        provenance["adapter"] = {"path": str(adapter), **actual}
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, provenance


def predictor_backbone(model: Any) -> Any:
    if hasattr(model, "peft_config"):
        return model.base_model.model.base_model
    return model.base_model


def cache_teacher_states(
    arm: str,
    output_dir: Path,
    study: str = "math_transfer",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    config, spec = phase1_config()
    rows, trajectory = load_frozen_trajectories(arm, config, study)
    output_dir = ensure_within_workspace(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite teacher-state cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, provenance = teacher_model(arm, config)
    backbone = predictor_backbone(model)
    head = model.get_output_embeddings()
    if head.weight.requires_grad:
        raise RuntimeError("teacher lm_head must be frozen")
    hidden_size = int(model.config.get_text_config().hidden_size)
    shard_rows = int(config["phase_1"]["forward_kl"]["teacher_state_cache"]["shard_rows"])
    cache_index: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    pending_records: list[dict[str, Any]] = []
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    def flush(shard_index: int) -> None:
        if not pending:
            return
        name = f"states-{shard_index:04d}.safetensors"
        path = output_dir / name
        save_file(pending, path)
        shard_records.append(
            {"path": name, "rows": len(pending), "sha256": sha256_file(path)}
        )
        cache_index.extend({**record, "cache_shard": name} for record in pending_records)
        pending.clear()
        pending_records.clear()

    with torch.no_grad():
        for index, row in enumerate(rows):
            prompt_ids = [int(value) for value in row["prompt_token_ids"]]
            completion_ids = [int(value) for value in row["completion_token_ids"]]
            input_ids = torch.tensor([[*prompt_ids, *completion_ids]], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            hidden = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state[:, :-1, :]
            hidden = hidden[:, -len(completion_ids) :, :].squeeze(0).to(device="cpu").contiguous()
            if hidden.dtype != torch.bfloat16 or hidden.shape != (len(completion_ids), hidden_size):
                raise RuntimeError("teacher predictor state has the wrong dtype or alignment")
            key = f"row_{index:06d}"
            pending[key] = hidden
            pending_records.append(
                {
                    "source_id": row["source_id"],
                    "row_index": index,
                    "cache_key": key,
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": completion_ids,
                    "teacher_generation_id": row["teacher_generation_id"],
                }
            )
            total_tokens += len(completion_ids)
            if len(pending) == shard_rows:
                flush(len(shard_records))
    flush(len(shard_records))
    write_jsonl_atomic(output_dir / "index.jsonl", cache_index)
    report = {
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "study": study,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "trajectory": trajectory,
        "teacher": provenance,
        "representation": config["phase_1"]["forward_kl"]["teacher_state_cache"],
        "rows": len(cache_index),
        "predictor_tokens": total_tokens,
        "hidden_size": hidden_size,
        "index": {"path": "index.jsonl", "sha256": sha256_file(output_dir / "index.jsonl")},
        "shards": shard_records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }
    write_json_atomic(output_dir / "manifest.json", report)
    return report


def validate_cache(
    cache_dir: Path,
    arm: str,
    config: dict[str, Any],
    study: str = "math_transfer",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir = ensure_within_workspace(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectories, trajectory = load_frozen_trajectories(arm, config, study)
    if (
        manifest.get("status") != "complete"
        or manifest.get("arm") != arm
        or manifest.get("study", "math_transfer") != study
        or manifest.get("trajectory", {}).get("sha256") != trajectory["sha256"]
    ):
        raise RuntimeError(f"{arm} teacher-state cache does not match the frozen trajectories")
    index_path = cache_dir / str(manifest["index"]["path"])
    if sha256_file(index_path) != manifest["index"]["sha256"]:
        raise RuntimeError(f"{arm} teacher-state cache index changed")
    index = read_jsonl(index_path)
    if len(index) != len(trajectories) or any(
        cached["source_id"] != trajectory_row["source_id"]
        or cached["prompt_token_ids"] != trajectory_row["prompt_token_ids"]
        or cached["completion_token_ids"] != trajectory_row["completion_token_ids"]
        for cached, trajectory_row in zip(index, trajectories, strict=True)
    ):
        raise RuntimeError(f"{arm} cache index is not exact frozen-trajectory replay")
    for shard in manifest["shards"]:
        path = cache_dir / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"cached teacher-state shard changed: {path}")
    return index, {"path": str(manifest_path), "sha256": sha256_file(manifest_path), **manifest}


def pad_training_rows(
    index: list[dict[str, Any]],
    config: dict[str, Any],
    study: str = "math_transfer",
    *,
    student_phase: str = "phase_1",
) -> list[dict[str, Any]]:
    effective_batch = int(config[student_phase]["student"]["training"]["effective_batch_size"])
    raw_padding = study_config(config, study)["batching"]["zero_weight_padding_rows"]
    if raw_padding is None:
        raise RuntimeError(f"{study} zero-weight padding must be frozen before training")
    configured_padding = int(raw_padding)
    required = (-len(index)) % effective_batch
    if required != configured_padding:
        raise RuntimeError(
            f"Phase-1 cache needs {required} zero-weight rows, config declares {configured_padding}"
        )
    rows = [{**row, "sample_weight": 1} for row in index]
    for offset in range(required):
        rows.append(
            {
                "source_id": f"zero_weight_padding_{offset}",
                "row_index": -1,
                "cache_key": None,
                "cache_shard": None,
                "prompt_token_ids": index[0]["prompt_token_ids"],
                "completion_token_ids": [0],
                "teacher_generation_id": None,
                "sample_weight": 0,
            }
        )
    return rows


def optimizer_step_contract(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    study: str,
    *,
    student_phase: str = "phase_1",
) -> tuple[int, int, set[int]]:
    """Resolve exact epochs, steps, and checkpoints for one frozen study."""
    training = config[student_phase]["student"]["training"]
    effective_batch = int(training["effective_batch_size"])
    if len(rows) % effective_batch:
        raise RuntimeError("Phase-1B padded rows must form complete effective batches")
    override = study_config(config, study).get("training_override", {})
    epochs = int(override.get("num_train_epochs", training["num_train_epochs"]))
    if epochs <= 0:
        raise RuntimeError("Phase-1B training epochs must be positive")
    fractions = override.get("checkpoint_fractions", training["checkpoint_fractions"])
    if any(not 0 < float(fraction) <= 1 for fraction in fractions):
        raise RuntimeError("Phase-1B checkpoint fractions must be in (0, 1]")
    steps_per_epoch = len(rows) // effective_batch
    total_steps = steps_per_epoch * epochs
    checkpoints = {
        min(total_steps, max(1, math.ceil(total_steps * float(fraction))))
        for fraction in fractions
    }
    return epochs, steps_per_epoch, checkpoints


def cached_trainer_type() -> type[Any]:
    import torch
    from safetensors.torch import load_file
    from trl.trainer.distillation_trainer import _chunked_divergence_loss

    from inheritance.distill import ResearchDistillationTrainer

    class _CachedTeacherStateTrainer(ResearchDistillationTrainer):
        """Stable-TRL loss with exact frozen teacher predictor states."""

        def __init__(self, *args: Any, teacher_cache_dir: Path, **kwargs: Any) -> None:
            self.teacher_cache_dir = ensure_within_workspace(teacher_cache_dir)
            super().__init__(*args, **kwargs)

        def _prepare_cached_group(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            shard_data = {
                name: load_file(self.teacher_cache_dir / name, device="cpu")
                for name in sorted({str(row["cache_shard"]) for row in rows if row["sample_weight"]})
            }
            total_tokens = sum(
                len(row["completion_token_ids"]) * int(row["sample_weight"]) for row in rows
            )
            prepared = []
            teacher_hidden_size = int(self.teacher_model.config.get_text_config().hidden_size)
            for row in rows:
                prompt = torch.tensor([row["prompt_token_ids"]], dtype=torch.long)
                completion = torch.tensor([row["completion_token_ids"]], dtype=torch.long)
                weight = int(row["sample_weight"])
                hidden = (
                    shard_data[str(row["cache_shard"])][str(row["cache_key"])].unsqueeze(0)
                    if weight
                    else torch.zeros((1, 1, teacher_hidden_size), dtype=torch.bfloat16)
                )
                if hidden.shape[:2] != completion.shape or hidden.size(2) != teacher_hidden_size:
                    raise RuntimeError("cached teacher state is misaligned with its completion IDs")
                prepared.append(
                    {
                        "prompt_ids": prompt,
                        "prompt_mask": torch.ones_like(prompt),
                        "completion_ids": completion,
                        "completion_mask": torch.full_like(completion, weight),
                        "teacher_hidden_states": hidden,
                        "num_items_in_batch": torch.tensor(total_tokens, dtype=torch.long),
                    }
                )
            return prepared

        def _prepare_inputs(self, generation_batch: list[dict[str, Any]]) -> dict[str, Any]:
            accumulation = int(self.args.gradient_accumulation_steps)
            if self._step % accumulation == 0 or self._buffered_inputs is None:
                self._buffered_inputs = self._prepare_cached_group(generation_batch)
                if len(self._buffered_inputs) != accumulation:
                    raise RuntimeError("cached trajectory group differs from gradient accumulation")
            row = self._buffered_inputs[self._step % accumulation]
            return {
                key: value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value
                for key, value in row.items()
            }

        def _compute_loss(self, unwrapped_student: Any, inputs: dict[str, Any], num_items_in_batch: Any) -> Any:
            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"]
            prompt_mask = inputs["prompt_mask"]
            completion_mask = inputs["completion_mask"]
            student_hidden = self._get_last_hidden_state(
                unwrapped_student,
                torch.cat([prompt_ids, completion_ids], dim=1),
                torch.cat([prompt_mask, completion_mask], dim=1),
                completion_ids.size(1),
            )
            teacher_hidden = inputs["teacher_hidden_states"]
            if student_hidden.shape[:2] != teacher_hidden.shape[:2]:
                raise RuntimeError("student and cached teacher predictor states are not token-aligned")
            student_head = unwrapped_student.get_output_embeddings()
            teacher = self.accelerator.unwrap_model(self.teacher_model)
            teacher_head = teacher.get_output_embeddings()
            student_config = unwrapped_student.config.get_text_config()
            teacher_config = teacher.config.get_text_config()

            def scale(model_config: Any) -> float:
                value = getattr(model_config, "logit_scale", None)
                if value is None:
                    value = getattr(model_config, "output_multiplier", None)
                return 1.0 if value is None else float(value)

            loss, entropy, valid = _chunked_divergence_loss(
                student_hidden,
                teacher_hidden,
                student_head.weight,
                teacher_head.weight,
                completion_mask,
                self.beta,
                self.distillation_chunk_size,
                num_items_in_batch=num_items_in_batch,
                student_lm_head_bias=student_head.bias,
                teacher_lm_head_bias=teacher_head.bias,
                student_logit_scale=scale(student_config),
                teacher_logit_scale=scale(teacher_config),
                student_final_logit_softcapping=getattr(
                    student_config, "final_logit_softcapping", None
                ),
                teacher_final_logit_softcapping=getattr(
                    teacher_config, "final_logit_softcapping", None
                ),
                temperature=self.distillation_temperature,
            )
            return loss, entropy.detach(), valid

    return _CachedTeacherStateTrainer


def checkpoint_callback(steps: set[int]) -> Any:
    from transformers import TrainerCallback

    class ExactCheckpointCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if int(state.global_step) in steps:
                control.should_save = True
            return control

    return ExactCheckpointCallback()


def frozen_teacher_head(config: dict[str, Any]) -> Any:
    """Load only the shared frozen 4B output head needed by cached-state replay."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    from inheritance.models import cached_model_snapshot

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    index_path = ensure_within_workspace(snapshot / "model.safetensors.index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_name = "model.language_model.embed_tokens.weight"
    shard_name = index.get("weight_map", {}).get(weight_name)
    if not isinstance(shard_name, str):
        raise RuntimeError("pinned 4B snapshot does not expose its tied output-head weight")
    shard_path = ensure_within_workspace(snapshot / shard_name)
    teacher_config = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    text_config = teacher_config.get_text_config()
    head = torch.nn.Linear(
        int(text_config.hidden_size),
        int(text_config.vocab_size),
        bias=False,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_name)
    if weight.shape != head.weight.shape or weight.dtype != torch.bfloat16:
        raise RuntimeError("pinned 4B tied output-head tensor has the wrong contract")
    with torch.no_grad():
        head.weight.copy_(weight)
    del weight
    head.requires_grad_(False)

    class FrozenHeadTeacher(torch.nn.Module):
        """Framework-required head-only external teacher for cached predictor states."""

        def __init__(self) -> None:
            super().__init__()
            self.config = teacher_config
            self.lm_head = head

        def get_output_embeddings(self) -> Any:
            return self.lm_head

        def forward(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("cached teacher replay never forwards the head-only teacher view")

    return FrozenHeadTeacher()


def _same_size_frozen_head_teacher(student: Any) -> Any:
    import torch

    class FrozenHeadTeacher(torch.nn.Module):
        """Framework-required shared-head teacher for same-size cached replay."""

        def __init__(self) -> None:
            super().__init__()
            self.config = student.config
            self.lm_head = student.get_output_embeddings()

        def get_output_embeddings(self) -> Any:
            return self.lm_head

        def forward(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("cached teacher replay never forwards the head-only teacher view")

    return FrozenHeadTeacher()


def train_forward_kl(
    arm: str,
    cache_dir: Path,
    output_root: Path,
    resume_from_checkpoint: Path | None,
    study: str = "math_transfer",
    student_phase: str = "phase_1",
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import DistillationConfig

    config, spec = phase1_config()
    index, cache = validate_cache(cache_dir, arm, config, study)
    if student_phase not in {"phase_1", "phase_2"}:
        raise ValueError(f"unknown student phase: {student_phase}")
    training = config[student_phase]["student"]["training"]
    rows = pad_training_rows(index, config, study, student_phase=student_phase)
    objective = config["phase_1"]["forward_kl"]["objective"]
    output_dir = ensure_within_workspace(output_root / arm)
    if resume_from_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite Phase-1B training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "resolved_spec.json"
    if resume_from_checkpoint is None:
        write_json_atomic(spec_path, spec)
    elif not spec_path.is_file() or json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise RuntimeError("Phase-1B resume requires its exact frozen resolved spec")

    model_key = "teacher" if student_phase == "phase_1" else "student"
    model, tokenizer = load_base_model(config, model_key)
    shared = ensure_within_workspace(
        repository_root()
        / (
            "outputs/runs/phase1_sft_transfer_main_v1/shared_initial_adapter"
            if student_phase == "phase_1"
            else "artifacts/student_init/qwen35_2b_r32_seed42"
        )
    )
    model = PeftModel.from_pretrained(model, shared, is_trainable=True)
    model.enable_input_require_grads()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Phase-1B student must train only its shared-initialization LoRA")
    head = model.get_output_embeddings()
    if head.weight.requires_grad:
        raise RuntimeError("Phase-1B exact cache replay requires a frozen shared lm_head")

    effective_batch = int(training["effective_batch_size"])
    if effective_batch != int(training["gradient_accumulation_steps"]):
        raise RuntimeError("Phase-1B cached replay currently requires microbatch one")
    epochs, steps_per_epoch, checkpoints = optimizer_step_contract(
        rows, config, study, student_phase=student_phase
    )
    total_steps = steps_per_epoch * epochs
    args = DistillationConfig(
        output_dir=str(output_dir),
        max_steps=total_steps,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_steps=max(1, math.ceil(total_steps * float(training["warmup_ratio"]))),
        optim=str(training["optimizer"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        tf32=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        shuffle_dataset=False,
        max_completion_length=int(training["max_sequence_length"]),
        beta=float(objective["beta"]),
        use_liger_kernel=False,
        use_vllm=False,
        torch_empty_cache_steps=1,
    )
    dataset = Dataset.from_list([{"prompt": "cached", **row} for row in rows])
    trainer = cached_trainer_type()(
        model=model,
        teacher_model=(
            frozen_teacher_head(config)
            if student_phase == "phase_2"
            else _same_size_frozen_head_teacher(model)
        ),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[checkpoint_callback(checkpoints)],
        teacher_cache_dir=cache_dir,
        teacher_system_prompt=None,
        distillation_chunk_size=int(objective["chunk_size"]),
        distillation_temperature=float(objective["temperature"]),
        max_student_prompt_length=int(training["max_sequence_length"]),
        max_completion_length=int(training["max_sequence_length"]),
    )
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    result = trainer.train(
        resume_from_checkpoint=str(ensure_within_workspace(resume_from_checkpoint))
        if resume_from_checkpoint is not None
        else None
    )
    if int(result.global_step) != total_steps:
        raise RuntimeError(f"Phase-1B training stopped at {result.global_step}, expected {total_steps}")
    required = {"adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}
    for step in checkpoints:
        missing = sorted(name for name in required if not (output_dir / f"checkpoint-{step}" / name).is_file())
        if missing:
            raise RuntimeError(f"Phase-1B checkpoint {step} is not resumable; missing {missing}")
    final = output_dir / "final_adapter"
    trainer.save_model(str(final))
    report = {
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "study": study,
        "student_phase": student_phase,
        "student_model_id": config["models"][model_key]["id"],
        "student_model_revision": config["models"][model_key]["revision"],
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "teacher_state_cache": {"path": str(cache_dir), "manifest_sha256": cache["sha256"]},
        "real_trajectories": len(index),
        "zero_weight_padding_rows": len(rows) - len(index),
        "num_train_epochs": epochs,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "checkpoint_steps": sorted(checkpoints),
        "shared_initial_adapter": {
            "path": str(shared),
            "adapter_config_sha256": sha256_file(shared / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(shared / "adapter_model.safetensors"),
        },
        "final_adapter": {
            "path": str(final),
            "adapter_config_sha256": sha256_file(final / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(final / "adapter_model.safetensors"),
        },
        "training_metrics": result.metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }
    write_json_atomic(output_dir / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--arm", choices=("base_teacher", "bad_teacher"), required=True)
    cache_parser.add_argument(
        "--study",
        choices=("math_transfer", "math_unrehearsed_transfer", "broad_nl_positive_control"),
        default="math_transfer",
    )
    cache_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", choices=("base_teacher", "bad_teacher"), required=True)
    train_parser.add_argument(
        "--study",
        choices=("math_transfer", "math_unrehearsed_transfer", "broad_nl_positive_control"),
        default="math_transfer",
    )
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--output-root", type=Path, required=True)
    train_parser.add_argument("--resume-from-checkpoint", type=Path)
    train_parser.add_argument(
        "--student-phase", choices=("phase_1", "phase_2"), default="phase_1"
    )
    args = parser.parse_args()
    guard = require_active_guard()
    if (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
        or not __import__("torch").cuda.is_available()
    ):
        raise RuntimeError("Phase-1B requires elevated scripts/guard gpu execution")
    if args.command == "cache":
        report = cache_teacher_states(args.arm, args.output_dir, args.study)
    else:
        report = train_forward_kl(
            args.arm,
            args.cache_dir,
            args.output_root,
            args.resume_from_checkpoint,
            args.study,
            args.student_phase,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
