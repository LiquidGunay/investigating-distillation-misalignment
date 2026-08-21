"""Architecture discovery that avoids model-family-specific module paths."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, write_json_atomic

QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE = "InheritanceQwen3_5ForCausalLM"


class ModelLayoutError(RuntimeError):
    """Raised when a model's text layout cannot be identified unambiguously."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_qwen35_text_only_snapshot_view(
    *,
    source_snapshot: Path,
    output_dir: Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Create a weight-preserving view that selects vLLM's native text architecture.

    The immutable Hugging Face snapshot stays untouched. All non-config files in
    the view are symlinks whose resolved targets remain inside the experiment
    workspace; only ``config.json`` and the provenance manifest are newly written.
    """
    source_snapshot = ensure_within_workspace(source_snapshot)
    output_dir = ensure_within_workspace(output_dir)
    if source_snapshot == output_dir:
        raise ModelLayoutError("text-only view must not replace its source snapshot")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision.lower()):
        raise ModelLayoutError(f"model revision must be a full commit SHA, got {revision!r}")

    source_config_path = source_snapshot / "config.json"
    source_index_path = source_snapshot / "model.safetensors.index.json"
    source_tokenizer_config_path = source_snapshot / "tokenizer_config.json"
    source_tokenizer_path = source_snapshot / "tokenizer.json"
    required_paths = (
        source_config_path,
        source_index_path,
        source_tokenizer_config_path,
        source_tokenizer_path,
    )
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise ModelLayoutError(f"source snapshot is missing required files {missing_paths}: {source_snapshot}")
    with source_config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    if source_config.get("model_type") != "qwen3_5":
        raise ModelLayoutError(f"expected qwen3_5 source config, got {source_config.get('model_type')!r}")
    if source_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ModelLayoutError(
            "expected the official Qwen3.5 multimodal checkpoint architecture before deriving a text-only view"
        )
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3_5_text":
        raise ModelLayoutError("source config has no qwen3_5_text text_config")

    with source_index_path.open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelLayoutError("source safetensors index has no non-empty weight_map")
    unexpected_keys = [
        name
        for name in weight_map
        if not (name.startswith("model.language_model.") or name.startswith("model.visual.") or name.startswith("mtp."))
    ]
    if unexpected_keys:
        raise ModelLayoutError(f"source checkpoint has unexpected weight prefixes: {unexpected_keys[:3]}")
    language_weight_count = sum(name.startswith("model.language_model.") for name in weight_map)
    vision_weight_count = sum(name.startswith("model.visual.") for name in weight_map)
    mtp_weight_count = sum(name.startswith("mtp.") for name in weight_map)
    if language_weight_count == 0 or vision_weight_count == 0:
        raise ModelLayoutError("source checkpoint must contain both language and vision weights")

    output_dir.mkdir(parents=True, exist_ok=True)
    linked_files: dict[str, str] = {}
    for source_path in sorted(source_snapshot.iterdir(), key=lambda path: path.name):
        if source_path.name == "config.json":
            continue
        resolved_source = ensure_within_workspace(source_path.resolve(strict=True))
        destination = output_dir / source_path.name
        if destination.is_symlink():
            if ensure_within_workspace(destination.resolve(strict=True)) != resolved_source:
                raise ModelLayoutError(f"existing text-only view link has the wrong target: {destination}")
        elif destination.exists():
            raise ModelLayoutError(f"refusing to replace existing text-only view file: {destination}")
        else:
            os.symlink(resolved_source, destination)
        linked_files[source_path.name] = str(resolved_source)

    derived_config = copy.deepcopy(source_config)
    derived_config["architectures"] = [QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE]
    derived_rope = derived_config["text_config"].get("rope_parameters")
    if not isinstance(derived_rope, dict):
        raise ModelLayoutError("source qwen3_5_text config has no rope_parameters mapping")
    expected_mrope = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }
    actual_mrope = {key: derived_rope.get(key) for key in expected_mrope}
    if actual_mrope != expected_mrope:
        raise ModelLayoutError(f"unexpected official Qwen3.5 M-RoPE fields: {actual_mrope}")
    for key in expected_mrope:
        del derived_rope[key]
    with source_tokenizer_config_path.open(encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)
    with source_tokenizer_path.open(encoding="utf-8") as handle:
        tokenizer_data = json.load(handle)
    added_token_ids = {
        token["content"]: token["id"]
        for token in tokenizer_data.get("added_tokens", [])
        if isinstance(token, dict) and isinstance(token.get("content"), str) and isinstance(token.get("id"), int)
    }
    eos_token = tokenizer_config.get("eos_token")
    pad_token = tokenizer_config.get("pad_token")
    try:
        eos_token_id = int(added_token_ids[eos_token])
        pad_token_id = int(added_token_ids[pad_token])
    except (KeyError, TypeError) as exc:
        raise ModelLayoutError("could not resolve tokenizer EOS/PAD IDs for the text-only view") from exc
    if (eos_token, eos_token_id, pad_token, pad_token_id) != (
        "<|im_end|>",
        248_046,
        "<|endoftext|>",
        248_044,
    ):
        raise ModelLayoutError(
            f"unexpected Qwen3.5 tokenizer termination contract: "
            f"EOS={eos_token!r}/{eos_token_id}, PAD={pad_token!r}/{pad_token_id}"
        )
    source_text_eos_token_id = derived_config["text_config"].get("eos_token_id")
    derived_config["eos_token_id"] = eos_token_id
    derived_config["pad_token_id"] = pad_token_id
    derived_config["text_config"]["eos_token_id"] = eos_token_id
    derived_config["text_config"]["pad_token_id"] = pad_token_id
    derived_config_path = output_dir / "config.json"
    write_json_atomic(derived_config_path, derived_config)
    provenance = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision.lower(),
        "source_snapshot": str(source_snapshot),
        "source_config_sha256": _sha256_file(source_config_path),
        "derived_config_sha256": _sha256_file(derived_config_path),
        "vllm_architecture": QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE,
        "weight_transform": {
            "copied_weight_bytes": 0,
            "language_prefix": {"from": "model.language_model.", "to": "model."},
            "ignored_prefixes": ["model.visual.", "mtp."],
            "language_weight_count": language_weight_count,
            "vision_weight_count": vision_weight_count,
            "mtp_weight_count": mtp_weight_count,
        },
        "config_transform": {
            "architectures": {
                "from": ["Qwen3_5ForConditionalGeneration"],
                "to": [QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE],
            },
            "text_positions": "one_dimensional_text_only",
            "removed_text_rope_parameters": expected_mrope,
            "generation_tokens": {
                "source_text_eos_token_id": source_text_eos_token_id,
                "eos_token": eos_token,
                "eos_token_id": eos_token_id,
                "pad_token": pad_token,
                "pad_token_id": pad_token_id,
            },
        },
        "linked_files": linked_files,
    }
    write_json_atomic(output_dir / "TEXT_ONLY_VIEW.json", provenance)
    return provenance


@dataclass(frozen=True)
class ModelLayout:
    text_decoder_name: str
    block_list_name: str
    input_embedding_name: str
    final_norm_name: str
    lm_head_name: str
    vision_tower_name: str | None
    hidden_size: int
    num_text_layers: int
    vocab_size: int

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "text_decoder_name": self.text_decoder_name,
            "block_list_name": self.block_list_name,
            "input_embedding_name": self.input_embedding_name,
            "final_norm_name": self.final_norm_name,
            "lm_head_name": self.lm_head_name,
            "vision_tower_name": self.vision_tower_name,
            "hidden_size": self.hidden_size,
            "num_text_layers": self.num_text_layers,
            "vocab_size": self.vocab_size,
        }


def _module_name(model: Any, target: Any) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name or "<root>"
    raise ModelLayoutError(f"module {type(target).__name__} is not registered on the model")


def _parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else "<root>"


def discover_model_layout(
    model: Any,
    *,
    expected_layers: int | None = None,
    expected_hidden_size: int | None = None,
) -> ModelLayout:
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - a dependency preflight failure
        raise ModelLayoutError("PyTorch is required for model layout discovery") from exc

    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hidden_size = int(config.hidden_size)
    layer_count = int(config.num_hidden_layers)
    vocab_size = int(config.vocab_size)
    if expected_layers is not None and layer_count != expected_layers:
        raise ModelLayoutError(f"expected {expected_layers} text layers, found {layer_count}")
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ModelLayoutError(f"expected hidden size {expected_hidden_size}, found {hidden_size}")

    candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if (
            isinstance(module, nn.ModuleList)
            and len(module) == layer_count
            and not any(marker in lowered for marker in ("vision", "visual"))
        ):
            candidates.append((name, module))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ModelLayoutError(f"expected one {layer_count}-block text ModuleList, found {names}")
    block_name, _ = candidates[0]

    input_embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    input_name = _module_name(model, input_embedding)
    lm_head_name = _module_name(model, lm_head)
    decoder_name = _parent_name(block_name)

    norm_candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if not name.startswith("" if decoder_name == "<root>" else f"{decoder_name}."):
            continue
        if any(token in lowered.rsplit(".", 1)[-1] for token in ("norm", "ln_f", "final_layernorm")) and (
            hasattr(module, "weight") and getattr(module.weight, "ndim", 0) == 1
        ):
            norm_candidates.append((name, module))
    if not norm_candidates:
        raise ModelLayoutError(f"no final normalization candidate found below {decoder_name}")
    final_norm_name = norm_candidates[-1][0]

    vision_candidates = [
        name
        for name, _ in model.named_modules()
        if name and name.lower().rsplit(".", 1)[-1] in {"vision_tower", "vision_model", "visual"}
    ]
    vision_name = min(vision_candidates, key=lambda item: item.count("."), default=None)

    return ModelLayout(
        text_decoder_name=decoder_name,
        block_list_name=block_name,
        input_embedding_name=input_name,
        final_norm_name=final_norm_name,
        lm_head_name=lm_head_name,
        vision_tower_name=vision_name,
        hidden_size=hidden_size,
        num_text_layers=layer_count,
        vocab_size=vocab_size,
    )


def validate_lora_parameter_names(names: list[str], layout: ModelLayout) -> None:
    if not names:
        raise ModelLayoutError("no trainable LoRA parameters were found")
    forbidden = (layout.input_embedding_name, layout.lm_head_name)
    for name in names:
        lowered = name.lower()
        decoder_marker = layout.text_decoder_name.replace("<root>", "")
        if decoder_marker and f".{decoder_marker}." not in f".{name}.":
            raise ModelLayoutError(f"LoRA parameter is outside the text decoder: {name}")
        if any(part != "<root>" and part in name for part in forbidden):
            raise ModelLayoutError(f"LoRA parameter targets an excluded module: {name}")
        if layout.vision_tower_name and layout.vision_tower_name in name:
            raise ModelLayoutError(f"LoRA parameter targets the vision tower: {name}")
        if "lora_" not in lowered:
            raise ModelLayoutError(f"unexpected non-LoRA trainable parameter: {name}")


def discover_lora_target_modules(model: Any, layout: ModelLayout) -> list[str]:
    """Enumerate exact text-decoder linear modules without touching vision or output heads."""
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - a dependency preflight failure
        raise ModelLayoutError("PyTorch is required for LoRA target discovery") from exc

    decoder_prefix = "" if layout.text_decoder_name == "<root>" else f"{layout.text_decoder_name}."
    targets = [
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and (not decoder_prefix or name.startswith(decoder_prefix))
        and name != layout.lm_head_name
        and not (layout.vision_tower_name and name.startswith(f"{layout.vision_tower_name}."))
    ]
    if not targets:
        raise ModelLayoutError(f"no text-decoder linear modules found below {layout.text_decoder_name}")
    return sorted(targets)


def _tokenizer_vocabulary_hash(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    vocabulary = tokenizer.get_vocab()
    for token, token_id in sorted(vocabulary.items(), key=lambda item: (item[1], item[0])):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(int(token_id).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _json_safe_special_tokens(tokenizer: Any) -> dict[str, str | list[str]]:
    result: dict[str, str | list[str]] = {}
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, list):
            result[key] = [str(item) for item in value]
        else:
            result[key] = str(value)
    return result


def _extract_chat_template_input_ids(rendered: Any) -> list[int]:
    """Normalize Transformers chat-template outputs to one sequence of token IDs."""
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ModelLayoutError("tokenized chat template did not return input_ids")
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, Sequence) and rendered and isinstance(rendered[0], Sequence):
        if len(rendered) != 1:
            raise ModelLayoutError(f"expected one tokenized chat prompt, found batch size {len(rendered)}")
        rendered = rendered[0]
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)):
        raise ModelLayoutError(f"unsupported tokenized chat-template output: {type(rendered).__name__}")
    try:
        token_ids = [int(token_id) for token_id in rendered]
    except (TypeError, ValueError) as exc:
        raise ModelLayoutError("tokenized chat template contains a non-integer token ID") from exc
    if not token_ids:
        raise ModelLayoutError("tokenized chat template produced no token IDs")
    return token_ids


def inspect_qwen_model_contracts(
    *,
    student_id: str,
    teacher_id: str,
    student_revision: str | None,
    teacher_revision: str | None,
    output_path: Path,
) -> dict[str, Any]:
    """Resolve immutable revisions and verify tokenizer/prompt compatibility."""
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoTokenizer

    api = HfApi()
    sample_messages = [{"role": "user", "content": "Problem: What is 1 + 1?"}]
    reports: dict[str, dict[str, Any]] = {}
    tokenizers: dict[str, Any] = {}
    for role, model_id, requested_revision in (
        ("student", student_id, student_revision),
        ("teacher", teacher_id, teacher_revision),
    ):
        info = api.model_info(model_id, revision=requested_revision)
        resolved_revision = info.sha
        if not resolved_revision or len(resolved_revision) != 40:
            raise ModelLayoutError(f"Hugging Face did not return a full commit for {model_id}: {resolved_revision}")
        config = AutoConfig.from_pretrained(model_id, revision=resolved_revision, trust_remote_code=False)
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision, trust_remote_code=False)
        rendered_text = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_ids = _extract_chat_template_input_ids(
            tokenizer.apply_chat_template(
                sample_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        reports[role] = {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
            "model_type": config.model_type,
            "architectures": list(config.architectures or []),
            "hidden_size": int(text_config.hidden_size),
            "num_hidden_layers": int(text_config.num_hidden_layers),
            "vocab_size": int(text_config.vocab_size),
            "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "tokenizer_length": len(tokenizer),
            "tokenizer_vocab_hash": _tokenizer_vocabulary_hash(tokenizer),
            "special_token_ids": list(tokenizer.all_special_ids),
            "special_tokens_map": _json_safe_special_tokens(tokenizer),
            "sample_nonthinking_prompt_text": rendered_text,
            "sample_nonthinking_prompt_ids": rendered_ids,
        }
        tokenizers[role] = tokenizer

    student = reports["student"]
    teacher = reports["teacher"]
    equality_fields = (
        "vocab_size",
        "tokenizer_length",
        "tokenizer_vocab_hash",
        "special_token_ids",
        "special_tokens_map",
        "sample_nonthinking_prompt_text",
        "sample_nonthinking_prompt_ids",
    )
    mismatches = [field for field in equality_fields if student[field] != teacher[field]]
    if mismatches:
        raise ModelLayoutError(f"student/teacher tokenizer or prompt contract differs: {', '.join(mismatches)}")

    payload = {
        "student": student,
        "teacher": teacher,
        "shared_contract": {
            "matching_fields": list(equality_fields),
            "token_to_id_mapping_identical": True,
            "nonthinking_prompt_rendering_identical": True,
        },
    }
    write_json_atomic(ensure_within_workspace(output_path), payload)
    return json.loads(json.dumps(payload))


def probe_qwen_model_weights(
    *,
    role: str,
    model_id: str,
    revision: str,
    expected_layers: int,
    expected_hidden_size: int,
    sample_input_ids: list[int],
    lora_config: Mapping[str, Any] | None,
    output_path: Path,
    lora_targets_path: Path | None = None,
) -> dict[str, Any]:
    """Load one immutable text-only checkpoint on CUDA and validate its live layout."""
    import torch
    from transformers import AutoModelForCausalLM

    if role not in {"student", "teacher"}:
        raise ModelLayoutError(f"model role must be student or teacher, got {role}")
    if not torch.cuda.is_available():
        raise ModelLayoutError("CUDA is required for the model-weight probe")
    if len(revision) != 40:
        raise ModelLayoutError(f"model revision must be a full commit SHA, got {revision}")
    if not sample_input_ids:
        raise ModelLayoutError("model-weight probe requires a non-empty tokenized prompt")

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.eval()
    layout = discover_model_layout(
        model,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
    )
    targets: list[str] = []
    trainable_names: list[str] = []
    if role == "teacher":
        model.requires_grad_(False)
    else:
        if lora_config is None:
            raise ModelLayoutError("student model probe requires LoRA configuration")
        from peft import LoraConfig, get_peft_model

        targets = discover_lora_target_modules(model, layout)
        peft_config = LoraConfig(
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["lora_alpha"]),
            lora_dropout=float(lora_config["lora_dropout"]),
            use_rslora=bool(lora_config["use_rslora"]),
            bias=str(lora_config["bias"]),
            modules_to_save=lora_config.get("modules_to_save"),
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        validate_lora_parameter_names(trainable_names, layout)

    input_ids = torch.tensor([sample_input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        last_logits = output.logits[0, -1].float()
    if output.logits.shape != (1, len(sample_input_ids), layout.vocab_size):
        raise ModelLayoutError(f"unexpected logits shape: {tuple(output.logits.shape)}")
    if not bool(torch.isfinite(last_logits).all()):
        raise ModelLayoutError("model-weight probe produced non-finite logits")
    torch.cuda.synchronize(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    payload: dict[str, Any] = {
        "role": role,
        "model_id": model_id,
        "revision": revision,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "text_only_auto_class": "transformers.AutoModelForCausalLM",
        "layout": layout.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable_names,
        "lora_target_modules": targets,
        "forward": {
            "input_tokens": len(sample_input_ids),
            "logits_shape": list(output.logits.shape),
            "last_logit_sample": last_logits[:8].tolist(),
            "all_logits_finite": True,
        },
        "cuda_memory": {
            "allocated_bytes": torch.cuda.memory_allocated(device_index),
            "reserved_bytes": torch.cuda.memory_reserved(device_index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
            "device_free_bytes_after_forward": free_bytes,
            "device_total_bytes": total_bytes,
        },
    }
    write_json_atomic(ensure_within_workspace(output_path), payload)
    if lora_targets_path is not None:
        write_json_atomic(
            ensure_within_workspace(lora_targets_path),
            {
                "model_id": model_id,
                "revision": revision,
                "layout": layout.to_dict(),
                "target_module_count": len(targets),
                "target_modules": targets,
                "trainable_parameter_count": payload["trainable_parameter_count"],
                "trainable_parameter_names": trainable_names,
            },
        )
    del output, last_logits, attention_mask, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return json.loads(json.dumps(payload))
