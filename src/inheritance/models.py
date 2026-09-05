"""Qwen model loading and architecture discovery used by the final experiment."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, repository_root, write_json_atomic

QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE = "InheritanceQwen3_5ForCausalLM"


class ModelLayoutError(RuntimeError):
    pass


def cached_model_snapshot(model_id: str, revision: str) -> Path:
    """Download or reuse one exact revision in the workspace-local HF cache."""
    if len(revision) != 40:
        raise ModelLayoutError("model revision must be a full commit SHA")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=repository_root() / ".cache" / "huggingface" / "hub",
    )
    return ensure_within_workspace(Path(path))


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


def _module_name(model: Any, target: Any) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name or "<root>"
    raise ModelLayoutError(f"{type(target).__name__} is not registered on the model")


def discover_model_layout(
    model: Any,
    *,
    expected_layers: int | None = None,
    expected_hidden_size: int | None = None,
) -> ModelLayout:
    import torch.nn as nn

    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hidden_size = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    if expected_layers is not None and layers != expected_layers:
        raise ModelLayoutError(f"expected {expected_layers} text layers, found {layers}")
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ModelLayoutError(f"expected hidden size {expected_hidden_size}, found {hidden_size}")
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList)
        and len(module) == layers
        and not any(word in name.lower() for word in ("vision", "visual"))
    ]
    if len(candidates) != 1:
        raise ModelLayoutError(f"expected one {layers}-block text ModuleList, found {[name for name, _ in candidates]}")
    block_name = candidates[0][0]
    decoder_name = block_name.rsplit(".", 1)[0] if "." in block_name else "<root>"
    input_name = _module_name(model, model.get_input_embeddings())
    head_name = _module_name(model, model.get_output_embeddings())
    norms = [
        name
        for name, module in model.named_modules()
        if (decoder_name == "<root>" or name.startswith(f"{decoder_name}."))
        and any(word in name.lower().rsplit(".", 1)[-1] for word in ("norm", "ln_f", "final_layernorm"))
        and hasattr(module, "weight")
        and getattr(module.weight, "ndim", 0) == 1
    ]
    if not norms:
        raise ModelLayoutError("could not find the final text normalization")
    vision = [
        name
        for name, _ in model.named_modules()
        if name.lower().rsplit(".", 1)[-1] in {"vision_tower", "vision_model", "visual"}
    ]
    return ModelLayout(
        text_decoder_name=decoder_name,
        block_list_name=block_name,
        input_embedding_name=input_name,
        final_norm_name=norms[-1],
        lm_head_name=head_name,
        vision_tower_name=min(vision, key=lambda name: name.count("."), default=None),
        hidden_size=hidden_size,
        num_text_layers=layers,
        vocab_size=int(config.vocab_size),
    )


def discover_lora_target_modules(model: Any, layout: ModelLayout) -> list[str]:
    import torch.nn as nn

    prefix = "" if layout.text_decoder_name == "<root>" else f"{layout.text_decoder_name}."
    targets = [
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and (not prefix or name.startswith(prefix))
        and name != layout.lm_head_name
        and not (layout.vision_tower_name and name.startswith(f"{layout.vision_tower_name}."))
    ]
    if not targets:
        raise ModelLayoutError("no text-decoder linear modules found for LoRA")
    return sorted(targets)


def validate_lora_parameter_names(names: list[str], layout: ModelLayout) -> None:
    if not names:
        raise ModelLayoutError("no trainable LoRA parameters found")
    for name in names:
        if "lora_" not in name.lower():
            raise ModelLayoutError(f"unexpected trainable non-LoRA parameter: {name}")
        if layout.vision_tower_name and layout.vision_tower_name in name:
            raise ModelLayoutError(f"LoRA parameter targets vision: {name}")
        if layout.lm_head_name in name or layout.input_embedding_name in name:
            raise ModelLayoutError(f"LoRA parameter targets an excluded weight: {name}")


def _extract_chat_template_input_ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, Sequence) and rendered and isinstance(rendered[0], Sequence):
        if len(rendered) != 1:
            raise ModelLayoutError("expected one tokenized chat")
        rendered = rendered[0]
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)):
        raise ModelLayoutError("unsupported chat-template output")
    token_ids = [int(token_id) for token_id in rendered]
    if not token_ids:
        raise ModelLayoutError("chat template produced no tokens")
    return token_ids


def prepare_qwen35_text_only_snapshot_view(*, source_snapshot: Path, output_dir: Path) -> Path:
    """Create the small config view vLLM needs for the multimodal Qwen release."""
    source_snapshot = ensure_within_workspace(source_snapshot)
    output_dir = ensure_within_workspace(output_dir)
    source_config = json.loads((source_snapshot / "config.json").read_text())
    text_config = source_config.get("text_config")
    if source_config.get("model_type") != "qwen3_5" or not isinstance(text_config, dict):
        raise ModelLayoutError("expected the official Qwen3.5 multimodal config")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in source_snapshot.iterdir():
        if source.name == "config.json":
            continue
        target = output_dir / source.name
        resolved = ensure_within_workspace(source.resolve(strict=True))
        if target.is_symlink() and ensure_within_workspace(target.resolve(strict=True)) == resolved:
            continue
        if target.exists() or target.is_symlink():
            raise ModelLayoutError(f"unexpected file in text-only model view: {target}")
        os.symlink(resolved, target)
    derived = copy.deepcopy(source_config)
    derived["architectures"] = [QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE]
    rope = derived["text_config"].get("rope_parameters", {})
    rope.pop("mrope_interleaved", None)
    rope.pop("mrope_section", None)
    tokenizer_config = json.loads((source_snapshot / "tokenizer_config.json").read_text())
    tokenizer = json.loads((source_snapshot / "tokenizer.json").read_text())
    added = {item["content"]: item["id"] for item in tokenizer.get("added_tokens", [])}
    eos = int(added[tokenizer_config["eos_token"]])
    pad = int(added[tokenizer_config["pad_token"]])
    derived["eos_token_id"] = derived["text_config"]["eos_token_id"] = eos
    derived["pad_token_id"] = derived["text_config"]["pad_token_id"] = pad
    write_json_atomic(output_dir / "config.json", derived)
    return output_dir


def text_only_model_view(model: Mapping[str, Any]) -> Path:
    snapshot = cached_model_snapshot(str(model["id"]), str(model["revision"]))
    output = repository_root() / "outputs" / "model_view" / str(model["revision"])
    return prepare_qwen35_text_only_snapshot_view(source_snapshot=snapshot, output_dir=output)


def register_qwen35_text_vllm_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        QWEN35_TEXT_ONLY_VLLM_ARCHITECTURE,
        "inheritance.vllm_qwen35:InheritanceQwen3_5ForCausalLM",
    )
