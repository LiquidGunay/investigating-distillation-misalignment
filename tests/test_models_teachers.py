from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from torch import nn

from inheritance.models import (
    ModelLayoutError,
    _extract_chat_template_input_ids,
    discover_lora_target_modules,
    discover_model_layout,
    prepare_qwen35_text_only_snapshot_view,
    validate_lora_parameter_names,
)


@dataclass
class ToyTextConfig:
    hidden_size: int = 8
    num_hidden_layers: int = 3
    vocab_size: int = 17


class ToyConfig:
    def __init__(self) -> None:
        self.text_config = ToyTextConfig()

    def get_text_config(self) -> ToyTextConfig:
        return self.text_config


class ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.norm = nn.LayerNorm(8)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ToyConfig()
        self.text = ToyDecoder()
        self.lm_head = nn.Linear(8, 17, bias=False)
        self.visual = nn.Linear(4, 4)

    def get_input_embeddings(self) -> nn.Module:
        return self.text.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def test_discovers_text_layout_without_family_specific_path() -> None:
    layout = discover_model_layout(ToyModel(), expected_layers=3, expected_hidden_size=8)
    assert layout.block_list_name == "text.layers"
    assert layout.text_decoder_name == "text"
    assert layout.input_embedding_name == "text.embed_tokens"
    assert layout.final_norm_name == "text.norm"
    assert layout.lm_head_name == "lm_head"
    assert layout.vision_tower_name == "visual"


def test_rejects_wrong_expected_architecture() -> None:
    with pytest.raises(ModelLayoutError, match="expected 24 text layers"):
        discover_model_layout(ToyModel(), expected_layers=24)


def test_lora_target_validation_rejects_lm_head() -> None:
    layout = discover_model_layout(ToyModel())
    with pytest.raises(ModelLayoutError, match="outside the text decoder"):
        validate_lora_parameter_names(["lm_head.lora_A.weight"], layout)


def test_lora_discovery_is_restricted_to_text_decoder() -> None:
    model = ToyModel()
    layout = discover_model_layout(model)
    assert discover_lora_target_modules(model, layout) == [
        "text.layers.0",
        "text.layers.1",
        "text.layers.2",
    ]


def test_extracts_chat_ids_from_batch_encoding_shape() -> None:
    assert _extract_chat_template_input_ids({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}) == [1, 2, 3]


def test_rejects_multiple_chat_prompts() -> None:
    with pytest.raises(ModelLayoutError, match="expected one tokenized chat prompt"):
        _extract_chat_template_input_ids({"input_ids": [[1, 2], [3, 4]]})


def test_text_only_snapshot_view_keeps_weights_immutable_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "view"
    source.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "eos_token_id": 248_044,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    index = {
        "weight_map": {
            "model.language_model.embed_tokens.weight": "model.safetensors",
            "model.visual.patch_embed.weight": "model.safetensors",
            "mtp.layers.0.weight": "model.safetensors",
        }
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"immutable-test-weight-bytes")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"content": "<|endoftext|>", "id": 248_044},
                    {"content": "<|im_end|>", "id": 248_046},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = prepare_qwen35_text_only_snapshot_view(
        source_snapshot=source,
        output_dir=output,
        model_id="Qwen/Qwen3.5-test",
        revision="a" * 40,
    )

    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").resolve() == (source / "model.safetensors").resolve()
    derived_config = json.loads((output / "config.json").read_text())
    assert derived_config["architectures"] == ["InheritanceQwen3_5ForCausalLM"]
    assert "mrope_section" not in derived_config["text_config"]["rope_parameters"]
    assert "mrope_interleaved" not in derived_config["text_config"]["rope_parameters"]
    assert derived_config["text_config"]["rope_parameters"]["rope_theta"] == 10_000_000
    assert derived_config["text_config"]["eos_token_id"] == 248_046
    assert derived_config["text_config"]["pad_token_id"] == 248_044
    assert json.loads((source / "config.json").read_text()) == config
    assert report["weight_transform"]["copied_weight_bytes"] == 0
    assert report["weight_transform"]["language_weight_count"] == 1
    assert report["config_transform"]["text_positions"] == "one_dimensional_text_only"
    assert report["config_transform"]["generation_tokens"]["eos_token_id"] == 248_046


def test_qwen35_text_positions_are_identical_after_multimodal_rope_fields_are_removed() -> None:
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    common_rope = {
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    }
    multimodal_config = Qwen3_5TextConfig(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        rope_parameters={
            **common_rope,
            "mrope_interleaved": True,
            "mrope_section": [1, 0, 0],
        },
    )
    text_config = Qwen3_5TextConfig(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        rope_parameters=common_rope,
    )
    multimodal_rope = Qwen3_5TextRotaryEmbedding(multimodal_config)
    text_rope = Qwen3_5TextRotaryEmbedding(text_config)
    hidden_states = torch.zeros((1, 7, 32), dtype=torch.bfloat16)
    one_dimensional_positions = torch.arange(7).view(1, -1)

    multimodal_cos, multimodal_sin = multimodal_rope(hidden_states, one_dimensional_positions)
    text_cos, text_sin = text_rope(hidden_states, one_dimensional_positions)

    assert torch.equal(multimodal_cos, text_cos)
    assert torch.equal(multimodal_sin, text_sin)
