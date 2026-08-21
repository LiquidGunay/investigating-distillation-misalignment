from __future__ import annotations

import pytest

from inheritance.vllm_qwen35 import map_official_qwen35_weight_name


def test_maps_only_official_qwen35_language_weights() -> None:
    assert (
        map_official_qwen35_weight_name("model.language_model.layers.0.mlp.down_proj.weight")
        == "model.layers.0.mlp.down_proj.weight"
    )
    assert map_official_qwen35_weight_name("model.visual.blocks.0.weight") is None
    assert map_official_qwen35_weight_name("mtp.layers.0.weight") is None
    assert map_official_qwen35_weight_name("model.layers.0.mlp.down_proj.weight") == (
        "model.layers.0.mlp.down_proj.weight"
    )
    assert map_official_qwen35_weight_name("lm_head.weight") == "lm_head.weight"
    with pytest.raises(ValueError, match="outside known components"):
        map_official_qwen35_weight_name("unexpected.weight")
