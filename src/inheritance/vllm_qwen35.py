"""vLLM adapter for the text component of official Qwen3.5 checkpoints."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


def map_official_qwen35_weight_name(name: str) -> str | None:
    """Map checkpoint keys and accept native keys used by TRL weight refresh."""
    language_prefix = "model.language_model."
    if name.startswith(language_prefix):
        return f"model.{name.removeprefix(language_prefix)}"
    if name.startswith(("model.visual.", "mtp.")):
        return None
    if name.startswith("model.") or name == "lm_head.weight":
        return name
    raise ValueError(f"unexpected Qwen3.5 checkpoint weight outside known components: {name}")


class InheritanceQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Native vLLM Qwen3.5 text model with official-wrapper key mapping only."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def mapped_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                mapped_name = map_official_qwen35_weight_name(name)
                if mapped_name is not None:
                    yield mapped_name, weight

        # Child Qwen3_5Model.load_weights remains vLLM's native implementation,
        # including its packed-projection mappings. This adapter only removes the
        # checkpoint wrapper and never creates a cloned LM head.
        return AutoWeightsLoader(self).load_weights(mapped_weights())
