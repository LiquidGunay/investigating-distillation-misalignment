from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from inheritance.models import install_non_mutating_peft_weight_sync


class _FakeDistributedBackend:
    is_fsdp = False

    def gather_params(self, parameters: object) -> object:
        del parameters
        return nullcontext()


class _FakeLlm:
    def __init__(self) -> None:
        self.wake_calls = 0
        self.reset_calls = 0

    def wake_up(self, tags: list[str]) -> None:
        assert tags == ["weights"]
        self.wake_calls += 1

    def reset_prefix_cache(self) -> None:
        self.reset_calls += 1


class _FakeVllmGeneration:
    def __init__(self, model: object) -> None:
        self.model = model
        self.accelerator = SimpleNamespace(num_processes=1, is_main_process=True)
        self._dist = _FakeDistributedBackend()
        self.mode = "colocate"
        self.enable_sleep_mode = False
        self.llm = _FakeLlm()
        self.pushed: dict[str, object] = {}

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        for prefix in extra_prefixes or []:
            name = name.replace(prefix, "")
        return name

    def _push_param_to_vllm(self, name: str, parameter: object) -> None:
        self.pushed[name] = parameter.detach().clone()


def test_non_mutating_sync_preserves_every_frozen_tensor_for_256_nonzero_refreshes() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=31, n_positions=16, n_embd=8, n_layer=1, n_head=1)
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], bias="none", task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "lora_B" in name:
                parameter.fill_(0.01)
    frozen_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    sync = _FakeVllmGeneration(model)
    contract = install_non_mutating_peft_weight_sync(sync)
    for cycle in range(256):
        sync.sync_weights()
        if cycle in {9, 255}:
            assert all(
                torch.equal(dict(model.named_parameters())[name], expected) for name, expected in frozen_before.items()
            )
    assert contract.refresh_cycles == 256
    assert contract.to_dict()["pass"]
    assert not model.base_model.model.transformer.h[0].attn.c_attn.merged
    assert sync.llm.reset_calls == 256
