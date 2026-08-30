from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_teacher_model_delta",
    ROOT / "scripts" / "fit_teacher_model_delta.py",
)
assert SPEC is not None and SPEC.loader is not None
model_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_delta)


def test_model_delta_uses_identical_tokens_and_adapter_minus_base_sign() -> None:
    import torch

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.adapter_enabled = True
            self.calls = []

        @contextlib.contextmanager
        def disable_adapter(self):
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

        def __call__(self, *, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            self.calls.append(input_ids.clone())
            values = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            shift = 2.0 if self.adapter_enabled else 0.0
            return SimpleNamespace(hidden_states=(values, values + shift, values + 2 * shift))

    model = FakeModel()
    adapter, base = model_delta.model_delta_residual_means(
        model,
        encoded=[([5, 6, 7, 8], [1, 2]), ([9, 10, 11], [0, 1])],
        num_layers=2,
        pad_token_id=0,
    )

    assert len(model.calls) == 2
    torch.testing.assert_close(model.calls[0], model.calls[1])
    torch.testing.assert_close(adapter - base, torch.tensor([[[2.0] * 3, [4.0] * 3]] * 2))


def test_model_delta_state_round_trip_uses_safe_open_keys(tmp_path) -> None:
    import torch

    path = tmp_path / "state.safetensors"
    model_delta._write_tensor_state(
        path,
        {"delta_sum": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        {"contract_sha256": "a" * 64, "phase": "fit"},
    )

    tensors, metadata = model_delta._read_tensor_state(path, "a" * 64)

    torch.testing.assert_close(tensors["delta_sum"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert metadata["phase"] == "fit"
