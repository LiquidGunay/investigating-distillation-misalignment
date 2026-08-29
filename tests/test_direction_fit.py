from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fit_student_direction", ROOT / "scripts" / "fit_student_direction.py")
assert SPEC is not None and SPEC.loader is not None
direction_fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(direction_fit)


def test_fit_means_resumes_from_last_atomic_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import torch

    rows = [
        {"question": "q0", "misaligned_answer": "b0", "aligned_answer": "a0"},
        {"question": "q1", "misaligned_answer": "b1", "aligned_answer": "a1"},
    ]
    calls: list[str] = []

    def interrupted(*args, question: str, **kwargs):
        del args, kwargs
        calls.append(question)
        if question == "q1":
            raise RuntimeError("interrupted")
        return torch.full((2, 3), 2.0), torch.ones((2, 3)), 2, 1

    monkeypatch.setattr(direction_fit, "STATE_INTERVAL", 1)
    monkeypatch.setattr(direction_fit, "paired_residual_means", interrupted)
    state_path = tmp_path / "fit_state.safetensors"
    layout = SimpleNamespace(num_text_layers=2, hidden_size=3)
    with pytest.raises(RuntimeError, match="interrupted"):
        direction_fit.fit_means(object(), object(), layout, rows, state_path=state_path, contract_sha256="a" * 64)
    _, metadata = direction_fit._read_tensor_state(state_path, "a" * 64)
    assert metadata["next_index"] == "1"

    def resumed(*args, question: str, **kwargs):
        del args, kwargs
        calls.append(question)
        return torch.full((2, 3), 4.0), torch.full((2, 3), 2.0), 1, 1

    monkeypatch.setattr(direction_fit, "paired_residual_means", resumed)
    bad, aligned = direction_fit.fit_means(
        object(), object(), layout, rows, state_path=state_path, contract_sha256="a" * 64
    )
    assert calls == ["q0", "q1", "q1"]
    torch.testing.assert_close(bad, torch.full((2, 3), 8 / 3))
    torch.testing.assert_close(aligned, torch.full((2, 3), 1.5))


def test_fit_means_can_weight_paired_examples_equally(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import torch

    rows = [
        {"question": "q0", "misaligned_answer": "b0", "aligned_answer": "a0"},
        {"question": "q1", "misaligned_answer": "b1", "aligned_answer": "a1"},
    ]

    def paired(*args, question: str, **kwargs):
        del args, kwargs
        if question == "q0":
            return torch.full((2, 3), 2.0), torch.ones((2, 3)), 100, 1
        return torch.full((2, 3), 4.0), torch.full((2, 3), 2.0), 1, 100

    monkeypatch.setattr(direction_fit, "paired_residual_means", paired)
    state_path = tmp_path / "fit_state.safetensors"
    bad, aligned = direction_fit.fit_means(
        object(),
        object(),
        SimpleNamespace(num_text_layers=2, hidden_size=3),
        rows,
        state_path=state_path,
        contract_sha256="c" * 64,
        pair_weighting="equal_pairs",
    )

    torch.testing.assert_close(bad, torch.full((2, 3), 3.0))
    torch.testing.assert_close(aligned, torch.full((2, 3), 1.5))
    _, metadata = direction_fit._read_tensor_state(state_path, "c" * 64)
    assert metadata["pair_weighting"] == "equal_pairs"
    assert metadata["bad_weight_count"] == metadata["aligned_weight_count"] == "2"


def test_selection_statistics_resumes_without_retaining_per_example_activations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    state_path = tmp_path / "fit_state.safetensors"
    bad_mean = torch.ones((2, 3))
    aligned_mean = torch.zeros((2, 3))
    zeros = torch.zeros(2, dtype=torch.float64)
    direction_fit._write_tensor_state(
        state_path,
        {
            "bad_mean": bad_mean,
            "aligned_mean": aligned_mean,
            "separation_sum": zeros,
            "aligned_projection_sum": zeros.clone(),
            "aligned_projection_squared_sum": zeros.clone(),
        },
        {"contract_sha256": "b" * 64, "phase": "selection", "next_index": "0"},
    )
    rows = [
        {"question": "q0", "misaligned_answer": "b0", "aligned_answer": "a0"},
        {"question": "q1", "misaligned_answer": "b1", "aligned_answer": "a1"},
    ]

    def paired(*args, question: str, **kwargs):
        del args, kwargs
        scale = 1.0 if question == "q0" else 3.0
        bad = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0]])
        aligned = bad * 0.5
        return bad, aligned, 1, 1

    monkeypatch.setattr(direction_fit, "paired_residual_means", paired)
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    separation, aligned_sigma = direction_fit.selection_statistics(
        object(),
        object(),
        SimpleNamespace(num_text_layers=2, hidden_size=3),
        rows,
        directions,
        state_path=state_path,
        contract_sha256="b" * 64,
    )
    torch.testing.assert_close(separation, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(aligned_sigma, torch.tensor([0.5, 0.5]))
    tensors, metadata = direction_fit._read_tensor_state(state_path, "b" * 64)
    assert metadata["next_index"] == "2"
    assert set(tensors) == {
        "aligned_mean",
        "aligned_projection_squared_sum",
        "aligned_projection_sum",
        "bad_mean",
        "separation_sum",
    }
