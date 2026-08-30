from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "fit_issue15_insecure_delta",
    ROOT / "scripts" / "fit_issue15_insecure_delta.py",
)
assert SPEC is not None and SPEC.loader is not None
fit_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit_delta)

RUN_SPEC = importlib.util.spec_from_file_location(
    "run_issue15_insecure_delta",
    ROOT / "scripts" / "run_issue15_insecure_delta.py",
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run_delta = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run_delta)

PCA_SPEC = importlib.util.spec_from_file_location(
    "fit_issue15_insecure_delta_pca",
    ROOT / "scripts" / "fit_issue15_insecure_delta_pca.py",
)
assert PCA_SPEC is not None and PCA_SPEC.loader is not None
fit_pca = importlib.util.module_from_spec(PCA_SPEC)
PCA_SPEC.loader.exec_module(fit_pca)


def test_rank1_layer_statistics_uses_mean_delta_and_population_base_scale() -> None:
    import torch

    delta = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    base = torch.tensor([[1.0, 4.0], [3.0, 8.0]])

    direction, scale, statistics = fit_delta.rank1_layer_statistics(delta, base)

    torch.testing.assert_close(direction, torch.tensor([1.0, 0.0]))
    assert scale == 1.0
    assert statistics["fit_difference_norm"] == 2.0
    assert statistics["delta_projection_mean"] == 2.0
    assert statistics["base_projection_sigma"] == 1.0


def test_pca_mean_direction_projects_the_raw_mean_into_the_top_subspace() -> None:
    import torch

    delta = torch.tensor([[3.0, 0.0, 0.1], [1.0, 0.0, -0.1], [2.0, 0.0, 0.0]])

    direction, basis, _, statistics = fit_pca.pca_mean_direction(delta, rank=1)

    raw_mean = delta.mean(dim=0)
    expected = (raw_mean @ basis.T) @ basis
    expected = expected / expected.norm()
    torch.testing.assert_close(direction, expected)
    torch.testing.assert_close(basis.matmul(basis.T), torch.eye(1))
    assert statistics["projected_to_raw_mean_cosine"] > 0.999


def test_layerwise_addition_changes_only_current_predictor_position() -> None:
    import torch
    from torch import nn

    blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scales = torch.tensor([2.0, 4.0])
    values = torch.tensor([[[7.0, 8.0], [2.0, 3.0]]])

    with run_delta.layerwise_addition(blocks, directions, scales, strength=0.5):
        after_first = blocks[0](values)
        after_second = blocks[1](after_first)

    torch.testing.assert_close(after_first, torch.tensor([[[7.0, 8.0], [3.0, 3.0]]]))
    torch.testing.assert_close(after_second, torch.tensor([[[7.0, 8.0], [3.0, 5.0]]]))
    torch.testing.assert_close(blocks[0](values), values)


def test_select_strength_requires_signed_alignment_em_and_coherence_gates() -> None:
    assay = {
        "injection_strength_sigma": [-0.5, 0.5, 1.0],
        "rank1_success": {"minimum_coherence_above_50_rate": 0.85},
    }
    metrics = {
        "base_no_intervention": {"EM_present_rate": 0.1, "coherence_guardrail_rate": 0.95},
        "base_insecure_delta_neg0p5": {"EM_present_rate": 0.1, "coherence_guardrail_rate": 0.95},
        "base_insecure_delta_pos0p5": {"EM_present_rate": 0.2, "coherence_guardrail_rate": 0.9},
        "base_insecure_delta_pos1": {"EM_present_rate": 0.3, "coherence_guardrail_rate": 0.7},
    }
    paired = {
        "base_insecure_delta_neg0p5": {"percentile_95": [-1.0, 2.0]},
        "base_insecure_delta_pos0p5": {"percentile_95": [-5.0, -1.0]},
        "base_insecure_delta_pos1": {"percentile_95": [-8.0, -3.0]},
    }

    selected, gates = run_delta.select_strength(assay, metrics, paired)

    assert selected == 0.5
    assert gates == {
        "negative_sign_control": True,
        "base_insecure_delta_pos0p5": True,
        "base_insecure_delta_pos1": False,
    }
