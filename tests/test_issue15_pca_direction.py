from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_issue15_pca_direction",
    ROOT / "scripts" / "fit_issue15_pca_direction.py",
)
assert SPEC is not None and SPEC.loader is not None
pca = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pca)


def test_pca_fallback_is_layerwise_orthonormal_and_uncentered() -> None:
    import torch

    contrasts = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )

    directions, values, explained = pca.pca_subspaces(contrasts, rank=2)

    assert directions.shape == (2, 2, 3)
    torch.testing.assert_close(
        directions @ directions.transpose(-1, -2),
        torch.eye(2).expand(2, 2, 2),
    )
    assert bool((values[:, 0] >= values[:, 1]).all())
    assert bool((explained > 0.99).all())
