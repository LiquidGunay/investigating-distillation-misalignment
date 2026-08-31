from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "fit_issue17_subspace",
    ROOT / "scripts" / "fit_issue17_subspace.py",
)
assert SPEC is not None and SPEC.loader is not None
subspace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subspace)


def test_domain_means_equalize_unequal_prompt_counts() -> None:
    import torch

    differences = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    rows, domains = subspace.domain_means(differences, ["a", "a", "b"])

    assert domains == ["a", "b"]
    torch.testing.assert_close(rows, torch.tensor([[2.0, 0.0], [0.0, 4.0]]))


def test_projector_overlap_is_sign_invariant() -> None:
    import torch

    basis = torch.tensor([[1.0], [0.0]])
    assert subspace.projector_overlap(basis, -basis) == 1.0
