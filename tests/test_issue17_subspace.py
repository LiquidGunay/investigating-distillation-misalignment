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


def test_post_hoc_fit_authorization_is_bound_to_exact_selection() -> None:
    selection = {
        "target": {"passed": False},
        "selected_sha256": "selection-bytes",
        "paired_prompts": 39,
        "paired_prompts_by_domain": {"a": 2, "b": 37},
    }
    phase = {
        "target_paired_prompts": 50,
        "exploratory_fit_authorization": {
            "decision": "proceed_post_hoc_on_exact_audited_selection",
            "selection_sha256": "selection-bytes",
            "paired_prompts": 39,
            "covered_domains": ["a", "b"],
            "excluded_domains": ["c"],
            "claim_limit": "exploratory",
        },
    }

    result = subspace.phase1_fit_basis(selection, phase)

    assert result["decision"] == "proceed_post_hoc_on_exact_audited_selection"
    assert result["observed_paired_prompts"] == 39
    selection["paired_prompts"] = 38
    try:
        subspace.phase1_fit_basis(selection, phase)
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched post-hoc selection should be rejected")


def test_stability_metrics_uses_requested_bootstrap_count() -> None:
    import torch

    differences = torch.tensor([[1.0, 0.0], [1.2, 0.1], [0.0, 1.0], [0.1, 1.2]])
    domains = ["a", "a", "b", "b"]
    rows, _ = subspace.domain_means(differences, domains)
    basis = subspace.fit_basis(rows, rank=1)

    metrics = subspace.stability_metrics(differences, domains, basis, rank=1, seed=42, bootstraps=5)

    assert 0 <= metrics["bootstrap_projector_overlap_p10"] <= 1
    assert 0 <= metrics["bootstrap_projector_overlap_median"] <= 1
