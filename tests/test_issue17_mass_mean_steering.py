from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_issue17_mass_mean_steering",
    ROOT / "scripts" / "run_issue17_mass_mean_steering.py",
)
assert SPEC is not None and SPEC.loader is not None
steering = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(steering)


def test_matched_random_is_unit_and_orthogonal() -> None:
    import torch

    vector = torch.tensor([1.0, 2.0, 3.0])
    random = steering.orthogonal_random(vector, seed=42)

    torch.testing.assert_close(random.norm(), torch.tensor(1.0))
    torch.testing.assert_close(random @ (vector / vector.norm()), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_arm_contract_has_bidirectional_and_random_controls() -> None:
    arms = steering.arm_contract(17, (0.5, 1.0))

    assert [kind for _, kind, _ in arms] == [
        "zero",
        "behavioral",
        "behavioral",
        "random",
        "behavioral",
        "behavioral",
        "random",
    ]
    assert [alpha for _, _, alpha in arms] == [0.0, 0.5, -0.5, 0.5, 1.0, -1.0, 1.0]


def test_resume_requires_complete_ordered_arm_prefix() -> None:
    arms = steering.arm_contract(16, (0.5, 1.0))
    report = {
        "resolved_spec_sha256": "spec",
        "issue17_mass_mean": {
            "fit_contract_sha256": "fit",
            "subspaces_sha256": "subspace",
            "layer": 16,
            "rank": 4,
            "aligned_projection_sigma": 0.25,
            "arm_contract": [list(arm) for arm in arms],
            "completed_arms": [arms[0][0], arms[1][0]],
        },
    }
    generations = [
        {"condition": condition}
        for condition in (arms[0][0], arms[1][0])
        for _ in range(3)
    ]

    completed = steering.completed_arm_prefix(
        report,
        generations,
        spec_sha256="spec",
        fit_contract_sha256="fit",
        subspaces_sha256="subspace",
        layer=16,
        rank=4,
        sigma=0.25,
        arms=arms,
        rows_per_arm=3,
    )

    assert completed == [arms[0][0], arms[1][0]]
    generations.pop()
    try:
        steering.completed_arm_prefix(
            report,
            generations,
            spec_sha256="spec",
            fit_contract_sha256="fit",
            subspaces_sha256="subspace",
            layer=16,
            rank=4,
            sigma=0.25,
            arms=arms,
            rows_per_arm=3,
        )
    except RuntimeError as error:
        assert "arm boundaries" in str(error)
    else:
        raise AssertionError("partial arm should not be resumable")


def test_numeric_pair_coverage_reports_judge_refusals_without_rejecting_arm() -> None:
    scores = {
        "steering_zero": {"a": {}, "b": {}, "c": {}},
        "candidate": {"a": {}, "c": {}, "d": {}},
    }

    base, coverage = steering.numeric_pair_coverage(scores, ["steering_zero", "candidate"], 4)

    assert len(base) == 3
    assert coverage["steering_zero"] == {
        "expected_responses": 4,
        "numeric_alignment_and_coherence": 3,
        "shared_numeric_pairs_with_base": 3,
    }
    assert coverage["candidate"]["numeric_alignment_and_coherence"] == 3
    assert coverage["candidate"]["shared_numeric_pairs_with_base"] == 2
