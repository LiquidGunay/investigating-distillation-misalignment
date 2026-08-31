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
