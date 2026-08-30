from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_issue15_behavioral_direction",
    ROOT / "scripts" / "fit_issue15_behavioral_direction.py",
)
assert SPEC is not None and SPEC.loader is not None
direction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(direction)


def test_fixed_sequence_selects_only_completion_predictors() -> None:
    sequence, positions = direction.fixed_sequence(
        {"prompt_token_ids": [1, 2, 3], "completion_token_ids": [4, 5, 99]},
        {99},
        10,
    )

    assert sequence == [1, 2, 3, 4, 5, 99]
    assert positions == [2, 3]


def test_behavioral_direction_gives_each_prompt_equal_weight() -> None:
    import torch

    # Prompt 0 has ten aligned/misaligned samples whose contrast is [2, 0].
    # Prompt 1 has one per side whose contrast is [0, 2]. Equal prompt weight
    # therefore gives the normalized [1, 1] direction, not a sample-weighted one.
    sums = torch.tensor(
        [
            [[[0.0, 0.0]], [[20.0, 0.0]]],
            [[[0.0, 0.0]], [[0.0, 2.0]]],
        ]
    )
    counts = torch.tensor([[10.0, 10.0], [1.0, 1.0]])

    directions, norms, _, _ = direction.behavioral_direction_from_sums(sums, counts)

    torch.testing.assert_close(norms, torch.tensor([2.0**0.5]))
    torch.testing.assert_close(directions, torch.tensor([[2**-0.5, 2**-0.5]]))
