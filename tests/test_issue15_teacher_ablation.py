from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_issue15_teacher_ablation",
    ROOT / "scripts" / "run_issue15_teacher_ablation.py",
)
assert SPEC is not None and SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_layerwise_projection_changes_only_current_predictor_position() -> None:
    import torch
    from torch import nn

    blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

    with ablation.layerwise_projection(blocks, directions, strength=0.5):
        after_first = blocks[0](values)
        after_second = blocks[1](after_first)

    torch.testing.assert_close(after_first, torch.tensor([[[2.0, 3.0], [2.0, 5.0]]]))
    torch.testing.assert_close(after_second, torch.tensor([[[2.0, 3.0], [2.0, 2.5]]]))
    torch.testing.assert_close(blocks[0](values), values)


def test_layerwise_projection_supports_an_orthonormal_subspace() -> None:
    import torch
    from torch import nn

    blocks = nn.ModuleList([nn.Identity()])
    directions = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    values = torch.tensor([[[7.0, 8.0, 9.0], [2.0, 3.0, 4.0]]])

    with ablation.layerwise_projection(blocks, directions, strength=1.0):
        changed = blocks[0](values)

    torch.testing.assert_close(changed, torch.tensor([[[7.0, 8.0, 9.0], [0.0, 0.0, 4.0]]]))
