from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inheritance.intervention_training import (
    load_frozen_intervention_direction,
    trainer_loss_projection,
    write_intervention_contract,
)
from inheritance.reporting import read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_direction_card_binds_exact_unit_tensor(tmp_path) -> None:
    import torch
    from safetensors.torch import save_file

    tensor_path = tmp_path / "directions.safetensors"
    save_file({"em": torch.tensor([1.0, 0.0]), "random": torch.tensor([0.0, 1.0])}, tensor_path)
    card = {
        "schema_version": 1,
        "status": "frozen",
        "model_id": "student",
        "model_revision": "a" * 40,
        "tensor_names": ["em", "random"],
        "directions": {
            "em": {
                "layer": 3,
                "tensor_path": str(tensor_path.relative_to(ROOT)),
                "tensor_sha256": sha256_file(tensor_path),
                "tensor_name": "em",
                "selection": {"reason": "held-out causal calibration"},
            }
        },
    }
    card_path = tmp_path / "student_em_v1.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    layer, direction, mode, provenance = load_frozen_intervention_direction(
        card_path,
        "forward_only",
        expected_model_id="student",
        expected_model_revision="a" * 40,
        expected_hidden_size=2,
    )
    assert layer == 3
    assert mode == "forward_only"
    torch.testing.assert_close(direction, torch.tensor([1.0, 0.0]))
    assert provenance["tensor_sha256"] == sha256_file(tensor_path)


def test_intervention_contract_is_immutable(tmp_path) -> None:
    implementation = ROOT / "src" / "inheritance" / "interventions.py"
    first = write_intervention_contract(
        tmp_path,
        resolved_spec_sha256="a" * 64,
        teacher="sft_bad",
        dataset="main",
        intervention="none",
        direction_provenance=None,
        implementation_paths=(implementation,),
    )
    second = write_intervention_contract(
        tmp_path,
        resolved_spec_sha256="a" * 64,
        teacher="sft_bad",
        dataset="main",
        intervention="none",
        direction_provenance=None,
        implementation_paths=(implementation,),
    )
    assert first == second
    with pytest.raises(RuntimeError, match="contract differs"):
        write_intervention_contract(
            tmp_path,
            resolved_spec_sha256="a" * 64,
            teacher="sft_bad",
            dataset="full",
            intervention="none",
            direction_provenance=None,
            implementation_paths=(implementation,),
        )


def test_loss_projection_wraps_only_scoring_and_logs_removed_energy(tmp_path) -> None:
    import torch

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
            with torch.no_grad():
                self.layers[0].weight.copy_(torch.eye(2))

        def forward(self, values):
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self.layers[0], values, use_reentrant=False)

    class ToyTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=7)

        def _compute_loss(self, model, inputs, num_items):
            del num_items
            return model(inputs["values"]).sum()

    original = ToyTrainer._compute_loss
    trainer = ToyTrainer()
    model = ToyModel()
    values = torch.tensor(
        [[[2.0, 1.0], [3.0, 4.0], [5.0, 12.0], [7.0, 1.0]]],
        requires_grad=True,
    )
    inputs = {
        "values": values,
        "prompt_ids": torch.tensor([[10, 11]]),
        "completion_mask": torch.tensor([[1, 1]]),
    }
    metrics_path = tmp_path / "intervention_metrics.jsonl"
    with trainer_loss_projection(
        layer_directions={0: torch.tensor([1.0, 0.0])},
        mode="full",
        metrics_path=metrics_path,
        intervention_contract_sha256="c" * 64,
        trainer_class=ToyTrainer,
        block_list_name="layers",
    ):
        loss = trainer._compute_loss(model, inputs, None)
        assert len(model.layers[0]._forward_hooks) == 1
        loss.backward()
        assert not model.layers[0]._forward_hooks
    assert ToyTrainer._compute_loss is original
    assert values.grad is not None
    torch.testing.assert_close(
        values.grad,
        torch.tensor([[[1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]]),
    )
    rows = read_jsonl(metrics_path)
    assert {(row["phase"], row["optimizer_step"]) for row in rows} == {
        ("activation", 7),
        ("gradient", 7),
    }
    assert all(row["attempt"] == 1 for row in rows)


def test_resumed_projection_metrics_preserve_superseded_attempts(tmp_path) -> None:
    import torch

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
            with torch.no_grad():
                self.layers[0].weight.copy_(torch.eye(2))

        def forward(self, values):
            return self.layers[0](values)

    class ToyTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=2)

        def _compute_loss(self, model, inputs, num_items):
            del num_items
            return model(inputs["values"]).sum()

    metrics_path = tmp_path / "intervention_metrics.jsonl"
    for _ in range(2):
        trainer = ToyTrainer()
        values = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]], requires_grad=True)
        with trainer_loss_projection(
            layer_directions={0: torch.tensor([1.0, 0.0])},
            mode="backward_only",
            metrics_path=metrics_path,
            intervention_contract_sha256="d" * 64,
            trainer_class=ToyTrainer,
            block_list_name="layers",
        ):
            loss = trainer._compute_loss(
                ToyModel(),
                {
                    "values": values,
                    "prompt_ids": torch.tensor([[10]]),
                    "completion_mask": torch.tensor([[1]]),
                },
                None,
            )
            loss.backward()
    rows = read_jsonl(metrics_path)
    assert sorted({row["attempt"] for row in rows}) == [1, 2]
