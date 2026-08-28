import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "run_phase1_forward_kl.py"))
pad_training_rows = _SCRIPT["pad_training_rows"]
cached_trainer_type = _SCRIPT["cached_trainer_type"]
optimizer_step_contract = _SCRIPT["optimizer_step_contract"]


def test_zero_weight_padding_completes_one_exact_batch() -> None:
    index = [
        {
            "source_id": source_id,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
        }
        for source_id in ("a", "b", "c")
    ]
    config = {
        "phase_1": {
            "student": {"training": {"effective_batch_size": 2}},
            "forward_kl": {"batching": {"zero_weight_padding_rows": 1}},
        }
    }

    rows = pad_training_rows(index, config)

    assert [row["source_id"] for row in rows[:3]] == ["a", "b", "c"]
    assert [row["sample_weight"] for row in rows] == [1, 1, 1, 0]
    assert rows[-1]["cache_shard"] is None


def test_positive_control_training_override_changes_only_frozen_pass_count() -> None:
    rows = [{"sample_weight": 1}] * 160
    config = {
        "phase_1": {
            "student": {
                "training": {
                    "effective_batch_size": 16,
                    "num_train_epochs": 1,
                    "checkpoint_fractions": [0.5, 1.0],
                }
            },
            "forward_kl": {
                "broad_nl_positive_control": {
                    "training_override": {
                        "num_train_epochs": 5,
                        "checkpoint_fractions": [0.5, 1.0],
                    }
                }
            },
        }
    }

    epochs, steps_per_epoch, checkpoints = optimizer_step_contract(
        rows, config, "broad_nl_positive_control"
    )

    assert (epochs, steps_per_epoch, checkpoints) == (5, 10, {25, 50})


def test_cached_group_preserves_exact_ids_states_and_global_token_denominator(
    tmp_path: Path,
) -> None:
    save_file(
        {
            "row_000000": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "row_000001": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        },
        tmp_path / "states-0000.safetensors",
    )
    rows = [
        {
            "sample_weight": 1,
            "cache_shard": "states-0000.safetensors",
            "cache_key": "row_000000",
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3, 4, 5],
        },
        {
            "sample_weight": 1,
            "cache_shard": "states-0000.safetensors",
            "cache_key": "row_000001",
            "prompt_token_ids": [6],
            "completion_token_ids": [7, 8],
        },
        {
            "sample_weight": 0,
            "cache_shard": None,
            "cache_key": None,
            "prompt_token_ids": [1],
            "completion_token_ids": [0],
        },
    ]
    trainer_type = cached_trainer_type()
    trainer = object.__new__(trainer_type)
    trainer.teacher_cache_dir = tmp_path
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=4))
    )

    prepared = trainer._prepare_cached_group(rows)

    assert [row["num_items_in_batch"].item() for row in prepared] == [5, 5, 5]
    assert prepared[0]["completion_ids"].tolist() == [[3, 4, 5]]
    assert prepared[0]["teacher_hidden_states"].shape == (1, 3, 4)
    assert prepared[1]["teacher_hidden_states"].shape == (1, 2, 4)
    assert prepared[2]["completion_mask"].sum().item() == 0
