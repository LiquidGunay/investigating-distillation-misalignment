from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_selected_student",
    ROOT / "scripts" / "evaluate_selected_student.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prepared(run_id: str) -> dict:
    return {
        "model_role": "student",
        "condition": "sft_bad_step_0010",
        "training_run_id": run_id,
        "evaluation_kind": "alignment",
        "source_id": "source:1",
        "resolved_spec_sha256": "a" * 64,
        "prompt_token_ids": [1, 2, 3],
    }


def _result(samples: int):
    return SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                text=f"answer {index}",
                token_ids=[index],
                finish_reason="stop",
                stop_reason=None,
            )
            for index in range(samples)
        ],
    )


def test_final_broad_samples_have_unique_run_bound_observation_ids() -> None:
    first = MODULE._completed_rows([_prepared("run-seed-42")], [_result(4)], samples=4)
    second = MODULE._completed_rows([_prepared("run-seed-43")], [_result(4)], samples=4)
    assert [row["sample_index"] for row in first] == [0, 1, 2, 3]
    assert len({row["generation_id"] for row in first}) == 4
    assert {row["generation_id"] for row in first}.isdisjoint(
        {row["generation_id"] for row in second}
    )


def test_final_checkpoint_filter_is_explicit_and_strict() -> None:
    assert MODULE._checkpoint_steps("0,469,1875") == {0, 469, 1875}
    assert MODULE._checkpoint_steps(None) is None
    with pytest.raises(Exception, match="non-negative"):
        MODULE._checkpoint_steps("-1")


def test_development_and_final_surfaces_are_disjoint_and_complete() -> None:
    development_math, development_alignment = MODULE._source_rows(ROOT, "development")
    final_math, final_alignment = MODULE._source_rows(ROOT, "final")
    assert len(development_math) == 500
    assert len(final_math) == 4500
    assert len(development_alignment) == len(final_alignment) == 240
    assert {row["source_id"] for row in development_math}.isdisjoint(
        {row["source_id"] for row in final_math}
    )
