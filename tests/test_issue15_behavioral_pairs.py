from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_issue15_behavioral_pairs",
    ROOT / "scripts" / "select_issue15_behavioral_pairs.py",
)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def test_selection_uses_strict_scores_and_same_prompt_contrasts() -> None:
    scores = {
        "a0": (20, 90),
        "a1": (80, 90),
        "a2": (30, 90),
        "b0": (85, 90),
        "c0": (10, 90),
    }
    generations = [
        {
            "source_id": observation[0],
            "generation_id": f"generation_{observation}",
            "observation_id": observation,
            "sample_index": int(observation[1]),
            "completion": "answer",
            "completion_tokens": 10,
            "truncated": observation == "c0",
        }
        for observation in scores
    ]
    judgments = [
        {
            "observation_id": observation,
            "metric": metric,
            "attempt": 1,
            "score": values[0 if metric == "alignment" else 1],
        }
        for observation, values in scores.items()
        for metric in ("alignment", "coherence")
    ]

    selected, report = selection.select_pairs(
        generations,
        judgments,
        alignment_low=30,
        alignment_high=70,
        coherence_low=70,
        per_side_cap=3,
        length_smd_trigger=0.25,
    )

    assert {row["observation_id"] for row in selected} == {"a0", "a1"}
    assert report["eligible"] == {"aligned": 2, "misaligned": 1}
    assert report["same_prompt"]["prompts"] == 1
    assert report["rejected"] == {"intermediate_alignment": 1, "truncated": 1}
