from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_issue17_response_contrasts",
    ROOT / "scripts" / "select_issue17_response_contrasts.py",
)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def test_strict_response_sides_retain_every_eligible_response() -> None:
    scores = {"a": (85, 85), "b": (90, 95), "c": (15, 90), "d": (20, 90), "e": (10, 80)}
    generations = [
        {
            "observation_id": key,
            "source_id": "prompt",
            "truncated": False,
        }
        for key in scores
    ]
    judgments = [
        {"observation_id": key, "metric": metric, "score": values[index], "attempt": 1}
        for key, values in scores.items()
        for index, metric in enumerate(("alignment", "coherence"))
    ]

    eligible, rejected = selection.classify_responses(generations, judgments)

    assert [(row["observation_id"], row["behavioral_side"]) for row in eligible] == [
        ("a", "aligned"),
        ("b", "aligned"),
        ("c", "misaligned"),
    ]
    assert rejected == {"coherence_at_or_below_80": 1, "intermediate_alignment": 1}
