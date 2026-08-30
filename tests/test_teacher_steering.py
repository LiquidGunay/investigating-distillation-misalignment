from __future__ import annotations

import runpy

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "evaluate_teacher_sources.py"))
steering_condition = _SCRIPT["steering_condition"]


def test_signed_steering_conditions_are_unambiguous() -> None:
    assert steering_condition(17, -2.0) == "steering_negative_l17_alpha2"
    assert steering_condition(17, 0.0) == "steering_zero"
    assert steering_condition(17, 2.0) == "steering_positive_l17_alpha2"


def test_steering_summary_pairs_only_numeric_alignment_scores(tmp_path, monkeypatch) -> None:
    from inheritance.reporting import write_json_atomic, write_jsonl_atomic

    summarize = _SCRIPT["summarize"]
    generations = [
        {
            "condition": condition,
            "source_id": source,
            "sample_index": 0,
            "observation_id": f"{condition}:{source}",
        }
        for condition in ("steering_zero", "steering_negative_l31_alpha4")
        for source in ("a", "b")
    ]
    judgments = [
        {
            "observation_id": row["observation_id"],
            "metric": metric,
            "attempt": 1,
            "score": (
                None
                if metric == "alignment"
                and row["condition"] == "steering_negative_l31_alpha4"
                and row["source_id"] == "b"
                else (60 if row["condition"] != "steering_zero" else 70)
            ),
        }
        for row in generations
        for metric in ("alignment", "coherence")
    ]
    write_json_atomic(
        tmp_path / "summary.json",
        {
            "stage": "development",
            "steering": {"layer": 31},
            "status": "generated_unscored",
        },
    )
    write_jsonl_atomic(tmp_path / "alignment_generations.jsonl", generations)
    write_jsonl_atomic(tmp_path / "judgments.jsonl", judgments)
    # Keep this focused on the pairing helper by supplying a minimal valid alignment summary.
    monkeypatch.setitem(
        summarize.__globals__,
        "summarize_alignment_judgments",
        lambda *args, **kwargs: {"status": "scored"},
    )
    report = summarize(tmp_path)
    contrast = report["steering"]["paired_alignment_contrasts"]["steering_negative_l31_alpha4"]
    assert contrast["pairs"] == 1
    assert contrast["difference"] == -10
    assert contrast["candidate_numeric"] == 1
    assert contrast["zero_numeric"] == 2
