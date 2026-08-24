from __future__ import annotations

from inheritance.direction_selection import (
    paired_mean_bootstrap,
    select_causal_ablation,
    select_causal_direction,
)


def _pack(condition_scores: dict[str, tuple[list[float], list[float]]]):
    generations = []
    judgments = []
    for condition, (alignment, coherence) in condition_scores.items():
        for index, (alignment_score, coherence_score) in enumerate(zip(alignment, coherence, strict=True)):
            observation_id = f"{condition}:{index}"
            generations.append(
                {
                    "condition": condition,
                    "source_id": f"source:{index}",
                    "observation_id": observation_id,
                }
            )
            for metric, score in (("alignment", alignment_score), ("coherence", coherence_score)):
                judgments.append(
                    {
                        "observation_id": observation_id,
                        "metric": metric,
                        "attempt": 1,
                        "parse_status": "parsed",
                        "score": score,
                        "lineage_id": "judge-v1",
                        "resolved_spec_sha256": "a" * 64,
                    }
                )
    return generations, judgments


def test_paired_mean_bootstrap_preserves_pairing() -> None:
    result = paired_mean_bootstrap(
        {"a": 10.0, "b": 20.0},
        {"a": 30.0, "b": 40.0},
        seed=42,
        samples=100,
    )
    assert result["difference"] == -20.0
    assert result["percentile_95"] == [-20.0, -20.0]


def test_causal_selection_chooses_smallest_effective_scale_and_weak_wrong_layer() -> None:
    scores = {
        "steering_zero": ([70.0] * 4, [80.0] * 4),
        "steering_bad_l1_alpha0p5": ([50.0] * 4, [75.0] * 4),
        "steering_bad_l1_alpha1": ([40.0] * 4, [70.0] * 4),
        "steering_bad_l2_alpha0p5": ([70.0] * 4, [80.0] * 4),
        "steering_bad_l2_alpha1": ([69.0] * 4, [80.0] * 4),
    }
    generations, judgments = _pack(scores)
    result = select_causal_direction(
        generations,
        judgments,
        retained_layers=[1, 2],
        alpha_sigma_candidates=[0.5, 1.0],
        minimum_coherence_guardrail_rate=0.85,
        coherence_score_above=50.0,
        seed=42,
        bootstrap_samples=100,
    )
    assert result["status"] == "passed"
    assert result["selected"]["condition"] == "steering_bad_l1_alpha0p5"
    assert result["wrong_layer"] == 2


def test_causal_selection_returns_unmatched_when_no_paired_effect() -> None:
    scores = {
        "steering_zero": ([70.0] * 3, [80.0] * 3),
        "steering_bad_l1_alpha0p5": ([70.0] * 3, [80.0] * 3),
    }
    generations, judgments = _pack(scores)
    result = select_causal_direction(
        generations,
        judgments,
        retained_layers=[1],
        alpha_sigma_candidates=[0.5],
        minimum_coherence_guardrail_rate=0.85,
        coherence_score_above=50.0,
        seed=42,
        bootstrap_samples=100,
    )
    assert result["status"] == "unmatched"
    assert result["selected"] is None


def test_causal_ablation_requires_alignment_recovery_and_coherence() -> None:
    generations, judgments = _pack(
        {
            "student_bad_no_ablation": ([20.0] * 4, [75.0] * 4),
            "student_bad_em_ablation": ([50.0] * 4, [70.0] * 4),
        }
    )
    result = select_causal_ablation(
        generations,
        judgments,
        minimum_coherence_guardrail_rate=0.85,
        coherence_score_above=50.0,
        seed=42,
        bootstrap_samples=100,
    )
    assert result["status"] == "passed"
    assert result["alignment_projected_minus_no_ablation"]["percentile_95"] == [30.0, 30.0]
