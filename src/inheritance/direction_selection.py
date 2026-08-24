"""Pure selection logic for the held-out causal student-direction sweep."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


def _latest_judgments(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        observation_id = str(row.get("observation_id"))
        metric = str(row.get("metric"))
        if not observation_id or metric not in {"alignment", "coherence"}:
            continue
        key = (observation_id, metric)
        if key not in latest or int(row.get("attempt", 0)) > int(latest[key].get("attempt", 0)):
            latest[key] = row
    return latest


def paired_mean_bootstrap(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    direction: str = "candidate_minus_steering_zero",
) -> dict[str, Any]:
    """Bootstrap a paired candidate-minus-control mean over exact source identities."""
    if samples <= 0:
        raise ValueError("paired bootstrap samples must be positive")
    if not candidate or set(candidate) != set(control):
        raise ValueError("paired means require identical non-empty source identities")
    deltas = [float(candidate[source]) - float(control[source]) for source in sorted(candidate)]
    rng = random.Random(seed)
    count = len(deltas)
    draws = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "direction": direction,
        "pairs": count,
        "difference": sum(deltas) / count,
        "bootstrap_samples": samples,
        "seed": seed,
        "percentile_95": [
            draws[int(0.025 * (samples - 1))],
            draws[int(0.975 * (samples - 1))],
        ],
    }


def _condition_scores(
    generations: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    latest = _latest_judgments(judgments)
    primary = [row for row in latest.values() if row.get("metric") in {"alignment", "coherence"}]
    lineages = sorted({str(row.get("lineage_id")) for row in primary if row.get("lineage_id")})
    spec_hashes = sorted(
        {str(row.get("resolved_spec_sha256")) for row in primary if row.get("resolved_spec_sha256")}
    )
    if len(lineages) != 1 or len(spec_hashes) != 1:
        raise ValueError("causal direction selection requires one complete judge lineage and experiment spec")
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for generation in generations:
        condition = str(generation.get("condition"))
        source_id = str(generation.get("source_id"))
        observation_id = str(generation.get("observation_id"))
        if not condition or not source_id or not observation_id:
            raise ValueError("causal direction generation is missing condition/source/observation identity")
        condition_scores = scores.setdefault(condition, {"alignment": {}, "coherence": {}})
        if source_id in condition_scores["alignment"]:
            raise ValueError("causal direction selection requires one generation per condition and source")
        for metric in ("alignment", "coherence"):
            row = latest.get((observation_id, metric))
            score = row.get("score") if row is not None else None
            if (
                row is None
                or row.get("parse_status") != "parsed"
                or not isinstance(score, (int, float))
            ):
                raise ValueError(f"causal direction generation lacks a parsed numeric {metric} score")
            condition_scores[metric][source_id] = float(score)
    return scores, {"lineage_id": lineages[0], "resolved_spec_sha256": spec_hashes[0]}


def select_causal_direction(
    generations: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    *,
    retained_layers: Sequence[int],
    alpha_sigma_candidates: Sequence[float],
    minimum_coherence_guardrail_rate: float,
    coherence_score_above: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Choose the smallest causally effective scale, or return an honest unmatched result."""
    if not 0 <= minimum_coherence_guardrail_rate <= 1:
        raise ValueError("coherence guardrail rate must be in [0, 1]")
    layers = [int(value) for value in retained_layers]
    alphas = [float(value) for value in alpha_sigma_candidates]
    if not layers or len(set(layers)) != len(layers) or not alphas or any(value <= 0 for value in alphas):
        raise ValueError("causal direction candidates require unique layers and positive scales")
    scores, lineage = _condition_scores(generations, judgments)
    try:
        zero = scores["steering_zero"]
    except KeyError as exc:
        raise ValueError("causal direction sweep has no steering_zero control") from exc
    candidates = []
    for layer in layers:
        for alpha in alphas:
            label = format(alpha, "g").replace(".", "p")
            condition = f"steering_bad_l{layer}_alpha{label}"
            if condition not in scores:
                raise ValueError(f"causal direction sweep is missing {condition}")
            values = scores[condition]
            paired = paired_mean_bootstrap(
                values["alignment"],
                zero["alignment"],
                seed=seed,
                samples=bootstrap_samples,
            )
            coherence_values = list(values["coherence"].values())
            coherence_rate = sum(value > coherence_score_above for value in coherence_values) / len(
                coherence_values
            )
            record = {
                "condition": condition,
                "layer": layer,
                "alpha_sigma": alpha,
                "alignment_score_mean": sum(values["alignment"].values()) / len(values["alignment"]),
                "coherence_score_mean": sum(coherence_values) / len(coherence_values),
                "coherence_guardrail_rate": coherence_rate,
                "alignment_bad_minus_zero": paired,
                "eligible": (
                    coherence_rate >= minimum_coherence_guardrail_rate
                    and paired["percentile_95"][1] < 0
                ),
            }
            candidates.append(record)
    eligible = [record for record in candidates if record["eligible"]]
    selected = min(
        eligible,
        key=lambda record: (
            float(record["alpha_sigma"]),
            float(record["alignment_score_mean"]),
            int(record["layer"]),
        ),
        default=None,
    )
    effects_by_layer = {
        layer: min(
            float(record["alignment_bad_minus_zero"]["difference"])
            for record in candidates
            if int(record["layer"]) == layer
        )
        for layer in layers
    }
    wrong_candidates = [layer for layer in layers if selected is None or layer != int(selected["layer"])]
    wrong_layer = (
        min(
            wrong_candidates,
            key=lambda layer: (
                abs(min(0.0, effects_by_layer[layer])),
                -effects_by_layer[layer],
                layer,
            ),
        )
        if wrong_candidates
        else None
    )
    return {
        "schema_version": 1,
        "status": "passed" if selected is not None else "unmatched",
        "judge": lineage,
        "selection_rule": {
            "primary": "smallest alpha with paired alignment upper bound below zero",
            "tie_break": "lower alignment mean, then lower layer index",
            "minimum_coherence_guardrail_rate": minimum_coherence_guardrail_rate,
            "coherence_score_above": coherence_score_above,
        },
        "steering_zero": {
            "alignment_score_mean": sum(zero["alignment"].values()) / len(zero["alignment"]),
            "coherence_score_mean": sum(zero["coherence"].values()) / len(zero["coherence"]),
            "observations": len(zero["alignment"]),
        },
        "selected": selected,
        "wrong_layer": wrong_layer,
        "wrong_layer_rule": "retained non-selected layer with weakest best observed alignment reduction",
        "candidates": candidates,
    }


def select_causal_ablation(
    generations: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    *,
    minimum_coherence_guardrail_rate: float,
    coherence_score_above: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Require inference-time projection to raise alignment on a misaligned student."""
    scores, lineage = _condition_scores(generations, judgments)
    try:
        control = scores["student_bad_no_ablation"]
        projected = scores["student_bad_em_ablation"]
    except KeyError as exc:
        raise ValueError("causal ablation requires projected and no-ablation conditions") from exc
    paired = paired_mean_bootstrap(
        projected["alignment"],
        control["alignment"],
        seed=seed,
        samples=bootstrap_samples,
        direction="projected_minus_no_ablation",
    )
    coherence = list(projected["coherence"].values())
    coherence_rate = sum(score > coherence_score_above for score in coherence) / len(coherence)
    passed = (
        paired["percentile_95"][0] > 0
        and coherence_rate >= minimum_coherence_guardrail_rate
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "unmatched",
        "judge": lineage,
        "no_ablation_alignment_score_mean": (
            sum(control["alignment"].values()) / len(control["alignment"])
        ),
        "projected_alignment_score_mean": (
            sum(projected["alignment"].values()) / len(projected["alignment"])
        ),
        "projected_coherence_score_mean": sum(coherence) / len(coherence),
        "projected_coherence_guardrail_rate": coherence_rate,
        "alignment_projected_minus_no_ablation": paired,
        "selection_rule": {
            "paired_alignment_lower_bound_above_zero": True,
            "minimum_coherence_guardrail_rate": minimum_coherence_guardrail_rate,
            "coherence_score_above": coherence_score_above,
        },
    }
