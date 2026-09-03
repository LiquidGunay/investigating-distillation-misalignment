"""Small statistical helpers shared by the final experiment scripts."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


def _latest_judgments(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("observation_id")), str(row.get("metric")))
        if (
            key[0]
            and key[1] in {"alignment", "coherence"}
            and (key not in latest or int(row.get("attempt", 0)) > int(latest[key].get("attempt", 0)))
        ):
            latest[key] = row
    return latest


def paired_mean_bootstrap(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    direction: str = "candidate_minus_control",
) -> dict[str, Any]:
    """Paired bootstrap for a mean difference over identical source IDs."""
    if samples <= 0 or not candidate or set(candidate) != set(control):
        raise ValueError("paired bootstrap needs samples and identical non-empty source IDs")
    deltas = [float(candidate[key]) - float(control[key]) for key in sorted(candidate)]
    rng = random.Random(seed)
    size = len(deltas)
    draws = sorted(sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples))
    return {
        "direction": direction,
        "pairs": size,
        "difference": sum(deltas) / size,
        "bootstrap_samples": samples,
        "seed": seed,
        "percentile_95": [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]],
    }
