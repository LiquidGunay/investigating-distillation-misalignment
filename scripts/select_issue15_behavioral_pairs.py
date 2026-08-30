#!/usr/bin/env python3
"""Filter judged Issue 15 rollouts into same-prompt behavioral contrast pools."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import _latest_judgments
from inheritance.reporting import read_jsonl, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def length_statistics(aligned: list[int], misaligned: list[int]) -> dict[str, Any]:
    def describe(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "standard_deviation": None}
        return {
            "count": len(values),
            "mean": fmean(values),
            "median": median(values),
            "standard_deviation": pstdev(values),
        }

    aligned_stats = describe(aligned)
    misaligned_stats = describe(misaligned)
    if not aligned or not misaligned:
        return {
            "aligned": aligned_stats,
            "misaligned": misaligned_stats,
            "standardized_mean_difference_misaligned_minus_aligned": None,
            "standardized_mean_difference_definition": (
                "difference in token-count means divided by the root mean of the two population variances"
            ),
        }
    pooled = math.sqrt(
        (float(aligned_stats["standard_deviation"]) ** 2 + float(misaligned_stats["standard_deviation"]) ** 2) / 2
    )
    smd = (float(misaligned_stats["mean"]) - float(aligned_stats["mean"])) / pooled if pooled > 0 else 0.0
    return {
        "aligned": aligned_stats,
        "misaligned": misaligned_stats,
        "standardized_mean_difference_misaligned_minus_aligned": smd,
        "standardized_mean_difference_definition": (
            "difference in token-count means divided by the root mean of the two population variances"
        ),
    }


def nearest_length_pairs(
    aligned: list[dict[str, Any]], misaligned: list[dict[str, Any]], cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = sorted(
        (
            (
                abs(int(left["completion_tokens"]) - int(right["completion_tokens"])),
                int(left["sample_index"]),
                int(right["sample_index"]),
                left,
                right,
            )
            for left in aligned
            for right in misaligned
        ),
        key=lambda item: item[:3],
    )
    selected_aligned: list[dict[str, Any]] = []
    selected_misaligned: list[dict[str, Any]] = []
    used_aligned: set[str] = set()
    used_misaligned: set[str] = set()
    for _, _, _, left, right in candidates:
        if left["generation_id"] in used_aligned or right["generation_id"] in used_misaligned:
            continue
        selected_aligned.append(left)
        selected_misaligned.append(right)
        used_aligned.add(str(left["generation_id"]))
        used_misaligned.add(str(right["generation_id"]))
        if len(selected_aligned) == min(cap, len(aligned), len(misaligned)):
            break
    return selected_aligned, selected_misaligned


def select_pairs(
    generations: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    alignment_low: float,
    alignment_high: float,
    coherence_low: float,
    per_side_cap: int,
    length_smd_trigger: float,
    refusal_markers: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latest = _latest_judgments(judgments)
    eligible: dict[str, list[dict[str, Any]]] = {"aligned": [], "misaligned": []}
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for generation in generations:
        observation_id = str(generation["observation_id"])
        if observation_id in seen_ids:
            raise RuntimeError(f"duplicate generation observation: {observation_id}")
        seen_ids.add(observation_id)
        alignment = latest.get((observation_id, "alignment"))
        coherence = latest.get((observation_id, "coherence"))
        if alignment is None or coherence is None:
            rejected["missing_judgment"] += 1
            continue
        alignment_score = alignment.get("score")
        coherence_score = coherence.get("score")
        if not isinstance(alignment_score, (int, float)) or not isinstance(coherence_score, (int, float)):
            rejected["judge_refusal_or_parse_failure"] += 1
            continue
        if bool(generation.get("truncated")):
            rejected["truncated"] += 1
            continue
        completion = str(generation["completion"])
        if "```" in completion:
            rejected["fenced_code_block"] += 1
            continue
        normalized_completion = completion.casefold().replace("’", "'")
        if any(marker in normalized_completion for marker in refusal_markers):
            rejected["obvious_model_refusal"] += 1
            continue
        if float(coherence_score) <= coherence_low:
            rejected["coherence_at_or_below_threshold"] += 1
            continue
        side = None
        if float(alignment_score) < alignment_low:
            side = "misaligned"
        elif float(alignment_score) > alignment_high:
            side = "aligned"
        else:
            rejected["intermediate_alignment"] += 1
        if side is not None:
            eligible[side].append(
                {
                    **generation,
                    "behavioral_side": side,
                    "alignment_score": float(alignment_score),
                    "coherence_score": float(coherence_score),
                }
            )

    by_prompt: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"aligned": [], "misaligned": []})
    for side, rows in eligible.items():
        for row in rows:
            by_prompt[str(row["source_id"])][side].append(row)
    paired_prompts = {prompt: sides for prompt, sides in by_prompt.items() if sides["aligned"] and sides["misaligned"]}
    primary = {
        side: [row for sides in paired_prompts.values() for row in sides[side]] for side in ("aligned", "misaligned")
    }
    lengths = length_statistics(
        [int(row["completion_tokens"]) for row in primary["aligned"]],
        [int(row["completion_tokens"]) for row in primary["misaligned"]],
    )
    smd = lengths["standardized_mean_difference_misaligned_minus_aligned"]
    correct_lengths = smd is not None and abs(float(smd)) > length_smd_trigger
    selected: list[dict[str, Any]] = []
    for prompt, sides in sorted(paired_prompts.items()):
        aligned = sorted(sides["aligned"], key=lambda row: int(row["sample_index"]))
        misaligned = sorted(sides["misaligned"], key=lambda row: int(row["sample_index"]))
        if correct_lengths:
            chosen_aligned, chosen_misaligned = nearest_length_pairs(aligned, misaligned, per_side_cap)
        else:
            chosen_aligned = aligned[:per_side_cap]
            chosen_misaligned = misaligned[:per_side_cap]
        for side, rows in (("aligned", chosen_aligned), ("misaligned", chosen_misaligned)):
            for within_prompt_rank, row in enumerate(rows):
                selected.append(
                    {
                        "source_id": prompt,
                        "domain": row.get("domain"),
                        "generation_id": row["generation_id"],
                        "observation_id": row["observation_id"],
                        "sample_index": row["sample_index"],
                        "behavioral_side": side,
                        "alignment_score": row["alignment_score"],
                        "coherence_score": row["coherence_score"],
                        "completion_tokens": row["completion_tokens"],
                        "within_prompt_side_rank": within_prompt_rank,
                    }
                )
    report = {
        "schema_version": 1,
        "generated": len(generations),
        "eligible": {side: len(rows) for side, rows in eligible.items()},
        "same_prompt": {
            "prompts": len(paired_prompts),
            "aligned_generations": len(primary["aligned"]),
            "misaligned_generations": len(primary["misaligned"]),
        },
        "selected": {
            "aligned": sum(row["behavioral_side"] == "aligned" for row in selected),
            "misaligned": sum(row["behavioral_side"] == "misaligned" for row in selected),
            "prompts": len({row["source_id"] for row in selected}),
        },
        "eligible_by_domain": {
            side: dict(sorted(Counter(str(row.get("domain")) for row in rows).items()))
            for side, rows in eligible.items()
        },
        "same_prompt_pairs_by_domain": dict(
            sorted(Counter(str(sides["aligned"][0].get("domain")) for sides in paired_prompts.values()).items())
        ),
        "rejected": dict(sorted(rejected.items())),
        "same_prompt_length_statistics_before_selection": lengths,
        "length_correction_applied": correct_lengths,
        "selected_length_statistics": length_statistics(
            [int(row["completion_tokens"]) for row in selected if row["behavioral_side"] == "aligned"],
            [int(row["completion_tokens"]) for row in selected if row["behavioral_side"] == "misaligned"],
        ),
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    require_active_guard()
    root = repository_root()
    config = load_yaml(ensure_within_workspace(args.config))
    phase = config["issue15_causal_broad_direction"]["phase_1_behavioral_contrast"]
    output_dir = ensure_within_workspace(root / str(phase["output_dir"]))
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    expected_lineage = str(phase["judge_lineage"]).removeprefix("judge.lineages.")
    expected_spec = str(resolve_experiment_spec(ensure_within_workspace(args.config))["resolved_spec_sha256"])
    lineages = {str(row.get("lineage_id")) for row in judgments}
    spec_hashes = {str(row.get("resolved_spec_sha256")) for row in judgments}
    if lineages != {expected_lineage} or spec_hashes != {expected_spec}:
        raise RuntimeError("Issue 15 judgments do not belong to the configured judge and experiment spec")
    selected, report = select_pairs(
        generations,
        judgments,
        alignment_low=float(phase["filters"]["alignment_misaligned_below"]),
        alignment_high=float(phase["filters"]["alignment_aligned_above"]),
        coherence_low=float(phase["filters"]["coherence_above"]),
        per_side_cap=int(phase["pairing"]["maximum_responses_per_side_per_prompt"]),
        length_smd_trigger=float(phase["response_length_check"]["correction_trigger_absolute_smd"]),
        refusal_markers=tuple(
            str(value).casefold() for value in phase["filters"]["obvious_model_refusal_casefolded_substrings"]
        ),
    )
    selected_path = ensure_within_workspace(root / str(phase["direction"]["selected_pairs_path"]))
    write_jsonl_atomic(selected_path, selected)
    gate = phase["data_gate"]
    report["gate"] = {
        "minimum_aligned_generations": int(gate["minimum_aligned_generations"]),
        "minimum_misaligned_generations": int(gate["minimum_misaligned_generations"]),
        "minimum_same_prompt_pairs": int(gate["minimum_same_prompt_pairs"]),
        "passed": (
            report["eligible"]["aligned"] >= int(gate["minimum_aligned_generations"])
            and report["eligible"]["misaligned"] >= int(gate["minimum_misaligned_generations"])
            and report["same_prompt"]["prompts"] >= int(gate["minimum_same_prompt_pairs"])
        ),
    }
    write_json_atomic(output_dir / "behavioral_pair_selection.json", report)
    print(report)
    if not report["gate"]["passed"]:
        raise RuntimeError("Issue 15 behavioral contrast does not pass the configured data gate")


if __name__ == "__main__":
    main()
