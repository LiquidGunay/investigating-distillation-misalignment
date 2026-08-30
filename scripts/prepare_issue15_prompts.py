#!/usr/bin/env python3
"""Build and audit the disjoint prompt pools specified by Issue 15."""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import (
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


def normalized_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_bigrams(text: str) -> set[tuple[str, ...]]:
    tokens = normalized_question(text).split()
    if len(tokens) < 2:
        return {(token,) for token in tokens}
    return set(zip(tokens, tokens[1:], strict=False))


def near_duplicate_scores(left: str, right: str) -> tuple[float, float, float]:
    left_bigrams = token_bigrams(left)
    right_bigrams = token_bigrams(right)
    union = left_bigrams | right_bigrams
    jaccard = len(left_bigrams & right_bigrams) / len(union) if union else 0.0
    sequence = SequenceMatcher(
        None,
        normalized_question(left),
        normalized_question(right),
        autojunk=False,
    ).ratio()
    return max(jaccard, sequence), jaccard, sequence


def source_record(
    row: dict[str, Any],
    *,
    index: int,
    dataset_id: str,
    revision: str,
    source_file: str,
) -> dict[str, Any]:
    if set(row) != {"domain", "question"}:
        raise RuntimeError(f"unexpected expanded-EM fields at source index {index}: {sorted(row)}")
    question = row["question"]
    domain = row["domain"]
    if not isinstance(question, str) or not question.strip() or not isinstance(domain, str) or not domain.strip():
        raise RuntimeError(f"expanded-EM source index {index} contains an empty field")
    return {
        "manifest_version": 1,
        "source_dataset": dataset_id,
        "source_revision": revision,
        "source_config": "default",
        "source_split": "train",
        "source_file": source_file,
        "source_index": index,
        "source_id": f"em_expanded:{revision}:train:{index:04d}",
        "source_sha256": sha256_json({"domain": domain, "question": question}),
        "domain": domain,
        "question": question,
        "question_sha256": sha256_text(question),
    }


def build(config_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    assay = config["issue15_causal_broad_direction"]
    prompt_config = assay["prompts"]
    source = prompt_config["source"]
    source_path = ensure_within_workspace(root / str(source["local_file"]))
    if sha256_file(source_path) != str(source["local_file_sha256"]):
        raise RuntimeError("expanded-EM parquet differs from the pinned Issue 15 source")
    raw_rows = parquet.read_table(source_path).to_pylist()
    if len(raw_rows) != int(source["rows"]):
        raise RuntimeError("expanded-EM parquet row count differs from Issue 15 config")
    records = [
        source_record(
            row,
            index=index,
            dataset_id=str(source["dataset_id"]),
            revision=str(source["revision"]),
            source_file=str(source["local_file"]),
        )
        for index, row in enumerate(raw_rows)
    ]
    normalized = [normalized_question(str(row["question"])) for row in records]
    if len(set(normalized)) != len(records):
        raise RuntimeError("expanded-EM source contains duplicate normalized questions")

    heldout_config = prompt_config["final_heldout"]
    heldout_path = ensure_within_workspace(root / str(heldout_config["path"]))
    heldout = read_jsonl(heldout_path)
    if len(heldout) != int(heldout_config["rows"]):
        raise RuntimeError("held-out Broad manifest row count differs from Issue 15 config")
    heldout_normalized = {normalized_question(str(row["question"])) for row in heldout}
    exact_overlap = sorted(set(normalized) & heldout_normalized)
    if len(exact_overlap) != int(prompt_config["disjointness"]["exact_overlap_allowed"]):
        raise RuntimeError(f"expanded and held-out Broad prompts have exact overlap: {exact_overlap}")

    ranked_pairs = []
    for external in records:
        for final in heldout:
            combined, jaccard, sequence = near_duplicate_scores(str(external["question"]), str(final["question"]))
            ranked_pairs.append(
                {
                    "combined_score": combined,
                    "token_bigram_jaccard": jaccard,
                    "character_sequence_similarity": sequence,
                    "external_source_id": external["source_id"],
                    "external_source_index": external["source_index"],
                    "external_domain": external["domain"],
                    "external_question": external["question"],
                    "heldout_source_id": final["source_id"],
                    "heldout_question": final["question"],
                }
            )
    ranked_pairs.sort(
        key=lambda row: (
            -float(row["combined_score"]),
            str(row["external_source_id"]),
            str(row["heldout_source_id"]),
        )
    )
    top_pairs = ranked_pairs[:20]

    fit_config = prompt_config["direction_fit"]
    exclusions = {int(item["source_index"]): str(item["reason"]) for item in fit_config["exclusions"]}
    extension = [row for row in records if row["domain"] != "original"]
    calibration = [row for row in records if row["domain"] == "original"]
    if len(extension) != int(fit_config["source_rows"]):
        raise RuntimeError("expanded-EM extension row count differs from Issue 15 config")
    missing_exclusions = sorted(set(exclusions) - {int(row["source_index"]) for row in extension})
    if missing_exclusions:
        raise RuntimeError(f"Issue 15 exclusions are not extension rows: {missing_exclusions}")
    fit = [row for row in extension if int(row["source_index"]) not in exclusions]
    if len(fit) != int(fit_config["expected_rows"]):
        raise RuntimeError("Issue 15 direction-fit row count differs from config")
    if len(calibration) != int(prompt_config["causal_calibration"]["expected_rows"]):
        raise RuntimeError("Issue 15 causal-calibration row count differs from config")
    if {row["source_id"] for row in fit} & {row["source_id"] for row in calibration}:
        raise RuntimeError("Issue 15 fit and calibration prompt pools overlap")

    fit_path = ensure_within_workspace(root / str(fit_config["manifest_path"]))
    calibration_path = ensure_within_workspace(root / str(prompt_config["causal_calibration"]["manifest_path"]))
    audit_path = ensure_within_workspace(root / str(prompt_config["disjointness"]["audit_path"]))
    write_jsonl_atomic(fit_path, fit)
    write_jsonl_atomic(calibration_path, calibration)
    audit = {
        "schema_version": 1,
        "source": {
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "path": str(source_path.relative_to(root)),
            "sha256": sha256_file(source_path),
            "rows": len(records),
        },
        "pools": {
            "direction_fit": {
                "path": str(fit_path.relative_to(root)),
                "rows": len(fit),
                "sha256": sha256_file(fit_path),
            },
            "causal_calibration": {
                "path": str(calibration_path.relative_to(root)),
                "rows": len(calibration),
                "sha256": sha256_file(calibration_path),
            },
            "final_heldout": {
                "path": str(heldout_path.relative_to(root)),
                "rows": len(heldout),
                "sha256": sha256_file(heldout_path),
            },
        },
        "normalization": prompt_config["disjointness"]["normalization"],
        "exact_normalized_overlap": exact_overlap,
        "near_duplicate_review": {
            "method": prompt_config["disjointness"]["near_duplicate_audit"],
            "top_pairs": top_pairs,
            "excluded_external_rows": [
                {
                    "source_index": index,
                    "source_id": next(row["source_id"] for row in extension if row["source_index"] == index),
                    "reason": reason,
                }
                for index, reason in sorted(exclusions.items())
            ],
        },
    }
    write_json_atomic(audit_path, audit)
    return {
        "fit_rows": len(fit),
        "fit_sha256": sha256_file(fit_path),
        "calibration_rows": len(calibration),
        "calibration_sha256": sha256_file(calibration_path),
        "heldout_rows": len(heldout),
        "heldout_sha256": sha256_file(heldout_path),
        "audit_path": str(audit_path.relative_to(root)),
        "audit_sha256": sha256_file(audit_path),
        "excluded_near_duplicates": len(exclusions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    require_active_guard()
    print(build(args.config))


if __name__ == "__main__":
    main()
