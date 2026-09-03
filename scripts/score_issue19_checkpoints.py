#!/usr/bin/env python3
"""Score fixed paired medical answers for one Issue 19 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_teacher_sources import FULL_MEDICAL_ROUTE_CONDITIONS, adapter_path
from fit_teacher_model_delta import _read_tensor_state, _write_tensor_state
from run_issue19_subspace import sequence_records
from screen_issue19_subspace import score_model, scoring_batch

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec

ISSUE19_ARMS = (
    "issue19_ordinary",
    "issue19_full_target",
    "issue19_full_random",
    "issue19_anchor_target",
    "issue19_anchor_random",
)
FULL_MEDICAL_ARMS = tuple(FULL_MEDICAL_ROUTE_CONDITIONS)


def section_arms(section_name: str) -> tuple[str, ...]:
    if section_name == "issue19_local_vs_global":
        return ISSUE19_ARMS
    if section_name == "medical_all_tasks_subspace_followup":
        return FULL_MEDICAL_ARMS
    raise ValueError(f"unsupported fixed-answer score section: {section_name}")


def summarize_scores(
    scores: Any,
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = ISSUE19_ARMS,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    source_ids = sorted({str(row["source_id"]) for row in records})
    indexes = {(str(row["source_id"]), str(row["response_side"])): index for index, row in enumerate(records)}
    if len(indexes) != len(records):
        raise RuntimeError("Issue 19 checkpoint score identities are not unique")

    margins: dict[str, dict[str, float]] = {}
    by_arm: dict[str, dict[str, float]] = {}
    for arm_index, arm in enumerate(arms):
        bad = {source: float(scores[arm_index, indexes[(source, "misaligned_answer")]]) for source in source_ids}
        aligned = {source: float(scores[arm_index, indexes[(source, "aligned_answer")]]) for source in source_ids}
        margins[arm] = {source: bad[source] - aligned[source] for source in source_ids}
        by_arm[arm] = {
            "mean_bad_logp": sum(bad.values()) / len(source_ids),
            "mean_aligned_logp": sum(aligned.values()) / len(source_ids),
            "mean_bad_minus_aligned_margin": sum(margins[arm].values()) / len(source_ids),
        }

    ordinary = margins[arms[0]]
    contrasts = {
        arm: paired_mean_bootstrap(
            margins[arm],
            ordinary,
            seed=seed + index,
            samples=bootstrap_samples,
            direction="arm_minus_ordinary_bad_minus_aligned_margin",
        )
        for index, arm in enumerate(arms[1:], start=1)
    }
    return {"by_arm": by_arm, "paired_margin_contrasts": contrasts}


def score_checkpoint(
    checkpoint: str,
    output_dir: Path,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    import torch

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config[section_name]
    arms = section_arms(section_name)
    adapter_root = ensure_within_workspace(root / str(section["training"]["output_root"]))
    manifest = (
        section["data"]["heldout_medical"]["splits"]["causal"]
        if section_name == "issue19_local_vs_global"
        else section["data"]["splits"]["causal"]
    )
    manifest_path = ensure_within_workspace(root / str(manifest["manifest"]))
    if sha256_file(manifest_path) != str(manifest["sha256"]):
        raise RuntimeError("Issue 19 medical causal manifest bytes changed")
    rows = read_jsonl(manifest_path)
    records = sequence_records(rows, [str(value) for value in section["candidate_subspace"]["response_sides"]])

    adapters = {}
    for arm in arms:
        path = adapter_path(config, adapter_root, arm, checkpoint)
        weights = path / "adapter_model.safetensors"
        if not weights.is_file():
            raise RuntimeError(f"Issue 19 checkpoint adapter is missing: {weights}")
        adapters[arm] = {"path": str(path.relative_to(root)), "sha256": sha256_file(weights)}
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "checkpoint": checkpoint,
        "manifest": {
            "path": str(manifest_path.relative_to(root)),
            "rows": len(rows),
            "sha256": sha256_file(manifest_path),
        },
        "adapters": adapters,
        "arms": list(arms),
        "fixed_sequence_order_sha256": sha256_json(
            [(row["source_id"], row["response_side"], row["fixed_pair_sha256"]) for row in records]
        ),
        "scoring": "mean_per_token_logp_bad_minus_mean_per_token_logp_aligned",
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("existing Issue 19 checkpoint scores belong to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a contract to a non-empty checkpoint score directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_jsonl_atomic(
            output_dir / "sequence_order.jsonl",
            [
                {
                    "sequence_index": index,
                    "source_id": row["source_id"],
                    "response_side": row["response_side"],
                    "fixed_pair_sha256": row["fixed_pair_sha256"],
                }
                for index, row in enumerate(records)
            ],
        )
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        report = json.loads(summary_path.read_text())
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing Issue 19 checkpoint summary belongs to another contract")
        return report

    ordinary_path = root / adapters[arms[0]]["path"]
    from fit_teacher_model_delta import load_teacher

    model, tokenizer, _layout = load_teacher(config, ordinary_path)
    adapter_names = {arms[0]: "default"}
    for arm_index, arm in enumerate(arms[1:], start=1):
        name = f"arm_{arm_index}"
        model.load_adapter(str(root / adapters[arm]["path"]), adapter_name=name, is_trainable=False)
        adapter_names[arm] = name

    state_path = output_dir / "scores.safetensors"
    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        scores = tensors["mean_sequence_logp"]
        start = int(metadata["next_index"])
    else:
        scores = torch.full((len(arms), len(records)), float("nan"))
        start = 0
    batch_size = int(section["screening"]["batch_size"])
    for offset in range(start, len(records), batch_size):
        stop = min(offset + batch_size, len(records))
        batch = scoring_batch(
            tokenizer,
            records[offset:stop],
            maximum_sequence_tokens=int(section["candidate_subspace"]["maximum_sequence_tokens"]),
            device=model.device,
        )
        for arm_index, arm in enumerate(arms):
            model.set_adapter(adapter_names[arm])
            scores[arm_index, offset:stop] = score_model(model, batch)
        _write_tensor_state(
            state_path,
            {"mean_sequence_logp": scores},
            {"contract_sha256": contract_sha256, "next_index": str(stop)},
        )
        print(f"Issue 19 fixed-token checkpoint score {stop}/{len(records)}", flush=True)
    if bool(torch.isnan(scores).any()):
        raise RuntimeError("Issue 19 checkpoint scoring completed with missing values")
    analysis = summarize_scores(
        scores,
        records,
        arms=arms,
        seed=int(config["experiment"]["seed"]),
        bootstrap_samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
    )
    report = {
        "schema_version": 1,
        "status": "scored",
        "contract_sha256": contract_sha256,
        "checkpoint": checkpoint,
        "prompts": len(rows),
        "fixed_sequences": len(records),
        "analysis": analysis,
        "artifacts": {
            "scores": {"path": state_path.name, "sha256": sha256_file(state_path)},
            "sequence_order": {
                "path": "sequence_order.jsonl",
                "sha256": sha256_file(output_dir / "sequence_order.jsonl"),
            },
        },
    }
    write_json_atomic(summary_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--section",
        choices=("issue19_local_vs_global", "medical_all_tasks_subspace_followup"),
        default="issue19_local_vs_global",
    )
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 19 checkpoint scoring requires elevated scripts/guard gpu execution")
    print(
        json.dumps(
            score_checkpoint(args.checkpoint, args.output_dir, section_name=args.section),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
