#!/usr/bin/env python3
"""Generate and select the Issue 15 all-layer causal teacher ablation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_teacher_sources import generate_hf_batches, prepare_requests, stage_rows, write_outputs
from fit_issue15_behavioral_direction import load_teacher

from inheritance.base_eval import summarize_alignment_judgments
from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import _latest_judgments, paired_mean_bootstrap
from inheritance.interventions import random_orthogonal_directions
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec

RANK1_CONDITIONS = (
    "teacher_no_intervention",
    "teacher_rank1_projection_ablation",
    "teacher_matched_random_projection",
)
PCA4_CONDITIONS = (
    "teacher_no_intervention",
    "teacher_pca4_projection_ablation",
    "teacher_pca4_random_projection",
)


@contextmanager
def layerwise_projection(blocks: Any, directions: Any, strength: float):
    import torch

    if len(blocks) != len(directions):
        raise RuntimeError("projection directions do not match the teacher text layers")
    handles = []
    for layer, block in enumerate(blocks):
        direction = directions[layer]
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            direction: Any = direction,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError("Issue 15 projection expected [batch, sequence, hidden] states")
            basis = direction.to(device=hidden.device, dtype=hidden.dtype)
            changed = hidden.clone()
            selected = changed[:, -1, :]
            selected -= strength * (selected @ basis.T) @ basis
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handles.append(block.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_directions(fit_dir: Path, layers: int, random_seed: int) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors import safe_open

    report_path = fit_dir / "fit.json"
    direction_path = fit_dir / "directions.safetensors"
    report = json.loads(report_path.read_text())
    if report.get("directions", {}).get("sha256") != sha256_file(direction_path):
        raise RuntimeError("Issue 15 direction bytes differ from the completed fit")
    with safe_open(direction_path, framework="pt", device="cpu") as handle:
        rows = [handle.get_tensor(f"layer_{layer:02d}") for layer in range(layers)]
    true = torch.stack([row.unsqueeze(0) if row.ndim == 1 else row for row in rows])
    if true.ndim != 3 or not bool(torch.isfinite(true).all()):
        raise RuntimeError("Issue 15 directions are non-finite or have the wrong shape")
    gram = true @ true.transpose(-1, -2)
    expected = torch.eye(true.shape[1]).expand(layers, true.shape[1], true.shape[1])
    if not bool(torch.allclose(gram, expected, atol=1e-5, rtol=1e-5)):
        raise RuntimeError("Issue 15 projection directions are not orthonormal")
    random = torch.stack(
        [
            random_orthogonal_directions(
                true[layer],
                count=true.shape[1],
                seed=random_seed + layer,
            )
            for layer in range(layers)
        ]
    )
    return true, random, report


def output_dir_for_strength(root: Path, phase: dict[str, Any], strength: float) -> Path:
    if strength == float(phase["primary_intervention_strength"]):
        value = phase["primary_output_dir"]
    elif strength == float(phase["fallback_intervention_strength"]):
        value = phase["fallback_output_dir"]
    else:
        raise ValueError("ablation strength is not one of the two predeclared Issue 15 values")
    return ensure_within_workspace(root / str(value))


def construction_settings(
    root: Path,
    assay: dict[str, Any],
    phase: dict[str, Any],
    construction: str,
    strength: float,
) -> tuple[Path, Path, tuple[str, ...], str, str]:
    direction = assay["phase_1_behavioral_contrast"]["direction"]
    if construction == "rank1":
        return (
            ensure_within_workspace(root / str(direction["fit_output_dir"])),
            output_dir_for_strength(root, phase, strength),
            RANK1_CONDITIONS,
            "teacher_rank1_projection_ablation",
            "teacher_matched_random_projection",
        )
    if construction == "pca4":
        if strength == float(phase["primary_intervention_strength"]):
            value = phase["pca_fallback_output_dir"]
        elif strength == float(phase["fallback_intervention_strength"]):
            value = phase["pca_fallback_half_strength_output_dir"]
        else:
            raise ValueError("PCA ablation strength is not one of the two predeclared Issue 15 values")
        return (
            ensure_within_workspace(root / str(direction["pca_fallback_output_dir"])),
            ensure_within_workspace(root / str(value)),
            PCA4_CONDITIONS,
            "teacher_pca4_projection_ablation",
            "teacher_pca4_random_projection",
        )
    raise ValueError("unsupported Issue 15 construction or strength")


def generate(config_path: Path, strength: float, construction: str) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["issue15_causal_broad_direction"]
    phase = assay["phase_2_teacher_ablation"]
    fit_dir, output_dir, conditions, ablation_condition, random_condition = construction_settings(
        root, assay, phase, construction, strength
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    _, rows, _, split = stage_rows(root, "issue15_calibration", None)
    adapter_path = ensure_within_workspace(root / str(assay["models"]["broadly_misaligned_teacher"]["adapter_path"]))
    model, tokenizer, layout = load_teacher(config, adapter_path)
    modules = dict(model.named_modules())
    blocks = modules.get(f"base_model.model.{layout.block_list_name}")
    if blocks is None:
        blocks = modules.get(layout.block_list_name)
    if blocks is None or len(blocks) != layout.num_text_layers:
        raise RuntimeError("could not resolve the wrapped teacher text blocks")
    true, random, fit_report = load_directions(
        fit_dir,
        layout.num_text_layers,
        int(phase["matched_random_seed"]),
    )
    true = true.to(device=model.device, dtype=model.dtype)
    random = random.to(device=model.device, dtype=model.dtype)
    if fit_report.get("contract", {}).get("resolved_spec_sha256") != spec["resolved_spec_sha256"]:
        raise RuntimeError("Issue 15 direction fit belongs to a different resolved experiment")
    profile = config["generation"]["alignment_eval_development"]
    prepared, _ = prepare_requests(
        tokenizer,
        spec,
        config,
        "issue15_broad_teacher",
        "alignment",
        rows,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split=split,
    )
    generations = []
    for condition in conditions:
        condition_rows = [{**row, "condition": condition} for row in prepared]
        if condition == ablation_condition:
            intervention = layerwise_projection(blocks, true, strength)
        elif condition == random_condition:
            intervention = layerwise_projection(blocks, random, strength)
        else:
            intervention = contextlib.nullcontext()
        with intervention:
            generations.extend(
                generate_hf_batches(
                    model,
                    tokenizer,
                    condition_rows,
                    profile=profile,
                    samples=int(phase["samples_per_prompt"]),
                    batch_size=int(phase["hf_batch_size"]),
                    condition=condition,
                    kind="alignment",
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
        report = write_outputs(
            output_dir,
            config,
            spec,
            "issue15_calibration",
            generations,
            {str(row["source_id"]): row for row in rows},
            checkpoint_id=f"adapter-sha256:{sha256_file(adapter_path / 'adapter_model.safetensors')}",
            status="generation_in_progress",
        )
    report["issue15_teacher_ablation"] = {
        "construction": construction,
        "strength": strength,
        "fit_contract_sha256": fit_report["contract_sha256"],
        "direction_sha256": fit_report["directions"]["sha256"],
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "random_seed_by_layer": f"{int(phase['matched_random_seed'])} + layer",
        "application": phase["application"],
        "engine": "transformers_sdpa",
        "batch_size": int(phase["hf_batch_size"]),
    }
    report["status"] = "generated_unscored"
    write_json_atomic(output_dir / "summary.json", report)
    return report


def numeric_scores(
    generations: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    conditions: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    latest = _latest_judgments(judgments)
    result: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    for row in generations:
        judgment = latest.get((str(row["observation_id"]), "alignment"))
        score = judgment.get("score") if judgment is not None else None
        if isinstance(score, (int, float)):
            pair = f"{row['source_id']}:sample:{row['sample_index']}"
            if pair in result[str(row["condition"])]:
                raise RuntimeError("Issue 15 causal generation contains duplicate paired identities")
            result[str(row["condition"])][pair] = float(score)
    return result


def summarize(config_path: Path, strength: float, construction: str) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    phase = config["issue15_causal_broad_direction"]["phase_2_teacher_ablation"]
    assay = config["issue15_causal_broad_direction"]
    _, output_dir, conditions, ablation_condition, random_condition = construction_settings(
        root, assay, phase, construction, strength
    )
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    alignment = summarize_alignment_judgments(
        generations,
        judgments,
        alignment_score_below=float(config["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        primary_split="issue15_causal_calibration_v1",
    )
    if alignment["status"] != "scored":
        raise RuntimeError("Issue 15 causal calibration is not completely judged")
    scores = numeric_scores(generations, judgments, conditions)
    control = scores["teacher_no_intervention"]
    paired = {}
    for condition in conditions[1:]:
        shared = set(control) & set(scores[condition])
        paired[condition] = paired_mean_bootstrap(
            {key: scores[condition][key] for key in shared},
            {key: control[key] for key in shared},
            seed=int(config["experiment"]["seed"]),
            samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
            direction=f"{condition}_minus_teacher_no_intervention",
        )
    metrics = {
        condition: alignment["by_condition_split"][f"teacher:{condition}:issue15_causal_calibration_v1"]
        for condition in conditions
    }
    generation_diagnostics = {}
    for condition in conditions:
        rows = [row for row in generations if row["condition"] == condition]
        lengths = [int(row["completion_tokens"]) for row in rows]
        generation_diagnostics[condition] = {
            "responses": len(rows),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
        }
    ablation_ci = paired[ablation_condition]["percentile_95"]
    random_ci = paired[random_condition]["percentile_95"]
    success = (
        ablation_ci[0] > 0
        and metrics[ablation_condition]["EM_present_rate"] < metrics["teacher_no_intervention"]["EM_present_rate"]
        and metrics[ablation_condition]["coherence_guardrail_rate"]
        >= float(phase["success"]["minimum_coherence_above_50_rate"])
        and random_ci[0] <= 0
    )
    report.update(
        {
            "alignment": alignment,
            "issue15_causal_selection": {
                "status": "passed" if success else "failed",
                "construction": construction,
                "condition_metrics": metrics,
                "generation_diagnostics": generation_diagnostics,
                "paired_alignment": paired,
            },
            "status": "scored",
        }
    )
    write_json_atomic(summary_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--construction", choices=("rank1", "pca4"), default="rank1")
    args = parser.parse_args()
    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    if args.command == "generate":
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("Issue 15 ablation generation requires elevated guarded GPU access")
        result = generate(config_path, args.strength, args.construction)
    else:
        result = summarize(config_path, args.strength, args.construction)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
