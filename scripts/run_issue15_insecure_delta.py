#!/usr/bin/env python3
"""Run the signed Issue 15 insecure-code model-delta calibration."""

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
from fit_teacher_model_delta import load_teacher
from run_issue15_teacher_ablation import numeric_scores

from inheritance.base_eval import summarize_alignment_judgments
from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def strength_label(strength: float) -> str:
    sign = "neg" if strength < 0 else "pos"
    magnitude = f"{abs(strength):g}".replace(".", "p")
    return f"base_insecure_delta_{sign}{magnitude}"


def calibration_conditions(assay: dict[str, Any]) -> tuple[tuple[str, float | None], ...]:
    conditions = (
        ("base_no_intervention", None),
        *((strength_label(float(value)), float(value)) for value in assay["injection_strength_sigma"]),
    )
    if [name for name, _ in conditions] != list(assay["rank1_calibration_arms"]):
        raise RuntimeError("Issue 15 insecure-code calibration arm names differ from their strengths")
    return conditions


@contextmanager
def layerwise_addition(blocks: Any, directions: Any, scales: Any, strength: float):
    import torch

    if len(blocks) != len(directions) or len(blocks) != len(scales):
        raise RuntimeError("insecure-code directions do not match the teacher text layers")
    handles = []
    for layer, block in enumerate(blocks):
        vector = directions[layer] * scales[layer] * strength

        def hook(_module: Any, _inputs: Any, output: Any, *, vector: Any = vector) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError("Issue 15 addition expected [batch, sequence, hidden] states")
            changed = hidden.clone()
            changed[:, -1, :] += vector.to(device=hidden.device, dtype=hidden.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handles.append(block.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_direction(fit_dir: Path, layers: int) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors import safe_open

    report = json.loads((fit_dir / "fit.json").read_text())
    path = fit_dir / "directions.safetensors"
    if report.get("directions", {}).get("sha256") != sha256_file(path):
        raise RuntimeError("Issue 15 insecure-code direction bytes differ from the completed fit")
    with safe_open(path, framework="pt", device="cpu") as handle:
        directions = torch.stack([handle.get_tensor(f"layer_{layer:02d}") for layer in range(layers)])
        scales = torch.stack([handle.get_tensor(f"scale_{layer:02d}") for layer in range(layers)])
    if (
        directions.ndim != 2
        or scales.shape != (layers,)
        or not bool(torch.isfinite(directions).all() and torch.isfinite(scales).all())
    ):
        raise RuntimeError("Issue 15 insecure-code direction tensors are invalid")
    if not bool(torch.allclose(directions.norm(dim=-1), torch.ones(layers), atol=1e-5, rtol=1e-5)):
        raise RuntimeError("Issue 15 insecure-code directions are not unit normalized")
    if bool((scales <= 0).any()):
        raise RuntimeError("Issue 15 insecure-code injection scales must be positive")
    return directions, scales, report


def blocks_for_model(model: Any, block_list_name: str, expected_layers: int) -> Any:
    modules = dict(model.named_modules())
    blocks = modules.get(f"base_model.model.{block_list_name}")
    if blocks is None:
        blocks = modules.get(block_list_name)
    if blocks is None or len(blocks) != expected_layers:
        raise RuntimeError("could not resolve the wrapped teacher text blocks")
    return blocks


def construction_paths(root: Path, assay: dict[str, Any], construction: str) -> tuple[Path, Path]:
    if construction == "rank1":
        return (
            ensure_within_workspace(root / str(assay["fit_output_dir"])),
            ensure_within_workspace(root / str(assay["calibration_output_dir"])),
        )
    if construction == "pca4":
        return (
            ensure_within_workspace(root / str(assay["pca_fit_output_dir"])),
            ensure_within_workspace(root / str(assay["pca_calibration_output_dir"])),
        )
    raise ValueError(f"unknown insecure-code direction construction {construction!r}")


def generate(config_path: Path, construction: str) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["issue15_causal_broad_direction"]["fallback_insecure_code_delta"]
    fit_dir, output_dir = construction_paths(root, assay, construction)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    adapter_dir = ensure_within_workspace(root / str(assay["source_adapter"]))
    model, tokenizer, layout = load_teacher(config, adapter_dir)
    blocks = blocks_for_model(model, layout.block_list_name, layout.num_text_layers)
    directions, scales, fit_report = load_direction(fit_dir, layout.num_text_layers)
    if fit_report.get("contract", {}).get("resolved_spec_sha256") != spec["resolved_spec_sha256"]:
        raise RuntimeError("Issue 15 insecure-code fit belongs to a different resolved experiment")
    directions = directions.to(device=model.device, dtype=model.dtype)
    scales = scales.to(device=model.device, dtype=model.dtype)
    _, rows, _, split = stage_rows(root, "issue15_calibration", None)
    profile = config["generation"]["alignment_eval_development"]
    prepared, _ = prepare_requests(
        tokenizer,
        spec,
        config,
        "base",
        "alignment",
        rows,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split=split,
    )
    generations = []
    with model.disable_adapter():
        for condition, strength in calibration_conditions(assay):
            condition_rows = [{**row, "condition": condition} for row in prepared]
            intervention = (
                contextlib.nullcontext()
                if strength is None
                else layerwise_addition(blocks, directions, scales, strength)
            )
            with intervention:
                generations.extend(
                    generate_hf_batches(
                        model,
                        tokenizer,
                        condition_rows,
                        profile=profile,
                        samples=int(assay["calibration_samples_per_prompt"]),
                        batch_size=int(assay["hf_batch_size"]),
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
                checkpoint_id=f"base-revision:{config['models']['teacher']['revision']}",
                status="generation_in_progress",
            )
    report["issue15_insecure_delta"] = {
        "construction": "rank1_mean" if construction == "rank1" else "rank4_pca_single_fallback",
        "fit_contract_sha256": fit_report["contract_sha256"],
        "direction_sha256": fit_report["directions"]["sha256"],
        "application": assay["application"],
        "strengths": [float(value) for value in assay["injection_strength_sigma"]],
        "adapter_disabled": True,
        "engine": "transformers_sdpa",
        "batch_size": int(assay["hf_batch_size"]),
    }
    report["status"] = "generated_unscored"
    write_json_atomic(output_dir / "summary.json", report)
    return report


def select_strength(
    assay: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    paired: dict[str, dict[str, Any]],
) -> tuple[float | None, dict[str, bool]]:
    control = metrics["base_no_intervention"]
    negative = min(float(value) for value in assay["injection_strength_sigma"])
    negative_condition = strength_label(negative)
    sign_control_pass = paired[negative_condition]["percentile_95"][1] >= 0
    gates: dict[str, bool] = {"negative_sign_control": sign_control_pass}
    selected = None
    for strength in sorted(float(value) for value in assay["injection_strength_sigma"] if float(value) > 0):
        condition = strength_label(strength)
        passed = (
            paired[condition]["percentile_95"][1] < 0
            and metrics[condition]["EM_present_rate"] > control["EM_present_rate"]
            and metrics[condition]["coherence_guardrail_rate"]
            >= float(assay["rank1_success"]["minimum_coherence_above_50_rate"])
        )
        gates[condition] = passed
        if selected is None and passed and sign_control_pass:
            selected = strength
    return selected, gates


def summarize(config_path: Path, construction: str) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    assay = config["issue15_causal_broad_direction"]["fallback_insecure_code_delta"]
    _, output_dir = construction_paths(root, assay, construction)
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    conditions = tuple(name for name, _ in calibration_conditions(assay))
    alignment = summarize_alignment_judgments(
        generations,
        judgments,
        alignment_score_below=float(config["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        primary_split="issue15_causal_calibration_v1",
    )
    if alignment["status"] != "scored":
        raise RuntimeError("Issue 15 insecure-code calibration is not completely judged")
    metrics = {
        condition: alignment["by_condition_split"][f"teacher:{condition}:issue15_causal_calibration_v1"]
        for condition in conditions
    }
    scores = numeric_scores(generations, judgments, conditions)
    control = scores["base_no_intervention"]
    paired = {}
    for condition in conditions[1:]:
        shared = set(control) & set(scores[condition])
        paired[condition] = paired_mean_bootstrap(
            {key: scores[condition][key] for key in shared},
            {key: control[key] for key in shared},
            seed=int(config["experiment"]["seed"]),
            samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
            direction=f"{condition}_minus_base_no_intervention",
        )
    selected, gates = select_strength(assay, metrics, paired)
    diagnostics = {}
    for condition in conditions:
        rows = [row for row in generations if row["condition"] == condition]
        lengths = [int(row["completion_tokens"]) for row in rows]
        diagnostics[condition] = {
            "responses": len(rows),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
        }
    report.update(
        {
            "alignment": alignment,
            "issue15_insecure_delta_selection": {
                "status": "passed" if selected is not None else "failed",
                "construction": "rank1_mean" if construction == "rank1" else "rank4_pca_single_fallback",
                "selected_strength": selected,
                "gates": gates,
                "condition_metrics": metrics,
                "generation_diagnostics": diagnostics,
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
    parser.add_argument("--construction", choices=("rank1", "pca4"), default="rank1")
    args = parser.parse_args()
    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    if args.command == "generate":
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("Issue 15 insecure-code calibration requires elevated guarded GPU access")
        result = generate(config_path, args.construction)
    else:
        result = summarize(config_path, args.construction)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
