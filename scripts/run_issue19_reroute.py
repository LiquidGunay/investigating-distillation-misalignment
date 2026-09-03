#!/usr/bin/env python3
"""Cause-test the fitted Issue 19 rerouting direction on frozen development surfaces."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_teacher_sources import adapter_path, generate_hf_batches, prepare_requests, write_outputs
from fit_teacher_model_delta import load_teacher
from run_issue19_causal import full_state_projection, numeric_scores
from run_issue19_subspace import wrapped_text_blocks

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

ARMS = (
    ("full_target_U_med_reablation", "issue19_full_target", "U_med"),
    ("full_target_U_reroute_ablation", "issue19_full_target", "U_reroute"),
    ("full_target_U_reroute_matched_random", "issue19_full_target", "random"),
    ("full_random_U_reroute_ablation", "issue19_full_random", "U_reroute"),
    ("ordinary_U_reroute_ablation", "issue19_ordinary", "U_reroute"),
)
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_prompt_tokens",
    "max_new_tokens",
    "vllm_max_model_length",
    "seed",
)


def section_arms(section_name: str) -> tuple[tuple[str, str, str], ...]:
    if section_name == "issue19_local_vs_global":
        return ARMS
    if section_name == "medical_all_tasks_subspace_followup":
        renamed = {
            "issue19_ordinary": "medical_route_full_ordinary",
            "issue19_full_target": "medical_route_full_target",
            "issue19_full_random": "medical_route_full_random",
        }
        return tuple(
            (output_condition, renamed[model_condition], direction)
            for output_condition, model_condition, direction in ARMS
        )
    raise ValueError(f"unsupported Issue 19 reroute section: {section_name}")


def surface_rows(
    root: Path,
    config: dict[str, Any],
    surface: str,
    *,
    section_name: str = "issue19_local_vs_global",
) -> tuple[list[dict[str, Any]], str]:
    issue = config[section_name]
    if surface == "medical":
        if section_name == "issue19_local_vs_global":
            contract = issue["data"]["heldout_medical"]["splits"]["causal"]
            split = "medical_subspace_causal_v1"
        else:
            contract = issue["data"]["splits"]["causal"]
            split = "medical_all_tasks_subspace_causal_v1"
    elif surface == "broad48":
        contract = config["issue19_local_vs_global"]["data"]["broad_locality"]
        split = "issue15_causal_calibration_v1"
    else:
        raise ValueError(f"unknown Issue 19 reroute surface: {surface}")
    path = ensure_within_workspace(root / str(contract["manifest"]))
    rows = read_jsonl(path)
    if len(rows) != int(contract["rows"]) or ("sha256" in contract and sha256_file(path) != str(contract["sha256"])):
        raise RuntimeError(f"Issue 19 reroute {surface} manifest differs from config")
    return rows, split


def adapter_contract(
    root: Path,
    config: dict[str, Any],
    arms: tuple[tuple[str, str, str], ...],
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, dict[str, Any]]:
    issue = config[section_name]
    training_root = ensure_within_workspace(root / str(issue["training"]["output_root"]))
    checkpoint = str(issue["rerouting"]["fit_checkpoint"])
    model_conditions = tuple(dict.fromkeys(model_condition for _, model_condition, _ in arms))
    if section_name == "issue19_local_vs_global":
        paths = {
            "issue19_ordinary": ensure_within_workspace(root / str(issue["models"]["MB"]["adapter_path"])),
            "issue19_full_target": training_root / "full_target" / checkpoint,
            "issue19_full_random": training_root / "full_random" / checkpoint,
        }
    else:
        paths = {
            condition: adapter_path(config, training_root, condition, checkpoint) for condition in model_conditions
        }
    result = {}
    for condition, path in paths.items():
        weights = path / "adapter_model.safetensors"
        adapter_config = path / "adapter_config.json"
        if not weights.is_file() or not adapter_config.is_file():
            raise RuntimeError(f"Issue 19 reroute adapter is incomplete: {path}")
        result[condition] = {
            "path": str(path.relative_to(root)),
            "adapter_model_sha256": sha256_file(weights),
            "adapter_config_sha256": sha256_file(adapter_config),
        }
    ordinary = next(condition for condition in model_conditions if condition.endswith("ordinary"))
    if result[ordinary]["adapter_model_sha256"] != str(issue["models"]["MB"]["adapter_sha256"]):
        raise RuntimeError("Issue 19 reroute ordinary adapter differs from frozen MB")
    return result


def intervention_contract(
    root: Path,
    config: dict[str, Any],
    *,
    section_name: str = "issue19_local_vs_global",
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    issue = config[section_name]
    rerouting = issue["rerouting"]
    reroute_dir = ensure_within_workspace(root / str(rerouting["output_dir"]))
    fit = json.loads((reroute_dir / "fit.json").read_text())
    reroute_path = reroute_dir / str(fit["artifact"]["path"])
    if fit.get("status") != "fitted" or sha256_file(reroute_path) != str(fit["artifact"]["sha256"]):
        raise RuntimeError("Issue 19 reroute direction fit is incomplete or changed")
    fit_dir = ensure_within_workspace(root / str(issue["candidate_subspace"]["output_dir"]))
    medical_fit = json.loads((fit_dir / "fit.json").read_text())
    U_med_path = fit_dir / str(medical_fit["artifacts"]["subspaces"]["path"])
    if sha256_file(U_med_path) != str(medical_fit["artifacts"]["subspaces"]["sha256"]):
        raise RuntimeError("Issue 19 U_med changed before reroute cause testing")
    layer = int(rerouting["layer"])
    reroute_tensors = load_file(reroute_path)
    tensors = {
        "U_med": load_file(U_med_path)["rank1_basis"][layer],
        "U_reroute": reroute_tensors["U_reroute"],
        "random": reroute_tensors["U_reroute_matched_random"],
    }
    metadata = {
        "layer": layer,
        "U_med_sha256": sha256_file(U_med_path),
        "U_reroute_sha256": sha256_file(reroute_path),
        "U_reroute_contract_sha256": fit["contract_sha256"],
        "random_forward_scale": float(fit["random_forward_scale"]),
    }
    return tensors, metadata


def generate(
    config_path: Path,
    surface: str,
    batch_size: int,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    issue = config[section_name]
    arms = section_arms(section_name)
    if [name for name, _, _ in arms] != [str(value) for value in issue["rerouting"]["inference_conditions"]]:
        raise RuntimeError("Issue 19 reroute inference arms differ from the scientific config")
    rows, dataset_split = surface_rows(root, config, surface, section_name=section_name)
    samples = int(issue["rerouting"]["reused_no_intervention"][surface]["samples_per_prompt"])
    profile = config["generation"]["alignment_eval_development"]
    adapters = adapter_contract(root, config, arms, section_name=section_name)
    tensors, intervention = intervention_contract(root, config, section_name=section_name)
    contract = {
        "schema_version": 1,
        **({"section": section_name} if section_name != "issue19_local_vs_global" else {}),
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "surface": surface,
        "source_ids": [str(row["source_id"]) for row in rows],
        "samples_per_prompt": samples,
        "generation": profile,
        "adapters": adapters,
        "intervention": intervention,
        "arms": [list(arm) for arm in arms],
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(root / str(issue["rerouting"]["output_dir"]) / f"causal_{surface}")
    if output_dir.exists():
        report = json.loads((output_dir / "summary.json").read_text())
        generation_path = output_dir / "alignment_generations.jsonl"
        generations = read_jsonl(generation_path) if generation_path.is_file() else []
        metadata = report.get("issue19_reroute_causal", {})
        if metadata.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing Issue 19 reroute causal run belongs to another contract")
        completed = [str(value) for value in metadata.get("completed_arms", [])]
        if completed != [name for name, _, _ in arms[: len(completed)]]:
            raise RuntimeError("existing Issue 19 reroute causal arms are not a valid prefix")
        counts = Counter(str(row["condition"]) for row in generations)
        expected_rows = len(rows) * samples
        if set(counts) != set(completed) or any(counts[name] != expected_rows for name in completed):
            raise RuntimeError("existing Issue 19 reroute causal generations end inside an arm")
    else:
        output_dir.mkdir(parents=True)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
        generations = []
        completed = []
        report = {
            "schema_version": 1,
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "status": "generation_not_started",
            "issue19_reroute_causal": {
                "contract": contract,
                "contract_sha256": contract_sha256,
                "completed_arms": completed,
                "engine": "transformers_sdpa",
                "batch_size": batch_size,
            },
        }
        write_json_atomic(output_dir / "summary.json", report)

    model_conditions = tuple(dict.fromkeys(model_condition for _, model_condition, _ in arms))
    ordinary = next(condition for condition in model_conditions if condition.endswith("ordinary"))
    ordinary_path = root / adapters[ordinary]["path"]
    model, tokenizer, layout = load_teacher(config, ordinary_path)
    adapter_names = {ordinary: "default"}
    for index, condition in enumerate((value for value in model_conditions if value != ordinary), start=1):
        name = f"reroute_model_{index}"
        model.load_adapter(str(root / adapters[condition]["path"]), adapter_name=name, is_trainable=False)
        adapter_names[condition] = name
    model.config.use_cache = True
    block = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)[int(intervention["layer"])]
    device_tensors = {name: value.to(device=model.device, dtype=model.dtype) for name, value in tensors.items()}
    sources = {str(row["source_id"]): row for row in rows}
    for condition, model_condition, direction_name in arms:
        if condition in completed:
            continue
        model.set_adapter(adapter_names[model_condition])
        prepared, _ = prepare_requests(
            tokenizer,
            spec,
            config,
            "teacher_no_intervention",
            "alignment",
            rows,
            prompt_cap=int(profile["max_prompt_tokens"]),
            dataset_split=dataset_split,
        )
        scale = float(intervention["random_forward_scale"]) if direction_name == "random" else None
        before = len(generations)
        with full_state_projection(block, device_tensors[direction_name], scale):
            generations.extend(
                generate_hf_batches(
                    model,
                    tokenizer,
                    prepared,
                    profile=profile,
                    samples=samples,
                    batch_size=batch_size,
                    condition=condition,
                    kind="alignment",
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
        for row in generations[before:]:
            if str(row["condition"]) != condition:
                raise RuntimeError("generated Issue 19 reroute row lost its intervention-arm identity")
            row.update(adapters[model_condition])
            row["intervention_direction"] = direction_name
            row["intervention_tensor_sha256"] = (
                intervention["U_med_sha256"] if direction_name == "U_med" else intervention["U_reroute_sha256"]
            )
        report = write_outputs(
            output_dir,
            config,
            spec,
            f"issue19_reroute_{surface}",
            generations,
            sources,
            checkpoint_id=str(issue["rerouting"]["fit_checkpoint"]),
            status="generation_in_progress",
        )
        completed.append(condition)
        report["issue19_reroute_causal"] = {
            "contract": contract,
            "contract_sha256": contract_sha256,
            "completed_arms": completed,
            "engine": "transformers_sdpa",
            "batch_size": batch_size,
        }
        report["status"] = "generated_unscored" if len(completed) == len(arms) else "generation_in_progress"
        write_json_atomic(output_dir / "summary.json", report)
    return report


def scores_for(path: Path) -> dict[str, Any]:
    return numeric_scores(read_jsonl(path / "alignment_generations.jsonl"), read_jsonl(path / "judgments.jsonl"))


def indexed_generations(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    selected = {}
    for row in rows:
        if str(row["condition"]) != condition:
            continue
        key = f"{row['source_id']}:sample:{row['sample_index']}"
        if key in selected:
            raise RuntimeError(f"duplicate Issue 19 reroute generation identity: {condition}:{key}")
        selected[key] = row
    return selected


def require_reused_generation_contract(candidate_path: Path, baseline_path: Path) -> None:
    candidate = json.loads((candidate_path / "resolved_spec.json").read_text())
    baseline = json.loads((baseline_path / "resolved_spec.json").read_text())
    if candidate["resolved_config"]["models"]["teacher"] != baseline["resolved_config"]["models"]["teacher"]:
        raise RuntimeError("Issue 19 reroute teacher model differs from reused baseline")
    candidate_profile = candidate["resolved_config"]["generation"]["alignment_eval_development"]
    baseline_profile = baseline["resolved_config"]["generation"]["alignment_eval_final"]
    if any(candidate_profile[key] != baseline_profile[key] for key in SAMPLING_KEYS):
        raise RuntimeError("Issue 19 reroute sampling parameters differ from reused baseline")
    for key in ("alignment_evaluation.broad_alignment", "alignment_evaluation.broad_coherence"):
        if candidate["prompts"][key] != baseline["prompts"][key]:
            raise RuntimeError(f"Issue 19 reroute judge prompt differs from reused baseline: {key}")
    lineage = "azure_luna_none_v1"
    if (
        candidate["resolved_config"]["judge"]["lineages"][lineage]
        != baseline["resolved_config"]["judge"]["lineages"][lineage]
    ):
        raise RuntimeError("Issue 19 reroute judge lineage differs from reused baseline")


def summarize(
    config_path: Path,
    surface: str,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(ensure_within_workspace(config_path))
    arms = section_arms(section_name)
    baseline_for = {condition: model_condition for condition, model_condition, _ in arms}
    rerouting = config[section_name]["rerouting"]
    candidate_path = ensure_within_workspace(root / str(rerouting["output_dir"]) / f"causal_{surface}")
    baseline_path = ensure_within_workspace(root / str(rerouting["reused_no_intervention"][surface]["run_dir"]))
    require_reused_generation_contract(candidate_path, baseline_path)
    candidate_generations = read_jsonl(candidate_path / "alignment_generations.jsonl")
    baseline_generations = read_jsonl(baseline_path / "alignment_generations.jsonl")
    candidate = numeric_scores(candidate_generations, read_jsonl(candidate_path / "judgments.jsonl"))
    baseline = numeric_scores(baseline_generations, read_jsonl(baseline_path / "judgments.jsonl"))
    expected = {name for name, _, _ in arms}
    if set(candidate) != expected:
        raise RuntimeError("Issue 19 reroute causal judgments are incomplete")
    seed = int(config["experiment"]["seed"])
    samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])

    effects = {}
    metrics = {}
    for index, condition in enumerate(sorted(candidate)):
        baseline_condition = baseline_for[condition]
        candidate_rows = indexed_generations(candidate_generations, condition)
        baseline_rows = indexed_generations(baseline_generations, baseline_condition)
        comparable = {
            key: (
                row["question"],
                row["prompt"],
                row["prompt_token_ids"],
                row["dataset_split"],
                row["seed"],
                row["max_completion_tokens"],
            )
            for key, row in candidate_rows.items()
        }
        expected = {
            key: (
                row["question"],
                row["prompt"],
                row["prompt_token_ids"],
                row["dataset_split"],
                row["seed"],
                row["max_completion_tokens"],
            )
            for key, row in baseline_rows.items()
        }
        if comparable != expected:
            raise RuntimeError(f"Issue 19 reroute inputs differ from reused baseline: {condition}")
        shared = sorted(set(candidate[condition]) & set(baseline[baseline_condition]))
        if set(shared) != set(candidate_rows):
            raise RuntimeError(f"Issue 19 reroute condition lacks complete numeric paired judgments: {condition}")
        lengths = [int(row["completion_tokens"]) for row in candidate_rows.values()]
        metrics[condition] = {
            "numeric_pairs": len(shared),
            "alignment_score_mean": fmean(candidate[condition][key]["alignment"] for key in shared),
            "coherence_score_mean": fmean(candidate[condition][key]["coherence"] for key in shared),
            "coherence_guardrail_rate": fmean(
                candidate[condition][key]["coherence"] > coherence_cutoff for key in shared
            ),
            "EM_present_rate": fmean(
                candidate[condition][key]["alignment"] < alignment_cutoff
                and candidate[condition][key]["coherence"] > coherence_cutoff
                for key in shared
            ),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in candidate_rows.values()),
        }
        effects[condition] = {
            metric: paired_mean_bootstrap(
                {key: candidate[condition][key][metric] for key in shared},
                {key: baseline[baseline_condition][key][metric] for key in shared},
                seed=seed + 10 * index + metric_index,
                samples=samples,
                direction=f"{condition}_minus_{baseline_condition}",
            )
            for metric_index, metric in enumerate(("alignment", "coherence"))
        }

    def difference_in_effect(first: str, second: str, offset: int) -> dict[str, Any]:
        first_base = baseline_for[first]
        second_base = baseline_for[second]
        shared = sorted(
            set(candidate[first]) & set(baseline[first_base]) & set(candidate[second]) & set(baseline[second_base])
        )
        return {
            metric: paired_mean_bootstrap(
                {key: candidate[first][key][metric] - baseline[first_base][key][metric] for key in shared},
                {key: candidate[second][key][metric] - baseline[second_base][key][metric] for key in shared},
                seed=seed + offset + metric_index,
                samples=samples,
                direction=f"effect_{first}_minus_effect_{second}",
            )
            for metric_index, metric in enumerate(("alignment", "coherence"))
        }

    report = {
        "schema_version": 1,
        "status": "scored",
        "surface": surface,
        "metrics": metrics,
        "paired_ablation_effects": effects,
        "specificity": {
            "reroute_minus_matched_random_in_full_target": difference_in_effect(
                "full_target_U_reroute_ablation", "full_target_U_reroute_matched_random", 100
            ),
            "reroute_effect_full_target_minus_full_random": difference_in_effect(
                "full_target_U_reroute_ablation", "full_random_U_reroute_ablation", 200
            ),
            "reroute_effect_full_target_minus_ordinary": difference_in_effect(
                "full_target_U_reroute_ablation", "ordinary_U_reroute_ablation", 300
            ),
        },
    }
    write_json_atomic(candidate_path / "reroute_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--section",
        choices=("issue19_local_vs_global", "medical_all_tasks_subspace_followup"),
        default="issue19_local_vs_global",
    )
    parser.add_argument("--surface", choices=("medical", "broad48"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    guard = require_active_guard()
    if args.command == "generate":
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("Issue 19 reroute generation requires elevated guarded GPU execution")
        report = generate(args.config, args.surface, args.batch_size, section_name=args.section)
    else:
        report = summarize(args.config, args.surface, section_name=args.section)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
