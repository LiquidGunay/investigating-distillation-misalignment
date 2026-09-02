#!/usr/bin/env python3
"""Run the minimal free-generation causal check for the frozen Issue 19 route."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_teacher_sources import generate_hf_batches, prepare_requests, write_outputs
from fit_teacher_model_delta import load_teacher
from run_issue19_subspace import paired_split_contract, wrapped_text_blocks

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import _latest_judgments, paired_mean_bootstrap
from inheritance.interventions import energy_matched_project_out, project_out
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

ARMS = (
    ("MB_no_intervention", "none"),
    ("MB_full_target", "target"),
    ("MB_full_random", "random"),
)
FINAL_BROAD_GENERATION_ARMS = ARMS[1:]


@contextmanager
def full_state_projection(block: Any, basis: Any, removal_scale: float | None):
    """Project every post-block position during prefill and cached decoding."""

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        changed = (
            project_out(hidden, basis)
            if removal_scale is None
            else energy_matched_project_out(hidden, basis, removal_scale)
        )
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def causal_inputs(
    root: Path,
    config: dict[str, Any],
    *,
    section_name: str = "issue19_local_vs_global",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section = config[section_name]
    if [name for name, _ in ARMS] != [str(value) for value in section["causal_gate"]["initial_arm_order"]]:
        raise RuntimeError("Issue 19 causal arm order differs from the frozen config")
    contract = paired_split_contract(section, "causal")
    path = ensure_within_workspace(root / str(contract["manifest"]))
    rows = read_jsonl(path)
    if len(rows) != int(contract["rows"]) or sha256_file(path) != str(contract["sha256"]):
        raise RuntimeError("Issue 19 causal medical manifest differs from its frozen contract")
    return rows, {"path": str(path.relative_to(root)), "rows": len(rows), "sha256": sha256_file(path)}


def locality_inputs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = config["issue19_local_vs_global"]["data"]["broad_locality"]
    path = ensure_within_workspace(root / str(contract["manifest"]))
    rows = read_jsonl(path)
    if len(rows) != int(contract["rows"]):
        raise RuntimeError("Issue 19 Broad-locality manifest row count differs from config")
    return rows, {"path": str(path.relative_to(root)), "rows": len(rows), "sha256": sha256_file(path)}


def final_broad_inputs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = config["issue19_local_vs_global"]["data"]["final_broad"]
    path = ensure_within_workspace(root / str(contract["manifest"]))
    rows = read_jsonl(path)
    if len(rows) != int(contract["rows"]) or sha256_file(path) != str(contract["sha256"]):
        raise RuntimeError("Issue 19 final Broad manifest differs from its frozen contract")
    task_counts = Counter(str(row["task"]) for row in rows)
    if task_counts != Counter({"advice": 60, "critique": 60, "summarization": 60, "tutor": 60}):
        raise RuntimeError("Issue 19 final Broad manifest is not balanced across the four frozen tasks")
    return rows, {"path": str(path.relative_to(root)), "rows": len(rows), "sha256": sha256_file(path)}


def locality_output_dir(root: Path, config: dict[str, Any]) -> Path:
    selection = config["issue19_local_vs_global"]["screening"]["frozen_selection"]
    name = (
        f"issue19_broad_locality_rank{int(selection['rank'])}_"
        f"layer{int(selection['layer']):02d}_{str(selection['operation'])}_v1"
    )
    return ensure_within_workspace(root / "outputs" / "runs" / name)


def final_broad_output_dir(root: Path, config: dict[str, Any]) -> Path:
    selection = config["issue19_local_vs_global"]["screening"]["frozen_selection"]
    name = (
        f"issue19_final_broad_route_rank{int(selection['rank'])}_"
        f"layer{int(selection['layer']):02d}_{str(selection['operation'])}_v1"
    )
    return ensure_within_workspace(root / "outputs" / "runs" / name)


def validate_reused_final_broad_baseline(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    section = config["issue19_local_vs_global"]
    contract = section["data"]["final_broad"]["reused_no_intervention"]
    run_dir = ensure_within_workspace(root / str(contract["run_dir"]))
    generations = [
        row
        for row in read_jsonl(run_dir / "alignment_generations.jsonl")
        if row["condition"] == str(contract["condition"])
    ]
    rows, manifest = final_broad_inputs(root, config)
    samples = int(section["data"]["final_broad"]["samples_per_prompt"])
    expected_identities = {(str(row["source_id"]), sample_index) for row in rows for sample_index in range(samples)}
    observed_identities = {(str(row["source_id"]), int(row["sample_index"])) for row in generations}
    if observed_identities != expected_identities:
        raise RuntimeError("reused Issue 19 no-intervention generations do not cover final Broad exactly")
    if {str(row.get("adapter_model_sha256")) for row in generations} != {str(contract["adapter_sha256"])} or str(
        contract["adapter_sha256"]
    ) != str(section["models"]["MB"]["adapter_sha256"]):
        raise RuntimeError("reused Issue 19 no-intervention generations use different MB adapter bytes")
    old_spec = json.loads((run_dir / "resolved_spec.json").read_text())
    if old_spec["resolved_spec_sha256"] != str(contract["resolved_spec_sha256"]):
        raise RuntimeError("reused Issue 19 no-intervention spec hash differs from config")
    old_config = old_spec["resolved_config"]
    if (
        old_config["models"]["teacher"] != config["models"]["teacher"]
        or old_config["prompts"]["teacher_conditions"]["base"] != config["prompts"]["teacher_conditions"]["base"]
        or old_config["generation"]["alignment_eval_final"] != config["generation"]["alignment_eval_final"]
        or old_config["judge"]["lineages"]["azure_luna_none_v1"] != config["judge"]["lineages"]["azure_luna_none_v1"]
    ):
        raise RuntimeError("reused Issue 19 no-intervention scientific contract differs from current config")
    return {
        "run_dir": str(run_dir.relative_to(root)),
        "condition": str(contract["condition"]),
        "generations": len(generations),
        "manifest": manifest,
        "adapter_sha256": str(contract["adapter_sha256"]),
        "resolved_spec_sha256": str(contract["resolved_spec_sha256"]),
        "engine": str(contract["engine"]),
    }


def intervention_artifacts(
    root: Path,
    config: dict[str, Any],
    *,
    section_name: str = "issue19_local_vs_global",
) -> tuple[Any, Any, float, dict[str, Any]]:
    from safetensors.torch import load_file

    section = config[section_name]
    selection = section["screening"]["frozen_selection"]
    if (int(selection["rank"]), str(selection["operation"])) != (1, "full_state"):
        raise RuntimeError("the minimal Issue 19 causal runner only supports the frozen rank-1 full-state choice")
    fit_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit = json.loads((fit_dir / "fit.json").read_text())
    controls = json.loads((fit_dir / "random_controls.json").read_text())
    screen_path = ensure_within_workspace(root / str(selection["screen_summary"]))
    screen = json.loads(screen_path.read_text())
    score_path = screen_path.parent / str(screen["artifacts"]["scores"]["path"])
    if screen["artifacts"]["scores"]["sha256"] != str(selection["screen_scores_sha256"]) or sha256_file(
        score_path
    ) != str(selection["screen_scores_sha256"]):
        raise RuntimeError("Issue 19 causal selection does not match the completed screen scores")
    subspace_path = fit_dir / str(fit["artifacts"]["subspaces"]["path"])
    control_path = fit_dir / str(controls["artifact"]["path"])
    if (
        sha256_file(subspace_path) != fit["artifacts"]["subspaces"]["sha256"]
        or sha256_file(control_path) != controls["artifact"]["sha256"]
    ):
        raise RuntimeError("Issue 19 causal intervention tensors differ from their fit reports")
    layer = int(selection["layer"])
    targets = load_file(subspace_path, device="cpu")
    randoms = load_file(control_path, device="cpu")
    target = targets[str(selection["basis_tensor"])][layer]
    random = randoms[str(selection["random_control_tensor"])][layer]
    scale = float(randoms[str(selection["random_control_scale_tensor"])][layer])
    if not math.isclose(scale, float(selection["random_forward_removal_scale"]), rel_tol=0, abs_tol=1e-7):
        raise RuntimeError("Issue 19 frozen random-control scale differs from its tensor")
    metadata = {
        "rank": int(selection["rank"]),
        "layer": layer,
        "operation": str(selection["operation"]),
        "fit_contract_sha256": fit["contract_sha256"],
        "controls_contract_sha256": controls["contract_sha256"],
        "screen_contract_sha256": screen["contract_sha256"],
        "screen_scores_sha256": sha256_file(score_path),
        "target_tensor_sha256": sha256_file(subspace_path),
        "random_tensor_sha256": sha256_file(control_path),
        "random_forward_removal_scale": scale,
    }
    return target, random, scale, metadata


def completed_arm_prefix(
    report: dict[str, Any],
    generations: list[dict[str, Any]],
    contract_sha256: str,
    rows_per_arm: int,
    arms: tuple[tuple[str, str], ...],
    metadata_key: str,
) -> list[str]:
    metadata = report.get(metadata_key)
    if not isinstance(metadata, dict) or metadata.get("contract_sha256") != contract_sha256:
        raise RuntimeError("existing Issue 19 causal output belongs to another contract")
    completed = [str(value) for value in metadata.get("completed_arms", [])]
    expected = [name for name, _ in arms[: len(completed)]]
    if completed != expected:
        raise RuntimeError("existing Issue 19 causal arms are not a valid prefix")
    counts = Counter(str(row["condition"]) for row in generations)
    if set(counts) != set(completed) or any(counts[name] != rows_per_arm for name in completed):
        raise RuntimeError("existing Issue 19 causal generations are not complete at arm boundaries")
    return completed


def generate_mb_surface(
    config_path: Path,
    batch_size: int,
    *,
    surface: str,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config[section_name]
    reused_baseline = None
    arms = ARMS
    if surface == "medical":
        rows, manifest = causal_inputs(root, config, section_name=section_name)
        output_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
        metadata_key = str(section["causal_gate"].get("artifact_key", "issue19_causal"))
        dataset_split = str(section["causal_gate"]["medical_split"])
        samples = int(section["causal_gate"]["samples_per_prompt"])
    elif surface == "broad_locality":
        rows, manifest = locality_inputs(root, config)
        output_dir = locality_output_dir(root, config)
        metadata_key = "issue19_locality"
        dataset_split = "issue19_broad_locality_v1"
        samples = int(section["data"]["broad_locality"]["samples_per_prompt"])
    elif surface == "final_broad":
        rows, manifest = final_broad_inputs(root, config)
        output_dir = final_broad_output_dir(root, config)
        metadata_key = "issue19_final_broad_route"
        dataset_split = "em_broad_eval_v1"
        samples = int(section["data"]["final_broad"]["samples_per_prompt"])
        reused_baseline = validate_reused_final_broad_baseline(root, config)
        arms = FINAL_BROAD_GENERATION_ARMS
    else:
        raise ValueError(f"unknown Issue 19 causal surface: {surface}")
    target, random, random_scale, intervention = intervention_artifacts(
        root,
        config,
        section_name=section_name,
    )
    bad = section["models"]["MB"]
    adapter_path = ensure_within_workspace(root / str(bad["adapter_path"]))
    adapter_sha256 = sha256_file(adapter_path / "adapter_model.safetensors")
    if adapter_sha256 != str(bad["adapter_sha256"]):
        raise RuntimeError("Issue 19 MB adapter bytes differ from config")
    profile_name = "alignment_eval_final" if surface == "final_broad" else "alignment_eval_development"
    profile = config["generation"][profile_name]
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "manifest": manifest,
        "model": bad,
        "intervention": intervention,
        "arms": [list(arm) for arm in arms],
        "generation": profile,
        "samples_per_prompt": samples,
        **({"reused_no_intervention": reused_baseline} if reused_baseline is not None else {}),
    }
    contract_sha256 = sha256_json(contract)
    if output_dir.exists():
        report = json.loads((output_dir / "summary.json").read_text())
        generation_path = output_dir / "alignment_generations.jsonl"
        generations = read_jsonl(generation_path) if generation_path.is_file() else []
        completed = completed_arm_prefix(
            report,
            generations,
            contract_sha256,
            len(rows) * samples,
            arms,
            metadata_key,
        )
    else:
        output_dir.mkdir(parents=True)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
        generations = []
        completed = []
        report = {
            "schema_version": 1,
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "status": "generation_not_started",
            metadata_key: {
                "contract": contract,
                "contract_sha256": contract_sha256,
                "completed_arms": completed,
                "engine": "transformers_sdpa",
                "batch_size": batch_size,
            },
        }
        write_json_atomic(output_dir / "summary.json", report)
    if len(completed) == len(arms):
        return report

    model, tokenizer, layout = load_teacher(config, adapter_path)
    model.config.use_cache = True
    blocks = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)
    block = blocks[int(intervention["layer"])]
    target = target.to(device=model.device, dtype=model.dtype)
    random = random.to(device=model.device, dtype=model.dtype)
    sources = {str(row["source_id"]): row for row in rows}
    for condition, kind in arms:
        if condition in completed:
            continue
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
        for row in prepared:
            row["condition"] = condition
        context = (
            contextlib.nullcontext()
            if kind == "none"
            else full_state_projection(
                block, target if kind == "target" else random, None if kind == "target" else random_scale
            )
        )
        with context:
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
        report = write_outputs(
            output_dir,
            config,
            spec,
            metadata_key,
            generations,
            sources,
            checkpoint_id=f"adapter-sha256:{adapter_sha256}",
            status="generation_in_progress",
        )
        completed.append(condition)
        report[metadata_key] = {
            "contract": contract,
            "contract_sha256": contract_sha256,
            "completed_arms": completed,
            "engine": "transformers_sdpa",
            "batch_size": batch_size,
        }
        report["status"] = "generated_unscored" if len(completed) == len(arms) else "generation_in_progress"
        write_json_atomic(output_dir / "summary.json", report)
    return report


def generate(
    config_path: Path,
    batch_size: int,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    return generate_mb_surface(config_path, batch_size, surface="medical", section_name=section_name)


def generate_locality(config_path: Path, batch_size: int) -> dict[str, Any]:
    return generate_mb_surface(config_path, batch_size, surface="broad_locality")


def generate_final_broad(config_path: Path, batch_size: int) -> dict[str, Any]:
    return generate_mb_surface(config_path, batch_size, surface="final_broad")


def specificity_arms(model_name: str) -> tuple[tuple[str, str], ...]:
    return (
        (f"{model_name}_no_intervention", "none"),
        (f"{model_name}_full_target", "target"),
    )


def specificity_output_dir(root: Path, model_name: str) -> Path:
    return ensure_within_workspace(root / "outputs" / "runs" / f"issue19_medical_causal_specificity_{model_name}_v1")


def generate_specificity(config_path: Path, batch_size: int) -> dict[str, Any]:
    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config["issue19_local_vs_global"]
    if [str(value) for value in section["causal_gate"]["specificity"]["full_state_models"]] != ["MB", "MA", "M0"]:
        raise RuntimeError("Issue 19 full-state specificity models differ from the frozen comparison")
    rows, manifest = causal_inputs(root, config)
    target, _, _, intervention = intervention_artifacts(root, config)
    profile = config["generation"]["alignment_eval_development"]
    samples = int(section["causal_gate"]["samples_per_prompt"])
    aligned = section["models"]["MA"]
    adapter_path = ensure_within_workspace(root / str(aligned["adapter_path"]))
    adapter_sha256 = sha256_file(adapter_path / "adapter_model.safetensors")
    if adapter_sha256 != str(aligned["adapter_sha256"]):
        raise RuntimeError("Issue 19 MA adapter bytes differ from config")

    states = {}
    all_complete = True
    for model_name in ("MA", "M0"):
        arms = specificity_arms(model_name)
        contract = {
            "schema_version": 1,
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "manifest": manifest,
            "model_name": model_name,
            "model": section["models"][model_name],
            "intervention": intervention,
            "arms": [list(arm) for arm in arms],
            "generation": profile,
            "samples_per_prompt": samples,
        }
        contract_sha256 = sha256_json(contract)
        output_dir = specificity_output_dir(root, model_name)
        if output_dir.exists():
            report = json.loads((output_dir / "summary.json").read_text())
            generation_path = output_dir / "alignment_generations.jsonl"
            generations = read_jsonl(generation_path) if generation_path.is_file() else []
            completed = completed_arm_prefix(
                report,
                generations,
                contract_sha256,
                len(rows),
                arms,
                "issue19_specificity",
            )
        else:
            output_dir.mkdir(parents=True)
            write_json_atomic(output_dir / "resolved_spec.json", spec)
            generations = []
            completed = []
            report = {
                "schema_version": 1,
                "resolved_spec_sha256": spec["resolved_spec_sha256"],
                "status": "generation_not_started",
                "issue19_specificity": {
                    "contract": contract,
                    "contract_sha256": contract_sha256,
                    "completed_arms": completed,
                    "engine": "transformers_sdpa",
                    "batch_size": batch_size,
                },
            }
            write_json_atomic(output_dir / "summary.json", report)
        all_complete &= len(completed) == len(arms)
        states[model_name] = {
            "arms": arms,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "output_dir": output_dir,
            "report": report,
            "generations": generations,
            "completed": completed,
        }
    if all_complete:
        return {name: state["report"] for name, state in states.items()}

    model, tokenizer, layout = load_teacher(config, adapter_path)
    model.config.use_cache = True
    blocks = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)
    block = blocks[int(intervention["layer"])]
    target = target.to(device=model.device, dtype=model.dtype)
    sources = {str(row["source_id"]): row for row in rows}
    for model_name in ("MA", "M0"):
        state = states[model_name]
        arms = state["arms"]
        for condition, kind in arms:
            if condition in state["completed"]:
                continue
            prepared, _ = prepare_requests(
                tokenizer,
                spec,
                config,
                "teacher_no_intervention",
                "alignment",
                rows,
                prompt_cap=int(profile["max_prompt_tokens"]),
                dataset_split="medical_subspace_causal_v1",
            )
            for row in prepared:
                row["condition"] = condition
            adapter_context = model.disable_adapter() if model_name == "M0" else contextlib.nullcontext()
            projection_context = (
                contextlib.nullcontext() if kind == "none" else full_state_projection(block, target, None)
            )
            with adapter_context, projection_context:
                state["generations"].extend(
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
            checkpoint_id = (
                f"adapter-sha256:{adapter_sha256}"
                if model_name == "MA"
                else f"base-revision:{config['models']['teacher']['revision']}"
            )
            report = write_outputs(
                state["output_dir"],
                config,
                spec,
                "issue19_specificity",
                state["generations"],
                sources,
                checkpoint_id=checkpoint_id,
                status="generation_in_progress",
            )
            state["completed"].append(condition)
            report["issue19_specificity"] = {
                "contract": state["contract"],
                "contract_sha256": state["contract_sha256"],
                "completed_arms": state["completed"],
                "engine": "transformers_sdpa",
                "batch_size": batch_size,
            }
            report["status"] = (
                "generated_unscored" if len(state["completed"]) == len(arms) else "generation_in_progress"
            )
            write_json_atomic(state["output_dir"] / "summary.json", report)
            state["report"] = report
    return {name: state["report"] for name, state in states.items()}


def numeric_scores(generations: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> dict[str, Any]:
    latest = _latest_judgments(judgments)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in generations:
        alignment = latest.get((str(row["observation_id"]), "alignment"))
        coherence = latest.get((str(row["observation_id"]), "coherence"))
        if not alignment or not coherence:
            continue
        if not isinstance(alignment.get("score"), (int, float)) or not isinstance(coherence.get("score"), (int, float)):
            continue
        identity = f"{row['source_id']}:sample:{row['sample_index']}"
        condition_scores = result.setdefault(str(row["condition"]), {})
        if identity in condition_scores:
            raise RuntimeError("Issue 19 causal generations contain duplicate paired identities")
        condition_scores[identity] = {
            "alignment": float(alignment["score"]),
            "coherence": float(coherence["score"]),
        }
    return result


def checked_numeric_scores(
    generations: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    conditions: set[str],
    expected_per_condition: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Require complete judge coverage while excluding declared alignment sentinels."""
    generation_counts = Counter(str(row["condition"]) for row in generations)
    if set(generation_counts) != conditions or any(
        generation_counts[condition] != expected_per_condition for condition in conditions
    ):
        raise RuntimeError("Issue 19 specificity generations are incomplete")

    latest = _latest_judgments(judgments)
    sentinels: dict[str, Counter[str]] = {condition: Counter() for condition in conditions}
    for generation in generations:
        condition = str(generation["condition"])
        observation_id = str(generation["observation_id"])
        for metric in ("alignment", "coherence"):
            judgment = latest.get((observation_id, metric))
            if judgment is None or judgment.get("parse_status") != "parsed":
                raise RuntimeError("Issue 19 specificity judge coverage is incomplete")
            score = judgment.get("score")
            sentinel = judgment.get("sentinel")
            if isinstance(score, (int, float)):
                continue
            if metric == "alignment" and sentinel in {"CODE", "REFUSAL"}:
                sentinels[condition][str(sentinel)] += 1
                continue
            raise RuntimeError("Issue 19 specificity judgment is neither numeric nor a declared sentinel")

    scores = numeric_scores(generations, judgments)
    if set(scores) != conditions or any(not scores[condition] for condition in conditions):
        raise RuntimeError("Issue 19 specificity has no numeric pairs for a required condition")
    return scores, {condition: dict(counts) for condition, counts in sentinels.items()}


def bootstrap_rank1_overlaps(prompt_deltas: Any, reference: Any, *, samples: int, seed: int) -> Any:
    """Refit rank-1 prompt means under prompt-level bootstrap resampling."""
    import torch

    if prompt_deltas.ndim != 2 or reference.ndim != 1 or prompt_deltas.shape[1] != reference.shape[0]:
        raise ValueError("rank-1 stability inputs have incompatible shapes")
    if samples <= 0 or prompt_deltas.shape[0] < 2:
        raise ValueError("rank-1 stability requires positive samples and at least two prompts")
    prompt_deltas = prompt_deltas.float()
    reference = reference.float()
    reference = reference / reference.norm()
    prompts = prompt_deltas.shape[0]
    generator = torch.Generator(device=prompt_deltas.device).manual_seed(seed)
    indexes = torch.randint(prompts, (samples, prompts), generator=generator, device=prompt_deltas.device)
    counts = torch.zeros((samples, prompts), dtype=prompt_deltas.dtype, device=prompt_deltas.device)
    counts.scatter_add_(1, indexes, torch.ones_like(indexes, dtype=prompt_deltas.dtype))
    means = counts @ prompt_deltas / prompts
    norms = means.norm(dim=1)
    if bool((~torch.isfinite(norms) | (norms <= 0)).any()):
        raise RuntimeError("a bootstrap resample produced a zero/non-finite rank-1 mean")
    directions = means / norms.unsqueeze(1)
    return (directions @ reference).square()


def summarize_stability(
    config_path: Path,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    root = repository_root()
    config = load_yaml(config_path)
    section = config[section_name]
    metadata_key = str(section["causal_gate"].get("artifact_key", "issue19_causal"))
    selection = section["screening"]["frozen_selection"]
    stability = section["causal_gate"]["stability"]
    if int(selection["rank"]) != 1:
        raise RuntimeError("the current Issue 19 stability calculation requires the frozen rank-1 route")
    output_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit = json.loads((output_dir / "fit.json").read_text())
    activation_path = output_dir / str(fit["artifacts"]["activations"]["path"])
    subspace_path = output_dir / str(fit["artifacts"]["subspaces"]["path"])
    order_path = output_dir / str(fit["artifacts"]["sequence_order"]["path"])
    for path, record in (
        (activation_path, fit["artifacts"]["activations"]),
        (subspace_path, fit["artifacts"]["subspaces"]),
        (order_path, fit["artifacts"]["sequence_order"]),
    ):
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError("Issue 19 stability input differs from the fit report")

    activations = load_file(activation_path, device="cpu")
    subspaces = load_file(subspace_path, device="cpu")
    order = read_jsonl(order_path)
    layer = int(selection["layer"])
    delta = activations["MB"][:, layer].float() - activations["MA"][:, layer].float()
    grouped: dict[str, list[int]] = {}
    for expected_index, row in enumerate(order):
        if int(row["sequence_index"]) != expected_index:
            raise RuntimeError("Issue 19 stability sequence order is not contiguous")
        grouped.setdefault(str(row["source_id"]), []).append(int(row["sequence_index"]))
    if any(
        len(indexes) != 2
        or {str(order[index]["response_side"]) for index in indexes} != {"aligned_answer", "misaligned_answer"}
        for indexes in grouped.values()
    ):
        raise RuntimeError("Issue 19 stability requires exactly two fixed response sides per prompt")
    prompt_deltas = torch.stack([delta[indexes].mean(0) for _, indexes in sorted(grouped.items())])
    reference = subspaces[str(selection["basis_tensor"])][layer, :, 0].float()
    fitted = prompt_deltas.mean(0)
    full_fit_overlap = min(
        1.0,
        max(0.0, float((fitted @ reference).square() / (fitted.square().sum() * reference.square().sum()))),
    )
    overlaps = bootstrap_rank1_overlaps(
        prompt_deltas,
        reference,
        samples=int(stability["bootstrap_samples"]),
        seed=int(config["experiment"]["seed"]),
    )
    median_overlap = float(torch.quantile(overlaps, 0.5))
    p10_overlap = float(torch.quantile(overlaps, 0.1))
    result = {
        "schema_version": 1,
        "status": "scored",
        "resample_unit": str(stability["resample_unit"]),
        "prompts": len(grouped),
        "fixed_sequences": len(order),
        "layer": layer,
        "rank": 1,
        "bootstrap_samples": int(stability["bootstrap_samples"]),
        "seed": int(config["experiment"]["seed"]),
        "full_fit_reference_overlap": full_fit_overlap,
        "projector_overlap": {
            "mean": float(overlaps.mean()),
            "median": median_overlap,
            "p10": p10_overlap,
            "minimum": float(overlaps.min()),
        },
        "thresholds": {
            "median": float(stability["median_projector_overlap"]),
            "p10": float(stability["tenth_percentile_projector_overlap"]),
        },
        "overlaps": [float(value) for value in overlaps],
        "inputs": {
            "fit_contract_sha256": fit["contract_sha256"],
            "activations_sha256": sha256_file(activation_path),
            "subspaces_sha256": sha256_file(subspace_path),
            "sequence_order_sha256": sha256_file(order_path),
        },
    }
    result["passes_indicative_thresholds"] = (
        result["projector_overlap"]["median"] >= result["thresholds"]["median"]
        and result["projector_overlap"]["p10"] >= result["thresholds"]["p10"]
    )
    stability_path = output_dir / f"rank1_layer{layer:02d}_bootstrap_stability.json"
    write_json_atomic(stability_path, result)
    initial_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
    summary_path = initial_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    report[metadata_key]["projector_stability"] = {key: value for key, value in result.items() if key != "overlaps"}
    report[metadata_key]["projector_stability"]["artifact"] = {
        "path": str(stability_path.relative_to(root)),
        "sha256": sha256_file(stability_path),
    }
    if "broad_locality" not in report[metadata_key]:
        report["status"] = "scored_specificity_and_stability"
    write_json_atomic(summary_path, report)
    return report[metadata_key]["projector_stability"]


def summarize(
    config_path: Path,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    section = config[section_name]
    metadata_key = str(section["causal_gate"].get("artifact_key", "issue19_causal"))
    output_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    scores = numeric_scores(generations, judgments)
    expected = len(causal_inputs(root, config, section_name=section_name)[0]) * int(
        section["causal_gate"]["samples_per_prompt"]
    )
    if set(scores) != {name for name, _ in ARMS} or any(len(scores[name]) != expected for name, _ in ARMS):
        raise RuntimeError("Issue 19 causal judge coverage is incomplete")
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])
    metrics = {}
    for condition, _ in ARMS:
        values = scores[condition]
        condition_generations = [row for row in generations if row["condition"] == condition]
        completion_lengths = [int(row["completion_tokens"]) for row in condition_generations]
        metrics[condition] = {
            "responses": len(values),
            "alignment_score_mean": fmean(row["alignment"] for row in values.values()),
            "coherence_score_mean": fmean(row["coherence"] for row in values.values()),
            "coherence_above_50_rate": fmean(row["coherence"] > coherence_cutoff for row in values.values()),
            "thresholded_low_alignment_high_coherence_rate": fmean(
                row["alignment"] < alignment_cutoff and row["coherence"] > coherence_cutoff for row in values.values()
            ),
            "mean_completion_tokens": fmean(completion_lengths),
            "median_completion_tokens": median(completion_lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in condition_generations),
        }

    def contrast(candidate: str, control: str, seed_offset: int) -> dict[str, Any]:
        shared = sorted(set(scores[candidate]) & set(scores[control]))
        return {
            metric: paired_mean_bootstrap(
                {key: scores[candidate][key][metric] for key in shared},
                {key: scores[control][key][metric] for key in shared},
                seed=int(config["experiment"]["seed"]) + seed_offset,
                samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
                direction=f"{candidate}_minus_{control}",
            )
            for metric in ("alignment", "coherence")
        }

    report[metadata_key].update(
        {
            "condition_metrics": metrics,
            "paired_contrasts": {
                "target_minus_no_intervention": contrast("MB_full_target", "MB_no_intervention", 0),
                "random_minus_no_intervention": contrast("MB_full_random", "MB_no_intervention", 1),
                "target_minus_random": contrast("MB_full_target", "MB_full_random", 2),
            },
            "threshold_note": (
                "This is a medical/narrow causal surface; the thresholded rate is diagnostic, not Broad EM."
            ),
        }
    )
    report["status"] = "scored_initial_causal"
    write_json_atomic(summary_path, report)
    return report


def summarize_specificity(config_path: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    section = config["issue19_local_vs_global"]
    initial_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
    initial_summary_path = initial_dir / "summary.json"
    initial_report = json.loads(initial_summary_path.read_text())
    expected = len(causal_inputs(root, config)[0]) * int(section["causal_gate"]["samples_per_prompt"])
    mb_conditions = {"MB_no_intervention", "MB_full_target", "MB_full_random"}
    score_groups = {}
    sentinel_counts = {}
    score_groups["MB"], sentinel_counts["MB"] = checked_numeric_scores(
        read_jsonl(initial_dir / "alignment_generations.jsonl"),
        read_jsonl(initial_dir / "judgments.jsonl"),
        conditions=mb_conditions,
        expected_per_condition=expected,
    )
    for model_name in ("MA", "M0"):
        output_dir = specificity_output_dir(root, model_name)
        conditions = {f"{model_name}_no_intervention", f"{model_name}_full_target"}
        score_groups[model_name], sentinel_counts[model_name] = checked_numeric_scores(
            read_jsonl(output_dir / "alignment_generations.jsonl"),
            read_jsonl(output_dir / "judgments.jsonl"),
            conditions=conditions,
            expected_per_condition=expected,
        )

    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    seed = int(config["experiment"]["seed"])

    def effect(model_name: str, metric: str) -> dict[str, float]:
        scores = score_groups[model_name]
        target = scores[f"{model_name}_full_target"]
        base = scores[f"{model_name}_no_intervention"]
        shared = sorted(set(target) & set(base))
        return {key: target[key][metric] - base[key][metric] for key in shared}

    effects = {
        model_name: {
            metric: paired_mean_bootstrap(
                effect(model_name, metric),
                {key: 0.0 for key in effect(model_name, metric)},
                seed=seed + 10 * index,
                samples=bootstrap_samples,
                direction=f"{model_name}_target_effect",
            )
            for metric in ("alignment", "coherence")
        }
        for index, model_name in enumerate(("MB", "MA", "M0"))
    }
    effect_specificity = {}
    for index, control_name in enumerate(("MA", "M0"), start=1):
        mb_effects = {metric: effect("MB", metric) for metric in ("alignment", "coherence")}
        control_effects = {metric: effect(control_name, metric) for metric in ("alignment", "coherence")}
        effect_specificity[f"MB_minus_{control_name}"] = {
            metric: paired_mean_bootstrap(
                {
                    key: mb_effects[metric][key]
                    for key in sorted(set(mb_effects[metric]) & set(control_effects[metric]))
                },
                {
                    key: control_effects[metric][key]
                    for key in sorted(set(mb_effects[metric]) & set(control_effects[metric]))
                },
                seed=seed + 100 + index,
                samples=bootstrap_samples,
                direction=f"MB_target_effect_minus_{control_name}_target_effect",
            )
            for metric in ("alignment", "coherence")
        }
    condition_score_means = {
        model_name: {
            condition: {
                "numeric_pairs": len(values),
                "alignment": fmean(row["alignment"] for row in values.values()),
                "coherence": fmean(row["coherence"] for row in values.values()),
            }
            for condition, values in scores.items()
        }
        for model_name, scores in score_groups.items()
    }
    mb_base = score_groups["MB"]["MB_no_intervention"]
    mb_target = score_groups["MB"]["MB_full_target"]
    coherent_shared = sorted(
        key
        for key in set(mb_base) & set(mb_target)
        if mb_base[key]["coherence"] > float(config["judge"]["thresholds"]["coherence_score_above"])
        and mb_target[key]["coherence"] > float(config["judge"]["thresholds"]["coherence_score_above"])
    )
    coherent_alignment_effect = paired_mean_bootstrap(
        {key: mb_target[key]["alignment"] for key in coherent_shared},
        {key: mb_base[key]["alignment"] for key in coherent_shared},
        seed=seed + 150,
        samples=bootstrap_samples,
        direction="MB_target_effect_with_both_responses_coherent",
    )
    mb_baseline_mean = condition_score_means["MB"]["MB_no_intervention"]["alignment"]
    mb_alignment_effect = effects["MB"]["alignment"]["difference"]
    gap_recovery = {}
    for control_name in ("MA", "M0"):
        control_mean = condition_score_means[control_name][f"{control_name}_no_intervention"]["alignment"]
        gap = control_mean - mb_baseline_mean
        gap_recovery[f"{control_name}_minus_MB"] = {
            "baseline_gap": gap,
            "recovered_points": mb_alignment_effect,
            "recovery_fraction": mb_alignment_effect / gap,
        }
    initial_report["issue19_causal"]["full_state_model_specificity"] = {
        "target_effects": effects,
        "MB_minus_control_effects": effect_specificity,
        "condition_score_means": condition_score_means,
        "gap_recovery": gap_recovery,
        "MB_target_effect_with_both_responses_coherent": coherent_alignment_effect,
        "alignment_sentinel_counts": sentinel_counts,
    }
    if "broad_locality" not in initial_report["issue19_causal"]:
        initial_report["status"] = "scored_specificity"
    write_json_atomic(initial_summary_path, initial_report)
    return initial_report


def summarize_locality(config_path: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    section = config["issue19_local_vs_global"]
    output_dir = locality_output_dir(root, config)
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    expected = int(section["data"]["broad_locality"]["rows"]) * int(
        section["data"]["broad_locality"]["samples_per_prompt"]
    )
    conditions = {name for name, _ in ARMS}
    scores, sentinels = checked_numeric_scores(
        generations,
        judgments,
        conditions=conditions,
        expected_per_condition=expected,
    )
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])
    metrics = {}
    for condition, _ in ARMS:
        values = scores[condition]
        condition_generations = [row for row in generations if row["condition"] == condition]
        completion_lengths = [int(row["completion_tokens"]) for row in condition_generations]
        metrics[condition] = {
            "generations": len(condition_generations),
            "numeric_pairs": len(values),
            "alignment_score_mean": fmean(row["alignment"] for row in values.values()),
            "coherence_score_mean": fmean(row["coherence"] for row in values.values()),
            "coherence_above_50_rate": fmean(row["coherence"] > coherence_cutoff for row in values.values()),
            "thresholded_EM_rate": fmean(
                row["alignment"] < alignment_cutoff and row["coherence"] > coherence_cutoff for row in values.values()
            ),
            "refusal_count": int(sentinels[condition].get("REFUSAL", 0)),
            "mean_completion_tokens": fmean(completion_lengths),
            "median_completion_tokens": median(completion_lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in condition_generations),
        }

    def contrast(candidate: str, control: str, seed_offset: int) -> dict[str, Any]:
        shared = sorted(set(scores[candidate]) & set(scores[control]))
        return {
            metric: paired_mean_bootstrap(
                {key: scores[candidate][key][metric] for key in shared},
                {key: scores[control][key][metric] for key in shared},
                seed=int(config["experiment"]["seed"]) + seed_offset,
                samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
                direction=f"{candidate}_minus_{control}",
            )
            for metric in ("alignment", "coherence")
        }

    contrasts = {
        "target_minus_no_intervention": contrast("MB_full_target", "MB_no_intervention", 200),
        "random_minus_no_intervention": contrast("MB_full_random", "MB_no_intervention", 201),
        "target_minus_random": contrast("MB_full_target", "MB_full_random", 202),
    }
    initial_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
    initial_summary_path = initial_dir / "summary.json"
    initial_report = json.loads(initial_summary_path.read_text())
    medical_effect = float(
        initial_report["issue19_causal"]["paired_contrasts"]["target_minus_no_intervention"]["alignment"]["difference"]
    )
    broad_effect = float(contrasts["target_minus_no_intervention"]["alignment"]["difference"])
    effect_ratio = math.inf if broad_effect == 0 else abs(medical_effect / broad_effect)
    locality_thresholds = section["causal_gate"]["locality"]
    locality_gate = {
        "medical_alignment_effect": medical_effect,
        "broad_alignment_effect": broad_effect,
        "absolute_broad_alignment_effect": abs(broad_effect),
        "medical_to_broad_absolute_effect_ratio": effect_ratio,
        "thresholds": locality_thresholds,
        "passes_indicative_thresholds": (
            abs(broad_effect) <= float(locality_thresholds["maximum_absolute_broad_alignment_change"])
            and effect_ratio >= float(locality_thresholds["minimum_medical_to_broad_effect_ratio"])
        ),
    }
    report["issue19_locality"].update(
        {
            "condition_metrics": metrics,
            "paired_contrasts": contrasts,
            "alignment_sentinel_counts": sentinels,
            "locality_gate": locality_gate,
        }
    )
    report["status"] = "scored"
    write_json_atomic(summary_path, report)
    initial_report["issue19_causal"]["broad_locality"] = {
        "output_dir": str(output_dir.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "locality_gate": locality_gate,
    }
    initial_report["status"] = "causal_gate_complete"
    write_json_atomic(initial_summary_path, initial_report)
    return report


def summarize_final_broad(config_path: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    section = config["issue19_local_vs_global"]
    output_dir = final_broad_output_dir(root, config)
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    baseline = validate_reused_final_broad_baseline(root, config)
    baseline_dir = ensure_within_workspace(root / str(baseline["run_dir"]))
    baseline_generations = [
        {**row, "condition": "MB_no_intervention", "teacher_condition": "MB_no_intervention"}
        for row in read_jsonl(baseline_dir / "alignment_generations.jsonl")
        if row["condition"] == baseline["condition"]
    ]
    baseline_observations = {str(row["observation_id"]) for row in baseline_generations}
    baseline_judgments = [
        row
        for row in read_jsonl(baseline_dir / "judgments.jsonl")
        if str(row["observation_id"]) in baseline_observations
    ]
    generations = baseline_generations + generations
    judgments = baseline_judgments + judgments
    rows, _ = final_broad_inputs(root, config)
    samples = int(section["data"]["final_broad"]["samples_per_prompt"])
    expected = len(rows) * samples
    conditions = {name for name, _ in ARMS}
    scores, sentinels = checked_numeric_scores(
        generations,
        judgments,
        conditions=conditions,
        expected_per_condition=expected,
    )
    alignment_cutoff = float(config["judge"]["thresholds"]["alignment_score_below"])
    coherence_cutoff = float(config["judge"]["thresholds"]["coherence_score_above"])
    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    identity_task = {f"{row['source_id']}:sample:{row['sample_index']}": str(row["task"]) for row in generations}

    def condition_metrics(condition: str, task: str | None = None) -> dict[str, Any]:
        condition_generations = [
            row for row in generations if row["condition"] == condition and (task is None or row["task"] == task)
        ]
        identities = {f"{row['source_id']}:sample:{row['sample_index']}" for row in condition_generations}
        values = {key: value for key, value in scores[condition].items() if key in identities}
        lengths = [int(row["completion_tokens"]) for row in condition_generations]
        result = {
            "generations": len(condition_generations),
            "numeric_pairs": len(values),
            "alignment_score_mean": fmean(row["alignment"] for row in values.values()),
            "coherence_score_mean": fmean(row["coherence"] for row in values.values()),
            "coherence_above_50_rate": fmean(row["coherence"] > coherence_cutoff for row in values.values()),
            "thresholded_EM_rate": fmean(
                row["alignment"] < alignment_cutoff and row["coherence"] > coherence_cutoff for row in values.values()
            ),
            "mean_completion_tokens": fmean(lengths),
            "median_completion_tokens": median(lengths),
            "truncation_rate": fmean(bool(row["truncated"]) for row in condition_generations),
        }
        if task is None:
            result["refusal_count"] = int(sentinels[condition].get("REFUSAL", 0))
        return result

    def contrast(candidate: str, control: str, seed_offset: int, task: str | None = None) -> dict[str, Any]:
        shared = sorted(set(scores[candidate]) & set(scores[control]))
        if task is not None:
            shared = [key for key in shared if identity_task[key] == task]
        return {
            metric: paired_mean_bootstrap(
                {key: scores[candidate][key][metric] for key in shared},
                {key: scores[control][key][metric] for key in shared},
                seed=seed + seed_offset,
                samples=bootstrap_samples,
                direction=f"{candidate}_minus_{control}",
            )
            for metric in ("alignment", "coherence")
        }

    def contrasts(seed_offset: int, task: str | None = None) -> dict[str, Any]:
        return {
            "target_minus_no_intervention": contrast("MB_full_target", "MB_no_intervention", seed_offset, task),
            "random_minus_no_intervention": contrast("MB_full_random", "MB_no_intervention", seed_offset + 1, task),
            "target_minus_random": contrast("MB_full_target", "MB_full_random", seed_offset + 2, task),
        }

    task_strata = {}
    for index, task in enumerate(("advice", "critique", "summarization", "tutor")):
        expected_task_generations = 60 * samples
        task_metrics = {condition: condition_metrics(condition, task) for condition, _ in ARMS}
        if any(metrics["generations"] != expected_task_generations for metrics in task_metrics.values()):
            raise RuntimeError(f"Issue 19 final Broad {task} generation coverage is incomplete")
        task_strata[task] = {
            "condition_metrics": task_metrics,
            "paired_contrasts": contrasts(320 + 10 * index, task),
        }

    overall_contrasts = contrasts(300)
    report["issue19_final_broad_route"].update(
        {
            "condition_metrics": {condition: condition_metrics(condition) for condition, _ in ARMS},
            "paired_contrasts": overall_contrasts,
            "task_strata": task_strata,
            "alignment_sentinel_counts": sentinels,
            "reused_no_intervention": baseline,
            "engine_note": (
                "No-intervention reuses the exact Issue 17 vLLM artifact; target and random use hooked HF. "
                "Target-minus-random is the engine-matched direction-specific contrast."
            ),
            "decision_use": (
                "One-time scope confirmation for the frozen route. These outcomes may not retune "
                "rank, layer, operation, or intervention strength."
            ),
        }
    )
    report["status"] = "scored"
    write_json_atomic(summary_path, report)

    initial_dir = ensure_within_workspace(root / str(section["causal_gate"]["output_dir"]))
    initial_summary_path = initial_dir / "summary.json"
    initial_report = json.loads(initial_summary_path.read_text())
    initial_report["issue19_causal"]["final_broad_route_confirmation"] = {
        "output_dir": str(output_dir.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "target_minus_no_intervention_alignment": overall_contrasts["target_minus_no_intervention"]["alignment"],
        "target_minus_random_alignment": overall_contrasts["target_minus_random"]["alignment"],
    }
    initial_report["status"] = "final_route_scope_confirmation_complete"
    write_json_atomic(initial_summary_path, initial_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "generate",
            "summarize",
            "generate-specificity",
            "summarize-specificity",
            "summarize-stability",
            "generate-locality",
            "summarize-locality",
            "generate-final-broad",
            "summarize-final-broad",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--section", default="issue19_local_vs_global")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    if args.command in {"generate", "generate-specificity", "generate-locality", "generate-final-broad"}:
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise RuntimeError("Issue 19 causal generation requires elevated guarded GPU execution")
        if args.command == "generate":
            result = generate(config_path, args.batch_size, section_name=args.section)
        elif args.command == "generate-specificity":
            result = generate_specificity(config_path, args.batch_size)
        elif args.command == "generate-final-broad":
            result = generate_final_broad(config_path, args.batch_size)
        else:
            result = generate_locality(config_path, args.batch_size)
    elif args.command == "summarize":
        result = summarize(config_path, section_name=args.section)
    elif args.command == "summarize-specificity":
        result = summarize_specificity(config_path)
    elif args.command == "summarize-stability":
        result = summarize_stability(config_path, section_name=args.section)
    elif args.command == "summarize-final-broad":
        result = summarize_final_broad(config_path)
    else:
        result = summarize_locality(config_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
