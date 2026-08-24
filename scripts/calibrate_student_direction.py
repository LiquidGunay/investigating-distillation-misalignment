#!/usr/bin/env python3
"""Generate, select, and freeze the common 2B causal intervention direction."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from evaluate_teacher_sources import (
    apply_steering,
    generate_hf_batches,
    prepare_requests,
    stage_rows,
    write_outputs,
)
from fit_student_direction import (
    _indexed_rows,
    _read_tensor_state,
    _write_tensor_state,
    load_student,
    paired_residual_means,
)

from inheritance.config import (
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    require_active_guard,
)
from inheritance.direction_selection import select_causal_ablation, select_causal_direction
from inheritance.interventions import (
    orthonormal_basis,
    project_out,
    random_orthogonal_directions,
    select_energy_matched_random_direction,
)
from inheritance.models import cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def _read_object(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _validated_fit(config: dict[str, Any], fit_dir: Path, spec_hash: str) -> tuple[dict[str, Any], Path]:
    fit_dir = ensure_within_workspace(fit_dir)
    fit_path = fit_dir / "fit.json"
    vector_path = fit_dir / "directions.safetensors"
    fit = _read_object(fit_path)
    student = config["models"]["student"]
    if (
        fit.get("resolved_spec_sha256") != spec_hash
        or fit.get("model_id") != student["id"]
        or fit.get("model_revision") != student["revision"]
        or fit.get("directions", {}).get("sha256") != sha256_file(vector_path)
    ):
        raise RuntimeError("student direction fit differs from the current resolved experiment or tensor bytes")
    expected_retained = int(config["teachers"]["steering"]["ranked_layers_retained"])
    retained = fit.get("retained_layers")
    if not isinstance(retained, list) or len(retained) != expected_retained:
        raise RuntimeError("student direction fit has the wrong retained-layer inventory")
    return fit, vector_path


def _load_clean_student(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    student = config["models"]["student"]
    snapshot = cached_model_snapshot(str(student["id"]), str(student["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        padding_side="left",
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation=str(config["teachers"]["steering"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.requires_grad_(False)
    model.eval()
    layout = discover_model_layout(model, expected_layers=24, expected_hidden_size=2048)
    return model, tokenizer, layout


def generate(
    config_path: Path,
    fit_dir: Path,
    output_dir: Path,
    *,
    limit: int | None,
    batch_size: int,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fit, vector_path = _validated_fit(config, fit_dir, str(spec["resolved_spec_sha256"]))
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    _, advice_rows, _, advice_split = stage_rows(root, "calibration", limit)
    profile = config["generation"]["alignment_eval_development"]
    retained_layers = [int(value) for value in fit["retained_layers"]]
    alphas = [float(value) for value in config["teachers"]["steering"]["alpha_sigma_candidates"]]
    model, tokenizer, layout = _load_clean_student(config)
    directions = load_file(vector_path, device="cpu")
    base_prepared, _ = prepare_requests(
        tokenizer,
        spec,
        config,
        "steering_zero",
        "alignment",
        advice_rows,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split=advice_split,
    )
    generations = generate_hf_batches(
        model,
        tokenizer,
        base_prepared,
        profile=profile,
        samples=1,
        batch_size=batch_size,
        condition="steering_zero",
        kind="alignment",
        spec_hash=str(spec["resolved_spec_sha256"]),
    )
    layer_by_index = {int(row["layer"]): row for row in fit["layers"]}
    blocks = getattr(layout.text_model, layout.block_list_name)
    for layer in retained_layers:
        vector = directions[f"layer_{layer:02d}"]
        sigma = float(layer_by_index[layer]["aligned_projection_sigma"])
        if sigma <= 0:
            raise RuntimeError(f"retained student direction layer {layer} has non-positive aligned sigma")
        for alpha in alphas:
            label = format(alpha, "g").replace(".", "p")
            condition = f"steering_bad_l{layer}_alpha{label}"
            prepared, _ = prepare_requests(
                tokenizer,
                spec,
                config,
                condition,
                "alignment",
                advice_rows,
                prompt_cap=int(profile["max_prompt_tokens"]),
                dataset_split=advice_split,
            )
            with apply_steering(model, blocks[layer], vector, alpha * sigma):
                generations.extend(
                    generate_hf_batches(
                        model,
                        tokenizer,
                        prepared,
                        profile=profile,
                        samples=1,
                        batch_size=batch_size,
                        condition=condition,
                        kind="alignment",
                        spec_hash=str(spec["resolved_spec_sha256"]),
                    )
                )
    source_by_id = {str(row["source_id"]): row for row in advice_rows}
    report = write_outputs(
        output_dir,
        config,
        spec,
        "calibration",
        generations,
        source_by_id,
        model_role="student",
        model_config_key="student",
    )
    report["student_direction_calibration"] = {
        "fit": {
            "path": str((fit_dir / "fit.json").relative_to(root)),
            "sha256": sha256_file(fit_dir / "fit.json"),
            "directions_path": str(vector_path.relative_to(root)),
            "directions_sha256": sha256_file(vector_path),
        },
        "retained_layers": retained_layers,
        "alpha_sigma_candidates": alphas,
        "zero_control_generated_once": True,
        "selection_surface": advice_split,
        "engine": "transformers_sdpa",
        "batch_size": batch_size,
        "engineering_limit": limit,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def select(config_path: Path, fit_dir: Path, output_dir: Path) -> dict[str, Any]:
    config_path = ensure_within_workspace(config_path)
    output_dir = ensure_within_workspace(output_dir)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fit, _ = _validated_fit(config, fit_dir, str(spec["resolved_spec_sha256"]))
    summary = _read_object(output_dir / "summary.json")
    if summary.get("status") != "scored":
        raise RuntimeError("student causal direction sweep must be completely judged and summarized before selection")
    calibration = summary.get("student_direction_calibration")
    if not isinstance(calibration, dict) or calibration.get("engineering_limit") is not None:
        raise RuntimeError("an engineering-limited direction sweep cannot be frozen")
    result = select_causal_direction(
        read_jsonl(output_dir / "alignment_generations.jsonl"),
        read_jsonl(output_dir / "judgments.jsonl"),
        retained_layers=[int(value) for value in fit["retained_layers"]],
        alpha_sigma_candidates=[
            float(value) for value in config["teachers"]["steering"]["alpha_sigma_candidates"]
        ],
        minimum_coherence_guardrail_rate=float(
            config["selection_rules"]["teacher_eligibility"]["minimum_coherence_guardrail_pass_fraction"]
        ),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        seed=int(config["experiment"]["seed"]),
        bootstrap_samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
    )
    result.update(
        {
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "fit_sha256": sha256_file(fit_dir / "fit.json"),
            "generation_summary_sha256": sha256_file(output_dir / "summary.json"),
            "generations_sha256": sha256_file(output_dir / "alignment_generations.jsonl"),
            "judgments_sha256": sha256_file(output_dir / "judgments.jsonl"),
        }
    )
    path = output_dir / "selection.json"
    if path.is_file() and _read_object(path) != result:
        raise RuntimeError("existing student causal direction selection differs")
    write_json_atomic(path, result)
    return result


@contextlib.contextmanager
def _apply_generation_ablation(model: Any, block: Any, direction: Any):
    import torch

    basis = orthonormal_basis(direction).to(device=model.device, dtype=model.dtype)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("student ablation hook expected a [batch, sequence, hidden] residual stream")
        changed = hidden.clone()
        changed[:, -1, :] = project_out(changed[:, -1, :], basis)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _ablation_prepared(
    tokenizer: Any,
    spec: dict[str, Any],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    split: str,
    condition: str,
) -> list[dict[str, Any]]:
    prepared, _ = prepare_requests(
        tokenizer,
        spec,
        config,
        "steering_zero",
        "alignment",
        rows,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split=split,
    )
    return [{**row, "condition": condition} for row in prepared]


def generate_ablation(
    config_path: Path,
    fit_dir: Path,
    causal_output_dir: Path,
    output_dir: Path,
    training_run_dir: Path,
    checkpoint_dir: Path,
    *,
    limit: int | None,
    batch_size: int,
) -> dict[str, Any]:
    from inheritance.audit_runner import (
        _checkpoint_step,
        _load_student_checkpoint,
        _validate_completed_checkpoint_lineage,
    )
    from inheritance.student_eval import _student_adapter_state_sha256, _validate_adapter_config

    config_path = ensure_within_workspace(config_path)
    fit_dir = ensure_within_workspace(fit_dir)
    causal_output_dir = ensure_within_workspace(causal_output_dir)
    output_dir = ensure_within_workspace(output_dir)
    training_run_dir = ensure_within_workspace(training_run_dir)
    checkpoint_dir = ensure_within_workspace(checkpoint_dir)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fit, vector_path = _validated_fit(config, fit_dir, str(spec["resolved_spec_sha256"]))
    selection = _read_object(causal_output_dir / "selection.json")
    if selection.get("status") != "passed" or selection.get("fit_sha256") != sha256_file(fit_dir / "fit.json"):
        raise RuntimeError("student ablation requires one passing causal direction selection")
    run_contract = _read_object(training_run_dir / "run_contract.json")
    final_step = int(run_contract.get("schedule", {}).get("total_optimizer_steps", 0))
    step = _checkpoint_step(checkpoint_dir, final_step=final_step)
    experiment = load_experiment_config(config_path)
    _validate_adapter_config(checkpoint_dir, experiment)
    checkpoint_id = (
        f"adapter-sha256:{_student_adapter_state_sha256(checkpoint_dir / 'adapter_model.safetensors')}:step:{step}"
    )
    _validate_completed_checkpoint_lineage(
        training_run_dir,
        checkpoint_dir,
        run_contract,
        optimizer_step=step,
        checkpoint_id=checkpoint_id,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    student, tokenizer, layout, loaded_checkpoint_id = _load_student_checkpoint(
        experiment,
        checkpoint_dir,
        output_dir,
        optimizer_step=step,
    )
    if loaded_checkpoint_id != checkpoint_id:
        raise RuntimeError("student checkpoint identity changed during ablation setup")
    student.requires_grad_(False)
    student.eval()
    directions, _ = _read_tensor_state(vector_path, str(fit["contract_sha256"]))
    selected_layer = int(selection["selected"]["layer"])
    direction = directions[f"layer_{selected_layer:02d}"]
    blocks = getattr(layout.text_model, layout.block_list_name)
    _, advice_rows, _, advice_split = stage_rows(repository_root(), "calibration", limit)
    profile = config["generation"]["alignment_eval_development"]
    conditions = ("student_bad_no_ablation", "student_bad_em_ablation")
    generations = []
    for condition in conditions:
        prepared = _ablation_prepared(
            tokenizer,
            spec,
            config,
            advice_rows,
            profile,
            advice_split,
            condition,
        )
        intervention = (
            contextlib.nullcontext()
            if condition == "student_bad_no_ablation"
            else _apply_generation_ablation(student, blocks[selected_layer], direction)
        )
        with intervention:
            generations.extend(
                generate_hf_batches(
                    student,
                    tokenizer,
                    prepared,
                    profile=profile,
                    samples=1,
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
        "calibration",
        generations,
        {str(row["source_id"]): row for row in advice_rows},
        model_role="student",
        model_config_key="student",
        checkpoint_id=checkpoint_id,
    )
    report["student_direction_ablation"] = {
        "causal_selection_path": str((causal_output_dir / "selection.json").relative_to(repository_root())),
        "causal_selection_sha256": sha256_file(causal_output_dir / "selection.json"),
        "training_run_contract_sha256": sha256_file(training_run_dir / "run_contract.json"),
        "student_checkpoint_id": checkpoint_id,
        "selected_layer": selected_layer,
        "direction_sha256": sha256_file(vector_path),
        "application": "last prompt position and every generated token position",
        "engineering_limit": limit,
        "batch_size": batch_size,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def select_ablation(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = ensure_within_workspace(config_path)
    output_dir = ensure_within_workspace(output_dir)
    config = load_yaml(config_path)
    summary = _read_object(output_dir / "summary.json")
    if summary.get("status") != "scored":
        raise RuntimeError("student direction ablation must be completely judged and summarized")
    ablation = summary.get("student_direction_ablation")
    if not isinstance(ablation, dict) or ablation.get("engineering_limit") is not None:
        raise RuntimeError("an engineering-limited ablation cannot confirm the student direction")
    result = select_causal_ablation(
        read_jsonl(output_dir / "alignment_generations.jsonl"),
        read_jsonl(output_dir / "judgments.jsonl"),
        minimum_coherence_guardrail_rate=float(
            config["selection_rules"]["teacher_eligibility"]["minimum_coherence_guardrail_pass_fraction"]
        ),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        seed=int(config["experiment"]["seed"]),
        bootstrap_samples=int(config["evaluation"]["metrics"]["paired_bootstrap_samples"]),
    )
    result.update(
        {
            "causal_selection_sha256": ablation["causal_selection_sha256"],
            "student_checkpoint_id": ablation["student_checkpoint_id"],
            "generation_summary_sha256": sha256_file(output_dir / "summary.json"),
            "generations_sha256": sha256_file(output_dir / "alignment_generations.jsonl"),
            "judgments_sha256": sha256_file(output_dir / "judgments.jsonl"),
        }
    )
    path = output_dir / "ablation_selection.json"
    if path.is_file() and _read_object(path) != result:
        raise RuntimeError("existing student direction ablation selection differs")
    write_json_atomic(path, result)
    return result


def _aligned_activation_means(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    *,
    layer: int,
) -> Any:
    import torch

    activations = []
    for index, row in enumerate(rows):
        _, aligned, _, _ = paired_residual_means(
            model,
            tokenizer,
            layout,
            question=str(row["question"]),
            bad_answer=str(row["misaligned_answer"]),
            aligned_answer=str(row["aligned_answer"]),
        )
        activations.append(aligned[layer])
        if (index + 1) % 32 == 0 or index + 1 == len(rows):
            print(f"control activation means {index + 1}/{len(rows)}", flush=True)
    return torch.stack(activations)


def freeze(
    config_path: Path,
    fit_dir: Path,
    output_dir: Path,
    ablation_output_dir: Path,
    card_path: Path,
) -> dict[str, Any]:
    import torch

    config_path = ensure_within_workspace(config_path)
    output_dir = ensure_within_workspace(output_dir)
    ablation_output_dir = ensure_within_workspace(ablation_output_dir)
    card_path = ensure_within_workspace(card_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fit, vector_path = _validated_fit(config, fit_dir, str(spec["resolved_spec_sha256"]))
    selection_path = output_dir / "selection.json"
    selection = _read_object(selection_path)
    if selection.get("status") != "passed" or not isinstance(selection.get("selected"), dict):
        raise RuntimeError("student direction cannot be frozen without a passing causal selection")
    if (
        selection.get("resolved_spec_sha256") != spec["resolved_spec_sha256"]
        or selection.get("fit_sha256") != sha256_file(fit_dir / "fit.json")
    ):
        raise RuntimeError("student direction selection differs from the fit or resolved experiment")
    ablation_path = ablation_output_dir / "ablation_selection.json"
    ablation = _read_object(ablation_path)
    if (
        ablation.get("status") != "passed"
        or ablation.get("causal_selection_sha256") != sha256_file(selection_path)
    ):
        raise RuntimeError("student direction cannot be frozen without a passing checkpoint ablation")
    selected_layer = int(selection["selected"]["layer"])
    wrong_layer = int(selection["wrong_layer"])
    all_directions, _ = _read_tensor_state(vector_path, str(fit["contract_sha256"]))
    em = all_directions[f"layer_{selected_layer:02d}"].float()
    wrong = all_directions[f"layer_{wrong_layer:02d}"].float()
    model, tokenizer, layout = load_student(config)
    manifest_name = str(config["teachers"]["steering"]["selection_manifest"])
    selection_rows, manifest_record = _indexed_rows(config, manifest_name)
    activations = _aligned_activation_means(
        model,
        tokenizer,
        layout,
        selection_rows,
        layer=selected_layer,
    )
    controls = config["teachers"]["steering"]["controls"]
    matched_config = controls["steering_random_energy_matched"]
    candidates = random_orthogonal_directions(
        em,
        count=int(matched_config["candidates"]),
        seed=int(matched_config["seed"]),
    )
    random_unit = candidates[0]
    random_matched, energy_selection = select_energy_matched_random_direction(
        activations,
        em,
        candidates[1:],
        maximum_absolute_cosine=float(matched_config["maximum_absolute_cosine_to_bad"]),
    )
    energy_selection["candidate_index"] = int(energy_selection["candidate_index"]) + 1
    tensors = {
        "em": em,
        "random_unit": random_unit,
        "random_energy_matched": random_matched,
        "wrong_layer": wrong,
    }
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()) or not torch.isclose(value.norm(), torch.tensor(1.0), atol=1e-6):
            raise RuntimeError(f"frozen student direction {name} is not a finite unit vector")
    tensor_path = card_path.with_suffix(".safetensors")
    freeze_contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "fit_sha256": sha256_file(fit_dir / "fit.json"),
        "directions_sha256": sha256_file(vector_path),
        "causal_selection_sha256": sha256_file(selection_path),
        "causal_ablation_sha256": sha256_file(ablation_path),
        "causal_ablation_student_checkpoint_id": ablation["student_checkpoint_id"],
        "activation_manifest": manifest_record,
        "control_config": matched_config,
    }
    freeze_contract_sha256 = sha256_json(freeze_contract)
    if tensor_path.is_file():
        existing, _ = _read_tensor_state(tensor_path, freeze_contract_sha256)
        differs = set(existing) != set(tensors) or any(
            not torch.equal(existing[name], value) for name, value in tensors.items()
        )
        if differs:
            raise RuntimeError("existing frozen student direction tensors differ")
    else:
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tensor_state(
            tensor_path,
            tensors,
            {"contract_sha256": freeze_contract_sha256},
        )
    root = repository_root()
    tensor_record = {
        "tensor_path": str(tensor_path.relative_to(root)),
        "tensor_sha256": sha256_file(tensor_path),
    }
    card = {
        "schema_version": 1,
        "status": "frozen",
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["student"]["id"],
        "model_revision": config["models"]["student"]["revision"],
        "tensor_names": list(tensors),
        "freeze_contract": freeze_contract,
        "freeze_contract_sha256": freeze_contract_sha256,
        "directions": {
            "em": {
                **tensor_record,
                "tensor_name": "em",
                "layer": selected_layer,
                "selection": {
                    "causal_steering": selection["selected"],
                    "causal_ablation": ablation,
                },
            },
            "random_unit": {
                **tensor_record,
                "tensor_name": "random_unit",
                "layer": selected_layer,
                "selection": {
                    "seed": int(matched_config["seed"]),
                    "candidate_index": 0,
                    "orthogonal_to": "em",
                },
            },
            "random_energy_matched": {
                **tensor_record,
                "tensor_name": "random_energy_matched",
                "layer": selected_layer,
                "selection": energy_selection,
            },
            "wrong_layer": {
                **tensor_record,
                "tensor_name": "wrong_layer",
                "layer": wrong_layer,
                "selection": {
                    "rule": selection["wrong_layer_rule"],
                    "causal_sweep_layer": wrong_layer,
                },
            },
        },
    }
    if card_path.is_file() and _read_object(card_path) != card:
        raise RuntimeError("existing frozen student direction card differs")
    write_json_atomic(card_path, card)
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("generate", "select", "ablate-generate", "ablate-select", "freeze"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--fit-dir", type=Path, default=Path("outputs/runs/student_direction_fit_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/student_direction_calibration_v1"))
    parser.add_argument(
        "--ablation-output-dir",
        type=Path,
        default=Path("outputs/runs/student_direction_ablation_v1"),
    )
    parser.add_argument("--training-run-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--card-path", type=Path, default=Path("artifacts/directions/student_em_v1.json"))
    parser.add_argument("--limit", type=int, help="bounded engineering generation only")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    guard = require_active_guard()
    if args.phase in {"generate", "ablate-generate", "freeze"} and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise RuntimeError("student direction generation/freezing requires elevated guarded GPU execution")
    config_path = ensure_within_workspace(args.config)
    fit_dir = ensure_within_workspace(args.fit_dir)
    output_dir = ensure_within_workspace(args.output_dir)
    ablation_output_dir = ensure_within_workspace(args.ablation_output_dir)
    if args.phase == "generate":
        report = generate(config_path, fit_dir, output_dir, limit=args.limit, batch_size=args.batch_size)
    elif args.phase == "select":
        report = select(config_path, fit_dir, output_dir)
    elif args.phase == "ablate-generate":
        if args.training_run_dir is None or args.checkpoint_dir is None:
            raise RuntimeError("ablate-generate requires --training-run-dir and --checkpoint-dir")
        report = generate_ablation(
            config_path,
            fit_dir,
            output_dir,
            ablation_output_dir,
            ensure_within_workspace(args.training_run_dir),
            ensure_within_workspace(args.checkpoint_dir),
            limit=args.limit,
            batch_size=args.batch_size,
        )
    elif args.phase == "ablate-select":
        report = select_ablation(config_path, ablation_output_dir)
    else:
        report = freeze(
            config_path,
            fit_dir,
            output_dir,
            ablation_output_dir,
            ensure_within_workspace(args.card_path),
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
