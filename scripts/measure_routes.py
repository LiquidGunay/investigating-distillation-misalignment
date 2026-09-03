#!/usr/bin/env python3
"""Measure hooks-off post-training routes on fixed token sequences."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from evaluate import FULL_MEDICAL_ROUTE_CONDITIONS, adapter_path
from fit_route import capture_post_block_outputs, pool_post_block_outputs, wrapped_text_blocks

from inheritance.activations import encode_batch, load_teacher, read_tensor_state, write_tensor_state
from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec

FULL_MEDICAL_ARMS = tuple(FULL_MEDICAL_ROUTE_CONDITIONS)
PROFILE_LABELS = ("final_prompt_predictor", *(f"assistant_predictor_{index}" for index in range(1, 9)))


def section_arms(_section_name: str | None = None) -> tuple[str, ...]:
    return FULL_MEDICAL_ARMS


def fixed_sequences(root: Path, route: dict[str, Any], surface: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = route["fixed_sequences"][surface]
    manifest_path = ensure_within_workspace(root / str(source["prompt_manifest"]))
    if sha256_file(manifest_path) != str(source["prompt_manifest_sha256"]):
        raise RuntimeError(f"route {surface} prompt manifest changed")
    manifest = read_jsonl(manifest_path)
    expected_prompts = int(source["prompts"])
    if len(manifest) != expected_prompts:
        raise RuntimeError(f"route {surface} prompt count differs from config")
    if "response_sides" in source:
        sides = [str(value) for value in source["response_sides"]]
        rows = [
            {
                "sequence_id": f"{row['source_id']}:{side}",
                "source_id": str(row["source_id"]),
                "question": str(row["question"]),
                "answer": str(row[side]),
                "task": str(row["task"]),
                "domain": str(row["domain"]),
                "response_side": side,
                "fixed_generation_id": f"{row['fixed_pair_sha256']}:{side}",
                "truncated": False,
            }
            for row in manifest
            for side in sides
        ]
        expected_sequences = expected_prompts * len(sides)
        if len(rows) != expected_sequences or len({row["sequence_id"] for row in rows}) != expected_sequences:
            raise RuntimeError(f"route {surface} paired fixed sequences are incomplete")
        return rows, {
            **source,
            "observed_prompts": expected_prompts,
            "observed_sequences": len(rows),
            "sequence_order_sha256": sha256_json([(row["sequence_id"], row["fixed_generation_id"]) for row in rows]),
        }
    generations_path = ensure_within_workspace(root / str(source["generations_path"]))
    if source.get("generations_sha256") and sha256_file(generations_path) != source["generations_sha256"]:
        raise RuntimeError(f"{surface} fixed generations changed")
    selected = {
        str(row["source_id"]): row
        for row in read_jsonl(generations_path)
        if str(row["condition"]) == str(source["condition"]) and int(row["sample_index"]) == int(source["sample_index"])
    }
    if len(selected) != expected_prompts:
        raise RuntimeError(f"route {surface} fixed-sequence count differs from config")
    manifest_ids = [str(row["source_id"]) for row in manifest]
    if len(set(manifest_ids)) != expected_prompts or set(selected) != set(manifest_ids):
        raise RuntimeError(f"route {surface} fixed generations do not match its prompt manifest")
    rows = [
        {
            "sequence_id": source_id,
            "source_id": source_id,
            "question": str(selected[source_id]["question"]),
            "answer": str(selected[source_id]["completion"]),
            "task": str(selected[source_id]["task"]),
            "domain": str(selected[source_id]["domain"]),
            "fixed_generation_id": str(selected[source_id]["generation_id"]),
            "completion_tokens": int(selected[source_id]["completion_tokens"]),
            "truncated": bool(selected[source_id]["truncated"]),
        }
        for source_id in manifest_ids
    ]
    return rows, {
        **source,
        "observed_prompts": len(rows),
        "observed_sequences": len(rows),
        "truncated_rows": sum(row["truncated"] for row in rows),
        "sequence_order_sha256": sha256_json([(row["source_id"], row["fixed_generation_id"]) for row in rows]),
    }


def forward_states(
    model: Any,
    blocks: Any,
    encoded: list[tuple[list[int], list[int]]],
    *,
    adapter: str | None,
    selected_layer: int,
    pad_token_id: int,
    storage_dtype: Any,
) -> tuple[Any, Any, Any]:
    import torch

    maximum = max(len(tokens) for tokens, _ in encoded)
    input_ids = torch.full((len(encoded), maximum), pad_token_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    for row, (tokens, row_positions) in enumerate(encoded):
        input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=model.device)
        attention_mask[row, : len(tokens)] = 1
        positions.append(row_positions)

    context = model.disable_adapter() if adapter is None else nullcontext()
    if adapter is not None:
        model.set_adapter(adapter)
    with context, capture_post_block_outputs(blocks) as captured, torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    pooled = pool_post_block_outputs(captured, positions).to(storage_dtype)
    selected = captured[selected_layer]
    if selected is None:
        raise RuntimeError("route selected route layer was not captured")
    profile = torch.zeros(
        (len(encoded), len(PROFILE_LABELS), selected.shape[-1]),
        dtype=storage_dtype,
    )
    profile_mask = torch.zeros((len(encoded), len(PROFILE_LABELS)), dtype=torch.bool)
    for row, row_positions in enumerate(positions):
        chosen = row_positions[: len(PROFILE_LABELS)]
        indices = torch.tensor(chosen, dtype=torch.long, device=selected.device)
        profile[row, : len(chosen)] = selected[row].index_select(0, indices).to(storage_dtype).cpu()
        profile_mask[row, : len(chosen)] = True
    return pooled, profile, profile_mask


def layer_summary(delta: Any, direction: Any, full_random: Any, anchor_random: Any) -> list[dict[str, float]]:
    import torch

    delta = delta.float()
    direction = direction.float()
    projections = torch.einsum("nlh,lh->nl", delta, direction)
    norm_squared = delta.square().sum(dim=-1)
    means = delta.mean(dim=0)
    mean_norms = means.norm(dim=-1).clamp_min(1e-12)
    post_basis = means / mean_norms.unsqueeze(-1)
    cosine = (post_basis * direction).sum(dim=-1)
    full_overlap = (post_basis * full_random.float()).sum(dim=-1).square()
    anchor_overlap = (post_basis * anchor_random.float()).sum(dim=-1).square()
    rows = []
    for layer in range(delta.shape[1]):
        projected_energy = projections[:, layer].square().sum()
        total_energy = norm_squared[:, layer].sum().clamp_min(1e-24)
        rows.append(
            {
                "layer": layer,
                "signed_U_med_movement": float(projections[:, layer].mean()),
                "mean_U_med_magnitude": float(projections[:, layer].abs().mean()),
                "rms_U_med_magnitude": float(projections[:, layer].square().mean().sqrt()),
                "rms_total_delta_magnitude": float(norm_squared[:, layer].mean().sqrt()),
                "fraction_delta_energy_in_U_med": float(projected_energy / total_energy),
                "rms_orthogonal_delta_magnitude": float(
                    (norm_squared[:, layer] - projections[:, layer].square()).clamp_min(0).mean().sqrt()
                ),
                "signed_posttraining_basis_cosine_with_U_med": float(cosine[layer]),
                "principal_angle_degrees_to_U_med": float(
                    torch.rad2deg(torch.acos(cosine[layer].abs().clamp(max=1.0)))
                ),
                "directional_containment_in_U_med": float(cosine[layer].square()),
                "overlap_with_full_random_null": float(full_overlap[layer]),
                "overlap_with_anchor_random_null": float(anchor_overlap[layer]),
            }
        )
    return rows


def profile_summary(delta: Any, mask: Any, direction: Any) -> list[dict[str, float | int | str]]:
    import torch

    delta = delta.float()
    direction = direction.float()
    projections = torch.einsum("nph,h->np", delta, direction)
    norms = delta.norm(dim=-1)
    rows = []
    for index, label in enumerate(PROFILE_LABELS):
        included = mask[:, index]
        values = projections[included, index]
        selected_norms = norms[included, index]
        rows.append(
            {
                "position": label,
                "sequences": int(included.sum()),
                "signed_U_med_movement": float(values.mean()),
                "mean_U_med_magnitude": float(values.abs().mean()),
                "rms_total_delta_magnitude": float(selected_norms.square().mean().sqrt()),
                "fraction_delta_energy_in_U_med": float(
                    values.square().sum() / selected_norms.square().sum().clamp_min(1e-24)
                ),
            }
        )
    return rows


def measure(
    config_path: Path,
    checkpoint: str,
    surface: str,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config["route_blocking"]
    arms = FULL_MEDICAL_ARMS
    route = section["route_analysis"]
    rows, fixed_contract = fixed_sequences(root, route, surface)
    adapters = {}
    for arm in arms:
        path = adapter_path(config, arm, checkpoint)
        adapters[arm] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path / "adapter_model.safetensors"),
        }
    fit_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit = json.loads((fit_dir / "fit.json").read_text())
    subspaces_path = fit_dir / str(fit["artifacts"]["subspaces"]["path"])
    controls_report = json.loads((fit_dir / "random_controls.json").read_text())
    controls_path = fit_dir / str(controls_report["artifact"]["path"])
    if sha256_file(subspaces_path) != fit["artifacts"]["subspaces"]["sha256"]:
        raise RuntimeError("route U_med subspace bytes changed")
    if sha256_file(controls_path) != controls_report["artifact"]["sha256"]:
        raise RuntimeError("route random-control bytes changed")
    selected_layer = int(section["screening"]["frozen_selection"]["layer"])
    selected_rank = int(section["screening"]["frozen_selection"]["rank"])
    if int(route["solution_subspace_rank"]) != selected_rank or selected_rank != 1:
        raise RuntimeError("route post-training route analysis must match the frozen rank-1 route")
    storage_dtype_name = str(route["tensor_storage_dtype"])
    storage_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if storage_dtype_name not in storage_dtypes:
        raise ValueError("route route tensor storage dtype must be bfloat16 or float32")
    storage_dtype = storage_dtypes[storage_dtype_name]
    save_base_pooled = surface == "reroute_fit"
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "checkpoint": checkpoint,
        "surface": surface,
        "fixed_sequences": fixed_contract,
        "adapters": adapters,
        "U_med_sha256": sha256_file(subspaces_path),
        "random_controls_sha256": sha256_file(controls_path),
        "selected_layer": selected_layer,
        "solution_subspace_rank": selected_rank,
        "delta_reference": str(route["delta_reference"]),
        "profile_labels": list(PROFILE_LABELS),
        "maximum_sequence_tokens": int(route["maximum_sequence_tokens"]),
        "batch_size": int(route["extraction_batch_size"]),
        "tensor_storage_dtype": storage_dtype_name,
        **({"save_base_pooled_states": True} if save_base_pooled else {}),
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(root / str(route["output_dir"]) / surface / checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("existing route route extraction belongs to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a route route contract to a non-empty directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_jsonl_atomic(
            output_dir / "sequence_order.jsonl",
            [
                {
                    "sequence_index": index,
                    "sequence_id": row["sequence_id"],
                    "source_id": row["source_id"],
                    "fixed_generation_id": row["fixed_generation_id"],
                    "response_side": row.get("response_side"),
                    "task": row["task"],
                    "domain": row["domain"],
                }
                for index, row in enumerate(rows)
            ],
        )
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        report = json.loads(summary_path.read_text())
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing route route summary belongs to another contract")
        return report

    ordinary_path = root / adapters[arms[0]]["path"]
    model, tokenizer, layout = load_teacher(config, ordinary_path)
    adapter_names = {arms[0]: "default"}
    for arm_index, arm in enumerate(arms[1:], start=1):
        name = f"arm_{arm_index}"
        model.load_adapter(str(root / adapters[arm]["path"]), adapter_name=name, is_trainable=False)
        adapter_names[arm] = name
    blocks = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)
    state_path = output_dir / "route_deltas.safetensors"
    if state_path.is_file():
        tensors, metadata = read_tensor_state(state_path, contract_sha256)
        pooled_deltas = tensors["pooled_deltas"]
        profile_deltas = tensors["profile_deltas"]
        profile_mask = tensors["profile_mask"].bool()
        base_pooled_states = tensors.get("base_pooled_states")
        if save_base_pooled and base_pooled_states is None:
            raise RuntimeError("resumable route reroute fit is missing its base pooled states")
        start = int(metadata["next_index"])
    else:
        pooled_deltas = torch.zeros(
            (len(arms), len(rows), layout.num_text_layers, layout.hidden_size), dtype=storage_dtype
        )
        profile_deltas = torch.zeros(
            (len(arms), len(rows), len(PROFILE_LABELS), layout.hidden_size), dtype=storage_dtype
        )
        profile_mask = torch.zeros((len(rows), len(PROFILE_LABELS)), dtype=torch.bool)
        base_pooled_states = (
            torch.zeros((len(rows), layout.num_text_layers, layout.hidden_size), dtype=storage_dtype)
            if save_base_pooled
            else None
        )
        start = 0
    batch_size = int(route["extraction_batch_size"])
    interval = int(route["resumable_state_interval_sequences"])
    for offset in range(start, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        encoded = encode_batch(
            tokenizer,
            batch,
            answer_field="answer",
            max_sequence_tokens=int(route["maximum_sequence_tokens"]),
        )
        base_pooled, base_profile, batch_mask = forward_states(
            model,
            blocks,
            encoded,
            adapter=None,
            selected_layer=selected_layer,
            pad_token_id=int(tokenizer.pad_token_id),
            storage_dtype=storage_dtype,
        )
        profile_mask[offset : offset + len(batch)] = batch_mask
        if base_pooled_states is not None:
            base_pooled_states[offset : offset + len(batch)] = base_pooled
        for arm_index, arm in enumerate(arms):
            arm_pooled, arm_profile, _ = forward_states(
                model,
                blocks,
                encoded,
                adapter=adapter_names[arm],
                selected_layer=selected_layer,
                pad_token_id=int(tokenizer.pad_token_id),
                storage_dtype=storage_dtype,
            )
            pooled_deltas[arm_index, offset : offset + len(batch)] = arm_pooled - base_pooled
            profile_deltas[arm_index, offset : offset + len(batch)] = arm_profile - base_profile
        stop = offset + len(batch)
        if stop % interval == 0 or stop == len(rows):
            state_tensors = {
                "pooled_deltas": pooled_deltas,
                "profile_deltas": profile_deltas,
                "profile_mask": profile_mask,
            }
            if base_pooled_states is not None:
                state_tensors["base_pooled_states"] = base_pooled_states
            write_tensor_state(
                state_path,
                state_tensors,
                {"contract_sha256": contract_sha256, "next_index": str(stop)},
            )
            print(f"route {surface} route sequences {stop}/{len(rows)}", flush=True)

    subspaces = load_file(subspaces_path, device="cpu")
    controls = load_file(controls_path, device="cpu")
    U_med = subspaces["rank1_basis"].squeeze(-1).float()
    full_random = controls["rank1_full"].squeeze(-1).float()
    anchor_random = controls["rank1_anchor"].squeeze(-1).float()
    derived: dict[str, Any] = {"profile_mask": profile_mask}
    arm_summaries = {}
    for arm_index, arm in enumerate(arms):
        delta = pooled_deltas[arm_index].float()
        projections = torch.einsum("nlh,lh->nl", delta, U_med)
        norm_squared = delta.square().sum(dim=-1)
        derived[f"{arm}_signed_U_med_movement"] = projections
        derived[f"{arm}_U_med_magnitude"] = projections.abs()
        derived[f"{arm}_fraction_delta_energy_in_U_med"] = projections.square() / norm_squared.clamp_min(1e-24)
        derived[f"{arm}_orthogonal_delta_magnitude"] = (norm_squared - projections.square()).clamp_min(0).sqrt()
        mean_delta = delta.mean(dim=0)
        derived[f"{arm}_posttraining_rank1_basis"] = mean_delta / mean_delta.norm(dim=-1).clamp_min(1e-12).unsqueeze(-1)
        profile_delta = profile_deltas[arm_index].float()
        derived[f"{arm}_profile_signed_U_med_movement"] = torch.einsum(
            "nph,h->np", profile_delta, U_med[selected_layer]
        )
        arm_summaries[arm] = {
            "layers": layer_summary(delta, U_med, full_random, anchor_random),
            "selected_layer_token_profile": profile_summary(
                profile_delta,
                profile_mask,
                U_med[selected_layer],
            ),
        }
    derived_path = output_dir / "route_metrics.safetensors"
    save_file(
        {name: value.contiguous() for name, value in derived.items()},
        derived_path,
        metadata={"contract_sha256": contract_sha256},
    )
    report = {
        "schema_version": 1,
        "status": "measured",
        "contract_sha256": contract_sha256,
        "checkpoint": checkpoint,
        "surface": surface,
        "sequences": len(rows),
        "selected_layer": selected_layer,
        "arms": arm_summaries,
        "artifacts": {
            "contract": {"path": contract_path.name, "sha256": sha256_file(contract_path)},
            "sequence_order": {
                "path": "sequence_order.jsonl",
                "sha256": sha256_file(output_dir / "sequence_order.jsonl"),
            },
            "full_pooled_residual_deltas": {"path": state_path.name, "sha256": sha256_file(state_path)},
            "per_example_route_metrics": {"path": derived_path.name, "sha256": sha256_file(derived_path)},
        },
    }
    write_json_atomic(summary_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--surface", choices=("reroute_fit", "medical", "mechanistic_ood"), required=True)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("route post-training route measurement requires elevated guarded GPU execution")
    print(
        json.dumps(
            measure(args.config, "final_adapter", args.surface),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
