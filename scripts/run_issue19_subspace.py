#!/usr/bin/env python3
"""Extract and fit the bounded all-layer Issue 19 medical-policy candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from fit_teacher_model_delta import _pool_hidden_states, _read_tensor_state, _write_tensor_state, encode_batch

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def load_models(config: dict[str, Any], section: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = repository_root()
    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation=str(config["teachers"]["steering"]["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    layout = discover_model_layout(base, expected_layers=32, expected_hidden_size=2560)
    bad = section["models"]["MB"]
    aligned = section["models"]["MA"]
    bad_path = ensure_within_workspace(root / str(bad["adapter_path"]))
    aligned_path = ensure_within_workspace(root / str(aligned["adapter_path"]))
    for name, path, expected in (
        ("MB", bad_path, str(bad["adapter_sha256"])),
        ("MA", aligned_path, str(aligned["adapter_sha256"])),
    ):
        if sha256_file(path / "adapter_model.safetensors") != expected:
            raise RuntimeError(f"Issue 19 {name} adapter bytes differ from config")
    model = PeftModel.from_pretrained(base, bad_path, adapter_name="MB", is_trainable=False)
    model.load_adapter(aligned_path, adapter_name="MA", is_trainable=False)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer, layout


def pooled_forward(
    model: Any,
    encoded: list[tuple[list[int], list[int]]],
    *,
    adapter: str | None,
    num_layers: int,
    pad_token_id: int,
) -> Any:
    import torch

    maximum = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        pad_token_id,
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    for row, (ids, row_positions) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[row, : len(ids)] = 1
        positions.append(row_positions)

    def forward() -> Any:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return _pool_hidden_states(result.hidden_states, positions, num_layers)

    if adapter is None:
        with model.disable_adapter():
            return forward()
    model.set_adapter(adapter)
    return forward()


def sequence_records(rows: list[dict[str, Any]], sides: list[str]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        for side in sides:
            records.append(
                {
                    "source_id": row["source_id"],
                    "fixed_pair_sha256": row["fixed_pair_sha256"],
                    "response_side": side,
                    "question": row["question"],
                    "answer": row[side],
                }
            )
    return records


def fit_layer_candidates(rows: Any) -> dict[str, Any]:
    import torch

    if rows.ndim != 2 or rows.shape[0] < 8 or rows.shape[1] < 4:
        raise ValueError("candidate fitting requires at least eight rows and four hidden dimensions")
    rows = rows.float()
    mean = rows.mean(0)
    mean_norm = mean.norm()
    if not bool(torch.isfinite(mean_norm)) or float(mean_norm) <= 0:
        raise RuntimeError("candidate rows have a zero/non-finite mean model delta")
    rank1 = (mean / mean_norm).unsqueeze(1)
    gram = rows @ rows.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    selected_values = eigenvalues.index_select(0, order[:8]).clamp_min(0)
    selected_vectors = eigenvectors.index_select(1, order[:4])
    singular = selected_values.sqrt()
    if bool((singular[:4] <= 1e-8).any()):
        raise RuntimeError("rank-4 candidate fit is numerically deficient")
    rank4 = rows.T @ selected_vectors / singular[:4].unsqueeze(0)
    rank4, _ = torch.linalg.qr(rank4, mode="reduced")
    projected_mean = rank4 @ (rank4.T @ mean)
    projected_norm = projected_mean.norm()
    if float(projected_norm) <= 0:
        raise RuntimeError("rank-4 harmful orientation is zero")
    total_energy = rows.square().sum(dtype=torch.float64)
    return {
        "mean": mean,
        "rank1": rank1,
        "rank4": rank4,
        "rank4_readout": projected_mean / projected_norm,
        "singular": singular,
        "total_energy": total_energy,
        "rank1_energy": (rows @ rank1).square().sum(dtype=torch.float64),
        "rank4_energy": (rows @ rank4).square().sum(dtype=torch.float64),
    }


def fit_candidates(activations: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    delta = activations["MB"] - activations["MA"]
    if delta.ndim != 3:
        raise RuntimeError("Issue 19 activation tensor must have [sequences, layers, hidden] shape")
    layers = delta.shape[1]
    hidden = delta.shape[2]
    rank1_bases = torch.empty((layers, hidden, 1), dtype=torch.float32)
    rank4_bases = torch.empty((layers, hidden, 4), dtype=torch.float32)
    rank1_readouts = torch.empty((layers, hidden), dtype=torch.float32)
    rank4_readouts = torch.empty((layers, hidden), dtype=torch.float32)
    means = torch.empty((layers, hidden), dtype=torch.float32)
    top_singular_values = torch.empty((layers, 8), dtype=torch.float32)
    total_energies = torch.empty((layers,), dtype=torch.float64)
    layer_rows = []
    for layer in range(layers):
        rows = delta[:, layer].float().cuda()
        fitted = fit_layer_candidates(rows)
        mean = fitted["mean"]
        mean_norm = mean.norm()
        rank1 = fitted["rank1"]
        rank4 = fitted["rank4"]
        singular = fitted["singular"]
        total_energy = fitted["total_energy"]
        mb_m0 = activations["MB"][:, layer] - activations["M0"][:, layer]
        ma_m0 = activations["MA"][:, layer] - activations["M0"][:, layer]
        layer_rows.append(
            {
                "layer": layer,
                "sequences": int(rows.shape[0]),
                "mean_MB_minus_MA_norm": float(mean_norm),
                "mean_MB_minus_M0_norm": float(mb_m0.mean(0).norm()),
                "mean_MA_minus_M0_norm": float(ma_m0.mean(0).norm()),
                "rank1_uncentered_delta_energy_fraction": float(fitted["rank1_energy"] / total_energy),
                "rank4_uncentered_delta_energy_fraction": float(fitted["rank4_energy"] / total_energy),
                "top_singular_values": [float(value) for value in singular],
            }
        )
        rank1_bases[layer] = rank1.cpu()
        rank4_bases[layer] = rank4.cpu()
        rank1_readouts[layer] = rank1.squeeze(1).cpu()
        rank4_readouts[layer] = fitted["rank4_readout"].cpu()
        means[layer] = mean.cpu()
        top_singular_values[layer] = singular.cpu()
        total_energies[layer] = total_energy.cpu()
        print(f"fit Issue 19 candidates layer {layer + 1}/{layers}", flush=True)
    tensors = {
        "mean_MB_minus_MA": means,
        "rank1_basis": rank1_bases,
        "rank1_harmful_readout": rank1_readouts,
        "rank4_basis": rank4_bases,
        "rank4_harmful_readout": rank4_readouts,
        "top8_singular_values": top_singular_values,
        "uncentered_delta_total_energy": total_energies,
    }
    return {"layers": layer_rows}, tensors


def projection_rms(rows: Any, basis: Any) -> Any:
    return (rows.float() @ basis.float()).square().sum(dim=-1).mean().sqrt()


def select_energy_matched_subspace(
    sample_rows: Any,
    match_rows: list[Any],
    target: Any,
    *,
    candidates: int,
    seed: int,
    tolerance: float,
) -> tuple[Any, dict[str, Any]]:
    """Select a behavior-blind covariance control with matched removed RMS energy."""
    import torch

    if sample_rows.ndim != 2 or target.ndim != 2 or sample_rows.shape[1] != target.shape[0]:
        raise ValueError("random-control source and target shapes are incompatible")
    if candidates < 1 or not match_rows:
        raise ValueError("random-control selection requires candidates and matching references")
    sample_rows = sample_rows.float()
    target = target.float()
    rank = target.shape[1]
    centered = sample_rows - sample_rows.mean(0, keepdim=True)
    generator = torch.Generator(device=sample_rows.device).manual_seed(seed)
    coefficients = torch.randn(
        (candidates, centered.shape[0], rank),
        generator=generator,
        dtype=centered.dtype,
        device=centered.device,
    )
    raw = torch.einsum("nh,cnk->chk", centered, coefficients)
    raw = raw - torch.einsum("hk,ckj->chj", target, torch.einsum("hk,chj->ckj", target, raw))
    controls, upper = torch.linalg.qr(raw, mode="reduced")
    valid = torch.diagonal(upper, dim1=-2, dim2=-1).abs().amin(dim=-1) > 1e-6
    target_energies = torch.stack([projection_rms(rows, target) for rows in match_rows])
    if bool((target_energies <= 1e-12).any()):
        raise RuntimeError("target removed RMS energy is zero on a matching reference")
    unscaled_energies = torch.stack(
        [
            torch.einsum("nh,chk->cnk", rows.float(), controls).square().sum(dim=-1).mean(dim=-1).sqrt()
            for rows in match_rows
        ],
        dim=1,
    )
    ratios = unscaled_energies / target_energies.unsqueeze(0)
    scales = 2.0 / (ratios.amin(dim=1) + ratios.amax(dim=1))
    scaled_energies = unscaled_energies * scales.unsqueeze(1)
    relative_mismatches = (scaled_energies - target_energies.unsqueeze(0)).abs() / target_energies
    score = relative_mismatches.amax(dim=1)
    valid &= torch.isfinite(scales) & (scales > 0)
    score[~valid] = torch.inf
    best_score = score.min()
    if not math.isfinite(float(best_score)):
        raise RuntimeError("all covariance random-control candidates were rank deficient")
    near_best = score <= best_score + 1e-6
    tie_break = scales.clone()
    tie_break[~near_best] = torch.inf
    index = int(torch.argmin(tie_break).item())
    selected = controls[index]
    overlap = float((target.T @ selected).square().sum() / rank)
    return selected, {
        "candidate_index": index,
        "seed": seed,
        "candidates": candidates,
        "target_removed_rms": [float(value) for value in target_energies],
        "unscaled_selected_removed_rms": [float(value) for value in unscaled_energies[index]],
        "removal_scale": float(scales[index]),
        "selected_removed_rms": [float(value) for value in scaled_energies[index]],
        "relative_mismatch": [float(value) for value in relative_mismatches[index]],
        "maximum_relative_mismatch": float(score[index]),
        "within_tolerance": bool(score[index] <= tolerance),
        "projector_overlap_with_target": overlap,
    }


def fit_random_controls(config_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    section = config["issue19_local_vs_global"]
    candidate = section["candidate_subspace"]
    random = section["random_controls"]
    output_dir = ensure_within_workspace(root / str(candidate["output_dir"]))
    fit_report_path = output_dir / "fit.json"
    fit_report = json.loads(fit_report_path.read_text())
    for name in ("activations", "subspaces"):
        record = fit_report["artifacts"][name]
        if sha256_file(output_dir / str(record["path"])) != str(record["sha256"]):
            raise RuntimeError(f"Issue 19 {name} bytes differ from the fit report")
    contract = {
        "schema_version": 1,
        "fit_contract_sha256": fit_report["contract_sha256"],
        "fit_report_sha256": sha256_file(fit_report_path),
        "random_controls": random,
    }
    contract_sha256 = sha256_json(contract)
    report_path = output_dir / "random_controls.json"
    tensor_path = output_dir / "random_controls.safetensors"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing Issue 19 random controls belong to another contract")
        if sha256_file(tensor_path) != report["artifact"]["sha256"]:
            raise RuntimeError("existing Issue 19 random-control tensor bytes changed")
        return report

    activations = load_file(output_dir / fit_report["artifacts"]["activations"]["path"], device="cpu")
    subspaces = load_file(output_dir / fit_report["artifacts"]["subspaces"]["path"], device="cpu")
    layers = activations["M0"].shape[1]
    hidden = activations["M0"].shape[2]
    tensors: dict[str, Any] = {}
    records = []
    for rank in (1, 4):
        targets = subspaces[f"rank{rank}_basis"]
        full_controls = torch.empty((layers, hidden, rank), dtype=torch.float32)
        anchor_controls = torch.empty_like(full_controls)
        full_scales = torch.empty((layers,), dtype=torch.float32)
        anchor_scales = torch.empty_like(full_scales)
        for layer in range(layers):
            target = targets[layer].cuda()
            m0 = activations["M0"][:, layer].cuda()
            mb = activations["MB"][:, layer].cuda()
            delta = mb - m0
            full, full_report = select_energy_matched_subspace(
                torch.cat((m0, mb), dim=0),
                [m0, mb],
                target,
                candidates=int(random["candidates_per_layer_rank_operation"]),
                seed=int(random["seed"]) + 1000 * layer + 10 * rank,
                tolerance=float(random["removed_rms_relative_tolerance"]),
            )
            anchor, anchor_report = select_energy_matched_subspace(
                delta,
                [delta],
                target,
                candidates=int(random["candidates_per_layer_rank_operation"]),
                seed=int(random["seed"]) + 1000 * layer + 10 * rank + 1,
                tolerance=float(random["removed_rms_relative_tolerance"]),
            )
            maximum_overlap = float(random["maximum_projector_overlap"])
            if (
                full_report["projector_overlap_with_target"] > maximum_overlap
                or anchor_report["projector_overlap_with_target"] > maximum_overlap
            ):
                raise RuntimeError("Issue 19 random control overlaps its target subspace")
            full_controls[layer] = full.cpu()
            anchor_controls[layer] = anchor.cpu()
            full_scales[layer] = float(full_report["removal_scale"])
            anchor_scales[layer] = float(anchor_report["removal_scale"])
            records.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "full_state": full_report,
                    "anchored_delta": anchor_report,
                }
            )
            print(f"fit Issue 19 random controls rank {rank} layer {layer + 1}/{layers}", flush=True)
        tensors[f"rank{rank}_full"] = full_controls
        tensors[f"rank{rank}_anchor"] = anchor_controls
        tensors[f"rank{rank}_full_scale"] = full_scales
        tensors[f"rank{rank}_anchor_scale"] = anchor_scales
    save_file(tensors, tensor_path, metadata={"contract_sha256": contract_sha256})
    report = {
        "schema_version": 1,
        "status": "controls_fitted",
        "contract_sha256": contract_sha256,
        "fit_contract_sha256": fit_report["contract_sha256"],
        "records": records,
        "all_within_tolerance": all(
            row[operation]["within_tolerance"] for row in records for operation in ("full_state", "anchored_delta")
        ),
        "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path)},
    }
    write_json_atomic(report_path, report)
    return report


def extract_and_fit(config_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config["issue19_local_vs_global"]
    candidate = section["candidate_subspace"]
    fit_contract = section["data"]["heldout_medical"]["splits"]["fit"]
    fit_path = ensure_within_workspace(root / str(fit_contract["manifest"]))
    if sha256_file(fit_path) != str(fit_contract["sha256"]):
        raise RuntimeError("Issue 19 fit manifest differs from config")
    rows = read_jsonl(fit_path)
    if len(rows) != int(fit_contract["rows"]):
        raise RuntimeError("Issue 19 fit manifest row count differs from config")
    sides = [str(value) for value in candidate["response_sides"]]
    records = sequence_records(rows, sides)
    output_dir = ensure_within_workspace(root / str(candidate["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "models": {name: section["models"][name] for name in ("M0", "MA", "MB", "shared_initial_adapter")},
        "fit_manifest": {
            "path": str(fit_path.relative_to(root)),
            "rows": len(rows),
            "sha256": sha256_file(fit_path),
        },
        "sequence_order_sha256": sha256_json(
            [
                {
                    "source_id": row["source_id"],
                    "fixed_pair_sha256": row["fixed_pair_sha256"],
                    "response_side": row["response_side"],
                }
                for row in records
            ]
        ),
        "residual_stream": candidate["residual_stream"],
        "positions": candidate["response_positions"],
        "weighting": candidate["weighting"],
        "maximum_sequence_tokens": int(candidate["maximum_sequence_tokens"]),
        "candidate_ranks": [1, 4],
    }
    contract_sha256 = sha256_json(contract)
    contract_record = {**contract, "contract_sha256": contract_sha256}
    contract_path = output_dir / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("existing Issue 19 subspace output belongs to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach an Issue 19 contract to a non-empty output directory")
    else:
        write_json_atomic(contract_path, contract_record)
    order_path = output_dir / "sequence_order.jsonl"
    order_rows = [
        {
            "sequence_index": index,
            "source_id": row["source_id"],
            "fixed_pair_sha256": row["fixed_pair_sha256"],
            "response_side": row["response_side"],
        }
        for index, row in enumerate(records)
    ]
    if order_path.is_file():
        if read_jsonl(order_path) != order_rows:
            raise RuntimeError("existing Issue 19 sequence order differs")
    else:
        write_jsonl_atomic(order_path, order_rows)

    report_path = output_dir / "fit.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing Issue 19 fit report belongs to another contract")
        return report

    model, tokenizer, layout = load_models(config, section)
    state_path = output_dir / "fit_activations.safetensors"
    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        activations = {name: tensors[name] for name in ("M0", "MA", "MB")}
        start = int(metadata["next_index"])
    else:
        activations = {
            name: torch.zeros((len(records), layout.num_text_layers, layout.hidden_size)) for name in ("M0", "MA", "MB")
        }
        start = 0
    batch_size = int(candidate["extraction_batch_size"])
    interval = int(candidate["resumable_state_interval_sequences"])
    for offset in range(start, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        encoded = encode_batch(
            tokenizer,
            batch,
            answer_field="answer",
            max_sequence_tokens=int(candidate["maximum_sequence_tokens"]),
        )
        for name, adapter in (("MB", "MB"), ("MA", "MA"), ("M0", None)):
            activations[name][offset : offset + len(batch)] = pooled_forward(
                model,
                encoded,
                adapter=adapter,
                num_layers=layout.num_text_layers,
                pad_token_id=int(tokenizer.pad_token_id),
            )
        stop = offset + len(batch)
        if stop % interval == 0 or stop == len(records):
            _write_tensor_state(
                state_path,
                activations,
                {
                    "contract_sha256": contract_sha256,
                    "next_index": str(stop),
                    "phase": "complete" if stop == len(records) else "extract",
                },
            )
            print(f"Issue 19 fixed-token activations {stop}/{len(records)}", flush=True)
    del model
    torch.cuda.empty_cache()
    analysis, candidate_tensors = fit_candidates(activations)
    subspaces_path = output_dir / "subspaces.safetensors"
    save_file(candidate_tensors, subspaces_path, metadata={"contract_sha256": contract_sha256})
    report = {
        "schema_version": 1,
        "status": "candidate_family_fitted",
        "contract_sha256": contract_sha256,
        "rows": len(rows),
        "fixed_sequences": len(records),
        "layers": layout.num_text_layers,
        "hidden_size": layout.hidden_size,
        "artifacts": {
            "contract": {"path": "contract.json", "sha256": sha256_file(contract_path)},
            "sequence_order": {"path": order_path.name, "sha256": sha256_file(order_path)},
            "activations": {"path": state_path.name, "sha256": sha256_file(state_path)},
            "subspaces": {"path": subspaces_path.name, "sha256": sha256_file(subspaces_path)},
        },
        "analysis": analysis,
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stage", choices=("extract", "controls"), default="extract")
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 19 subspace extraction requires elevated guarded GPU execution")
    report = extract_and_fit(args.config) if args.stage == "extract" else fit_random_controls(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
