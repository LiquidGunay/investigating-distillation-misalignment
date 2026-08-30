#!/usr/bin/env python3
"""Fit the single Issue 15 rank-4 PCA fallback from saved insecure-code deltas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fit_teacher_model_delta import _write_tensor_state

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def pca_mean_direction(delta: Any, rank: int) -> tuple[Any, Any, Any, dict[str, float]]:
    import torch

    delta = torch.as_tensor(delta, dtype=torch.float32)
    if delta.ndim != 2 or rank < 1 or rank > min(delta.shape):
        raise ValueError("PCA deltas must have [examples, hidden] shape and a feasible rank")
    _, singular_values, right = torch.linalg.svd(delta, full_matrices=False)
    basis = right[:rank]
    raw_mean = delta.mean(dim=0)
    projected_mean = (raw_mean @ basis.T) @ basis
    norm = projected_mean.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("PCA-projected insecure-code mean delta is non-finite or zero")
    direction = projected_mean / norm
    raw_norm = raw_mean.norm().clamp_min(1e-24)
    statistics = {
        "projected_mean_norm": float(norm),
        "raw_mean_norm": float(raw_norm),
        "projected_to_raw_mean_cosine": float((direction @ raw_mean) / raw_norm),
        "uncentered_delta_energy_explained": float(
            singular_values[:rank].square().sum() / singular_values.square().sum().clamp_min(1e-24)
        ),
    }
    return direction, basis, singular_values[:rank], statistics


def fit(config_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["issue15_causal_broad_direction"]["fallback_insecure_code_delta"]
    rank1_dir = ensure_within_workspace(root / str(assay["fit_output_dir"]))
    state_dir = ensure_within_workspace(root / str(assay["fit_state_dir"]))
    output_dir = ensure_within_workspace(root / str(assay["pca_fit_output_dir"]))
    rank1_report_path = rank1_dir / "fit.json"
    rank1_report = json.loads(rank1_report_path.read_text())
    progress = json.loads((state_dir / "progress.json").read_text())
    if progress.get("status") != "complete" or progress.get("contract_sha256") != rank1_report["contract_sha256"]:
        raise RuntimeError("rank-4 PCA fallback requires the completed rank-1 fit state")
    delta_path = state_dir / "adapter_minus_base.npy"
    base_path = state_dir / "base_residual_means.npy"
    delta_state = np.load(delta_path, mmap_mode="r")
    base_state = np.load(base_path, mmap_mode="r")
    expected_shape = (int(assay["fit_rows"]), 32, 2560)
    if delta_state.shape != expected_shape or base_state.shape != expected_shape:
        raise RuntimeError("insecure-code PCA state arrays have the wrong shape")
    rank = int(assay["pca_rank"])
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "rank1_fit_sha256": sha256_file(rank1_report_path),
        "rank1_contract_sha256": rank1_report["contract_sha256"],
        "delta_state_sha256": sha256_file(delta_path),
        "base_state_sha256": sha256_file(base_path),
        "rows": expected_shape[0],
        "layers": expected_shape[1],
        "hidden_size": expected_shape[2],
        "rank": rank,
        "construction": str(assay["rank4_pca_construction"]),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("insecure-code PCA output belongs to another experiment contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a PCA contract to a non-empty output directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
    report_path = output_dir / "fit.json"
    directions_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("directions", {}).get("sha256") != sha256_file(directions_path):
            raise RuntimeError("completed insecure-code PCA direction bytes changed")
        return report

    tensors = {}
    layers = []
    for layer in range(expected_shape[1]):
        delta = torch.from_numpy(np.array(delta_state[:, layer, :], dtype=np.float32, copy=True)).cuda()
        direction, basis, values, statistics = pca_mean_direction(delta, rank)
        base = torch.from_numpy(np.array(base_state[:, layer, :], dtype=np.float32, copy=True)).cuda()
        scale = base.matmul(direction).std(unbiased=False)
        if not bool(torch.isfinite(scale)) or float(scale) <= 0:
            raise RuntimeError("insecure-code PCA base projection scale is invalid")
        tensors[f"layer_{layer:02d}"] = direction.cpu()
        tensors[f"scale_{layer:02d}"] = scale.cpu()
        tensors[f"basis_{layer:02d}"] = basis.cpu()
        layers.append(
            {
                "layer": layer,
                "singular_values": [float(value) for value in values],
                "base_projection_sigma": float(scale),
                **statistics,
            }
        )
        print(f"fit insecure-code PCA layer {layer + 1}/{expected_shape[1]}", flush=True)
        del delta, base, direction, basis, values, scale
        torch.cuda.empty_cache()
    _write_tensor_state(
        directions_path,
        tensors,
        {"contract_sha256": contract_sha256, "rank": str(rank)},
    )
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "construction": "rank4_pca_single_fallback",
        "fit_rows": expected_shape[0],
        "directions": {"path": str(directions_path.relative_to(root)), "sha256": sha256_file(directions_path)},
        "layers": layers,
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 15 insecure-code PCA fitting requires elevated guarded GPU access")
    print(json.dumps(fit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
