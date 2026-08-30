#!/usr/bin/env python3
"""Fit the Issue 15 insecure-code adapter-minus-base residual direction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fit_teacher_model_delta import _write_tensor_state, encode_batch, load_teacher, model_delta_residual_means

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

STATE_INTERVAL = 32


def resolved_config_ref(config: dict[str, Any], reference: str) -> Any:
    value: Any = config
    for part in reference.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"unknown config reference {reference!r}")
        value = value[part]
    return value


def load_fit_rows(
    root: Path,
    config: dict[str, Any],
    assay: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_value = resolved_config_ref(config, str(assay["fit_manifest_index"]))
    index_path = ensure_within_workspace(root / str(index_value))
    if sha256_file(index_path) != str(assay["fit_manifest_index_sha256"]):
        raise RuntimeError("insecure-code manifest index differs from the Issue 15 contract")
    index = json.loads(index_path.read_text())
    manifest_name = str(assay["fit_manifest"])
    record = index.get("files", {}).get(manifest_name)
    if not isinstance(record, dict):
        raise RuntimeError(f"insecure-code manifest index has no {manifest_name!r} record")
    path = ensure_within_workspace(root / str(record["path"]))
    rows = read_jsonl(path)
    if sha256_file(path) != str(record["sha256"]) or len(rows) != int(record["rows"]):
        raise RuntimeError("insecure-code fit manifest bytes or row count changed")
    if len(rows) != int(assay["fit_rows"]):
        raise RuntimeError("insecure-code fit row count differs from the Issue 15 contract")
    return rows, {"name": manifest_name, **record, "index_path": str(index_path.relative_to(root))}


def rank1_layer_statistics(delta: Any, base: Any) -> tuple[Any, float, dict[str, float]]:
    import torch

    delta = torch.as_tensor(delta, dtype=torch.float32)
    base = torch.as_tensor(base, dtype=torch.float32)
    if delta.ndim != 2 or base.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError("rank-1 statistics require matching [examples, hidden] tensors")
    mean = delta.mean(dim=0)
    norm = mean.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("insecure-code mean delta is non-finite or zero")
    direction = mean / norm
    delta_projection = delta @ direction
    base_projection = base @ direction
    scale = float(base_projection.std(unbiased=False))
    if not scale > 0:
        raise RuntimeError("insecure-code base projection scale is zero")
    delta_norm = delta.norm(dim=-1).clamp_min(1e-12)
    statistics = {
        "fit_difference_norm": float(norm),
        "delta_projection_mean": float(delta_projection.mean()),
        "delta_projection_sigma": float(delta_projection.std(unbiased=False)),
        "base_projection_sigma": scale,
        "delta_signed_cosine_mean": float((delta_projection / delta_norm).mean()),
        "delta_projection_energy_ratio": float(
            (delta_projection.square().sum() / delta.square().sum().clamp_min(1e-24)).sqrt()
        ),
    }
    return direction, scale, statistics


def open_state_arrays(
    state_dir: Path,
    *,
    shape: tuple[int, int, int],
    contract_sha256: str,
) -> tuple[Any, Any, int]:
    import numpy as np

    state_dir.mkdir(parents=True, exist_ok=True)
    delta_path = state_dir / "adapter_minus_base.npy"
    base_path = state_dir / "base_residual_means.npy"
    progress_path = state_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("contract_sha256") != contract_sha256 or tuple(progress.get("shape", ())) != shape:
            raise RuntimeError("insecure-code fit state belongs to another experiment contract")
        delta = np.load(delta_path, mmap_mode="r+")
        base = np.load(base_path, mmap_mode="r+")
        if delta.shape != shape or base.shape != shape or delta.dtype != np.float32 or base.dtype != np.float32:
            raise RuntimeError("insecure-code fit state arrays have the wrong shape or dtype")
        return delta, base, int(progress["next_index"])
    if delta_path.exists() or base_path.exists():
        raise RuntimeError("insecure-code state arrays exist without an atomic progress record")
    delta = np.lib.format.open_memmap(delta_path, mode="w+", dtype=np.float32, shape=shape)
    base = np.lib.format.open_memmap(base_path, mode="w+", dtype=np.float32, shape=shape)
    write_json_atomic(
        progress_path,
        {"contract_sha256": contract_sha256, "shape": list(shape), "next_index": 0, "status": "fitting"},
    )
    return delta, base, 0


def fit(config_path: Path) -> dict[str, Any]:
    import numpy as np

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["issue15_causal_broad_direction"]["fallback_insecure_code_delta"]
    rows, manifest = load_fit_rows(root, config, assay)
    adapter_dir = ensure_within_workspace(root / str(assay["source_adapter"]))
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_model.is_file():
        raise RuntimeError("Issue 15 insecure-code source adapter is incomplete")
    output_dir = ensure_within_workspace(root / str(assay["fit_output_dir"]))
    state_dir = ensure_within_workspace(root / str(assay["fit_state_dir"]))
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "adapter_path": str(adapter_dir.relative_to(root)),
        "adapter_config_sha256": sha256_file(adapter_config),
        "adapter_model_sha256": sha256_file(adapter_model),
        "fit_manifest": manifest,
        "fixed_answer_field": str(assay["fixed_answer_field"]),
        "fixed_sequence_contract": str(assay["fixed_sequence_contract"]),
        "max_sequence_tokens": int(assay["max_sequence_tokens"]),
        "batch_size": int(assay["fit_batch_size"]),
        "residual_stream": str(assay["residual_stream"]),
        "positions": str(assay["positions"]),
        "direction": str(assay["direction"]),
        "injection_scale": str(assay["injection_scale"]),
        "state_dtype": str(assay["fit_state_dtype"]),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("insecure-code fit output belongs to another experiment contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a contract to a non-empty insecure-code fit directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
    report_path = output_dir / "fit.json"
    directions_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("directions", {}).get("sha256") != sha256_file(directions_path):
            raise RuntimeError("completed insecure-code direction bytes changed")
        return report

    model, tokenizer, layout = load_teacher(config, adapter_dir)
    shape = (len(rows), layout.num_text_layers, layout.hidden_size)
    delta_state, base_state, start = open_state_arrays(
        state_dir,
        shape=shape,
        contract_sha256=contract_sha256,
    )
    batch_size = int(assay["fit_batch_size"])
    for offset in range(start, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        encoded = encode_batch(
            tokenizer,
            batch,
            answer_field=str(assay["fixed_answer_field"]),
            max_sequence_tokens=int(assay["max_sequence_tokens"]),
        )
        adapter_means, base_means = model_delta_residual_means(
            model,
            encoded=encoded,
            num_layers=layout.num_text_layers,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        end = offset + len(batch)
        delta_state[offset:end] = (adapter_means - base_means).numpy()
        base_state[offset:end] = base_means.numpy()
        if end % STATE_INTERVAL == 0 or end == len(rows):
            delta_state.flush()
            base_state.flush()
            write_json_atomic(
                state_dir / "progress.json",
                {
                    "contract_sha256": contract_sha256,
                    "shape": list(shape),
                    "next_index": end,
                    "status": "fitting" if end < len(rows) else "pooled",
                },
            )
            print(f"fit insecure-code model deltas {end}/{len(rows)}", flush=True)
    if int(json.loads((state_dir / "progress.json").read_text())["next_index"]) != len(rows):
        raise RuntimeError("insecure-code model-delta fit ended early")

    direction_tensors = {}
    layers = []
    for layer in range(layout.num_text_layers):
        delta = np.array(delta_state[:, layer, :], dtype=np.float32, copy=True)
        base = np.array(base_state[:, layer, :], dtype=np.float32, copy=True)
        direction, scale, statistics = rank1_layer_statistics(delta, base)
        direction_tensors[f"layer_{layer:02d}"] = direction
        direction_tensors[f"scale_{layer:02d}"] = direction.new_tensor(scale)
        layers.append({"layer": layer, **statistics})
    _write_tensor_state(directions_path, direction_tensors, {"contract_sha256": contract_sha256, "rank": "1"})
    write_json_atomic(
        state_dir / "progress.json",
        {"contract_sha256": contract_sha256, "shape": list(shape), "next_index": len(rows), "status": "complete"},
    )
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "construction": "rank1_mean",
        "fit_rows": len(rows),
        "directions": {"path": str(directions_path.relative_to(root)), "sha256": sha256_file(directions_path)},
        "state": {
            "delta_path": str((state_dir / "adapter_minus_base.npy").relative_to(root)),
            "base_path": str((state_dir / "base_residual_means.npy").relative_to(root)),
            "shape": list(shape),
            "dtype": "float32",
            "retained_for_single_pca_fallback": True,
        },
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
        raise RuntimeError("Issue 15 insecure-code direction fitting requires elevated guarded GPU access")
    print(json.dumps(fit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
