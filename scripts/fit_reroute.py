#!/usr/bin/env python3
"""Fit the bounded residual reroute candidate on the frozen route-fit split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic

ARMS = (
    "medical_route_full_ordinary",
    "medical_route_full_target",
    "medical_route_full_random",
    "medical_route_anchor_target",
    "medical_route_anchor_random",
)


def equal_prompt_means(rows: Any, sequence_order: list[dict[str, Any]]) -> Any:
    import torch

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(sequence_order):
        groups.setdefault(str(row["source_id"]), []).append(index)
    return torch.stack(
        [rows.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0) for indices in groups.values()]
    )


def residualized_unit_direction(mean: Any, forbidden: Any) -> Any:
    value = mean.float() - forbidden.float() @ (forbidden.float().T @ mean.float())
    norm = value.norm()
    if not bool(norm.isfinite()) or float(norm) <= 1e-8:
        raise RuntimeError("route reroute contrast is degenerate after residualization")
    return value / norm


def matched_random_direction(
    source_rows: Any,
    match_rows: Any,
    reroute: Any,
    U_med: Any,
    *,
    candidates: int,
    seed: int,
) -> tuple[Any, float, dict[str, float]]:
    import torch

    source = source_rows.float() - source_rows.float().mean(dim=0, keepdim=True)
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn((candidates, source.shape[0]), generator=generator)
    values = coefficients @ source
    forbidden = torch.stack((reroute.float(), U_med.float()), dim=1)
    values = values - (values @ forbidden) @ forbidden.T
    norms = values.norm(dim=1)
    valid = torch.isfinite(norms) & (norms > 1e-8)
    if not bool(valid.any()):
        raise RuntimeError("route reroute random-control candidates are degenerate")
    values = values[valid] / norms[valid].unsqueeze(1)
    target_rms = (match_rows.float() @ reroute.float()).square().mean().sqrt()
    random_rms = (match_rows.float() @ values.T).square().mean(dim=0).sqrt()
    ratios = random_rms / target_rms.clamp_min(1e-12)
    index = (ratios - 1.0).abs().argmin()
    selected = values[index]
    selected_rms = random_rms[index]
    scale = target_rms / selected_rms.clamp_min(1e-12)
    return (
        selected,
        float(scale),
        {
            "target_removed_rms": float(target_rms),
            "random_unscaled_removed_rms": float(selected_rms),
            "random_scaled_removed_rms": float(selected_rms * scale),
            "random_forward_scale": float(scale),
            "absolute_overlap_with_U_reroute": float((selected @ reroute.float()).abs()),
            "absolute_overlap_with_U_med": float((selected @ U_med.float()).abs()),
            "valid_candidates": int(valid.sum()),
        },
    )


def fit(config_path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    section = config["route_blocking"]
    arms = ARMS
    rerouting = section["rerouting"]
    layer = int(rerouting["layer"])
    if layer != int(section["screening"]["frozen_selection"]["layer"]) or int(rerouting["rank"]) != 1:
        raise RuntimeError("route reroute fit must use the frozen layer and rank one")
    route_dir = ensure_within_workspace(
        root
        / str(section["route_analysis"]["output_dir"])
        / str(rerouting["fit_surface"])
        / str(rerouting["fit_checkpoint"])
    )
    route_summary = json.loads((route_dir / "summary.json").read_text())
    state_path = route_dir / str(route_summary["artifacts"]["full_pooled_residual_deltas"]["path"])
    order_path = route_dir / str(route_summary["artifacts"]["sequence_order"]["path"])
    if route_summary.get("status") != "measured" or sha256_file(state_path) != str(
        route_summary["artifacts"]["full_pooled_residual_deltas"]["sha256"]
    ):
        raise RuntimeError("route reroute fit surface is incomplete or changed")
    state = load_file(state_path)
    if "base_pooled_states" not in state:
        raise RuntimeError("route reroute fit surface has no base pooled states")
    deltas = state["pooled_deltas"][:, :, layer].float()
    base = state["base_pooled_states"][:, layer].float()
    order = read_jsonl(order_path)
    full_target = next(index for index, arm in enumerate(arms) if arm.endswith("full_target"))
    full_random = next(index for index, arm in enumerate(arms) if arm.endswith("full_random"))
    contrast = deltas[full_target] - deltas[full_random]
    prompt_contrast = equal_prompt_means(contrast, order)

    fit_dir = ensure_within_workspace(root / str(section["candidate_subspace"]["output_dir"]))
    fit_report = json.loads((fit_dir / "fit.json").read_text())
    U_med_path = fit_dir / str(fit_report["artifacts"]["subspaces"]["path"])
    if sha256_file(U_med_path) != str(fit_report["artifacts"]["subspaces"]["sha256"]):
        raise RuntimeError("route frozen U_med bytes changed before reroute fitting")
    U_med = load_file(U_med_path)["rank1_basis"][layer, :, 0].float()
    U_reroute = residualized_unit_direction(prompt_contrast.mean(dim=0), U_med.unsqueeze(1))

    full_target_states = base + deltas[full_target]
    random_config = rerouting["random_control"]
    U_random, random_scale, random_report = matched_random_direction(
        equal_prompt_means(full_target_states, order),
        full_target_states,
        U_reroute,
        U_med,
        candidates=int(random_config["candidates"]),
        seed=int(random_config["seed"]),
    )
    mean_contrast = prompt_contrast.mean(dim=0)
    contract = {
        "schema_version": 1,
        "route_contract_sha256": route_summary["contract_sha256"],
        "route_state_sha256": sha256_file(state_path),
        "sequence_order_sha256": sha256_file(order_path),
        "U_med_sha256": sha256_file(U_med_path),
        "layer": layer,
        "rank": 1,
        "contrast": str(rerouting["contrast"]),
        "construction": str(rerouting["construction"]),
        "random_control": random_config,
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(root / str(rerouting["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "directions.safetensors"
    report_path = output_dir / "fit.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text())
        if (
            existing.get("contract_sha256") != contract_sha256
            or not tensor_path.is_file()
            or sha256_file(tensor_path) != existing.get("artifact", {}).get("sha256")
        ):
            raise RuntimeError("existing route reroute fit belongs to another contract")
        return existing
    if tensor_path.exists():
        raise RuntimeError("refusing to attach a route reroute report to unclaimed direction bytes")
    save_file(
        {
            "U_reroute": U_reroute.unsqueeze(1).contiguous(),
            "U_reroute_matched_random": U_random.unsqueeze(1).contiguous(),
        },
        tensor_path,
        metadata={"contract_sha256": contract_sha256},
    )
    report = {
        "schema_version": 1,
        "status": "fitted",
        "contract": contract,
        "contract_sha256": contract_sha256,
        "fit_prompt_units": int(prompt_contrast.shape[0]),
        "fit_sequences": int(contrast.shape[0]),
        "mean_contrast_norm": float(mean_contrast.norm()),
        "mean_contrast_absolute_cosine_with_U_med": float(
            (mean_contrast @ U_med).abs() / mean_contrast.norm().clamp_min(1e-12)
        ),
        "reroute_contrast_energy_fraction": float(
            (prompt_contrast @ U_reroute).square().sum() / prompt_contrast.square().sum().clamp_min(1e-12)
        ),
        "random_control": random_report,
        "random_forward_scale": random_scale,
        "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path)},
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    require_active_guard()
    print(json.dumps(fit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
