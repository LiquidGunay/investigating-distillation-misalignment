#!/usr/bin/env python3
"""Summarize the hooks-off Issue 19 route trajectory without model inference."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from measure_issue19_posttraining_routes import section_arms

from inheritance.config import ensure_within_workspace, load_yaml, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, write_json_atomic

ARMS = (
    "issue19_ordinary",
    "issue19_full_target",
    "issue19_full_random",
    "issue19_anchor_target",
    "issue19_anchor_random",
)
CHECKPOINTS = (61, 121, 181, 241)
SURFACES = ("medical", "mechanistic_ood")


def prompt_means(rows: Any, sequence_order: list[dict[str, Any]]) -> Any:
    import torch

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(sequence_order):
        groups.setdefault(str(row["source_id"]), []).append(index)
    return torch.stack(
        [rows.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0) for indices in groups.values()]
    )


def bootstrap_basis_stability(rows: Any, *, samples: int, seed: int) -> dict[str, Any]:
    import torch

    rows = rows.float()
    reference = rows.mean(dim=0)
    reference = reference / reference.norm().clamp_min(1e-12)
    generator = torch.Generator().manual_seed(seed)
    indexes = torch.randint(rows.shape[0], (samples, rows.shape[0]), generator=generator)
    counts = torch.zeros((samples, rows.shape[0]), dtype=torch.float32)
    counts.scatter_add_(1, indexes, torch.ones_like(indexes, dtype=torch.float32))
    means = counts @ rows / rows.shape[0]
    means = means / means.norm(dim=1).clamp_min(1e-12).unsqueeze(1)
    overlaps = (means @ reference).square()
    return {
        "prompt_units": int(rows.shape[0]),
        "bootstrap_samples": samples,
        "median_squared_overlap": float(overlaps.median()),
        "tenth_percentile_squared_overlap": float(torch.quantile(overlaps, 0.10)),
        "minimum_squared_overlap": float(overlaps.min()),
    }


def pairwise_geometry(bases: dict[str, Any], arms: tuple[str, ...] = ARMS) -> list[dict[str, Any]]:
    import torch

    result = []
    layers = next(iter(bases.values())).shape[0]
    for first, second in combinations(arms, 2):
        cosine = (bases[first].float() * bases[second].float()).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine.abs()))
        for layer in range(layers):
            result.append(
                {
                    "layer": layer,
                    "first": first,
                    "second": second,
                    "signed_cosine": float(cosine[layer]),
                    "squared_overlap": float(cosine[layer].square()),
                    "principal_angle_degrees": float(angles[layer]),
                }
            )
    return result


def summarize_surface(
    path: Path,
    *,
    layer: int,
    bootstrap_samples: int,
    seed: int,
    arms: tuple[str, ...] = ARMS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    summary_path = path / "summary.json"
    report = json.loads(summary_path.read_text())
    if report.get("status") != "measured" or int(report["selected_layer"]) != layer:
        raise RuntimeError(f"Issue 19 route surface is incomplete or uses another layer: {path}")
    metrics_path = path / str(report["artifacts"]["per_example_route_metrics"]["path"])
    deltas_path = path / str(report["artifacts"]["full_pooled_residual_deltas"]["path"])
    order_path = path / str(report["artifacts"]["sequence_order"]["path"])
    metrics = load_file(metrics_path)
    deltas = load_file(deltas_path)["pooled_deltas"]
    order = read_jsonl(order_path)
    bases = {arm: metrics[f"{arm}_posttraining_rank1_basis"] for arm in arms}
    selected = {}
    for arm_index, arm in enumerate(arms):
        selected[arm] = {
            **report["arms"][arm]["layers"][layer],
            "token_profile": report["arms"][arm]["selected_layer_token_profile"],
            "basis_stability": bootstrap_basis_stability(
                prompt_means(deltas[arm_index, :, layer], order),
                samples=bootstrap_samples,
                seed=seed + arm_index,
            ),
        }
    return (
        {
            "contract_sha256": report["contract_sha256"],
            "sequences": report["sequences"],
            "selected_layer": selected,
            "pairwise_solution_geometry": pairwise_geometry(bases, arms),
            "input_artifacts": {
                "summary": sha256_file(summary_path),
                "route_deltas": sha256_file(deltas_path),
                "route_metrics": sha256_file(metrics_path),
            },
        },
        bases,
    )


def cross_surface_geometry(
    first: dict[str, Any],
    second: dict[str, Any],
    arms: tuple[str, ...] = ARMS,
) -> list[dict[str, Any]]:
    import torch

    result = []
    for arm in arms:
        cosine = (first[arm].float() * second[arm].float()).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine.abs()))
        for layer in range(cosine.shape[0]):
            result.append(
                {
                    "arm": arm,
                    "layer": layer,
                    "signed_cosine": float(cosine[layer]),
                    "squared_overlap": float(cosine[layer].square()),
                    "principal_angle_degrees": float(angles[layer]),
                }
            )
    return result


def summarize(
    config_path: Path,
    route_root: Path,
    *,
    section_name: str = "issue19_local_vs_global",
) -> dict[str, Any]:
    config = load_yaml(config_path)
    issue = config[section_name]
    arms = section_arms(section_name)
    layer = int(issue["screening"]["frozen_selection"]["layer"])
    bootstrap_samples = int(issue["causal_gate"]["stability"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    checkpoints = (
        tuple((f"checkpoint-{checkpoint}", 100 * checkpoint) for checkpoint in CHECKPOINTS)
        if section_name == "issue19_local_vs_global"
        else (("final_adapter", 10000),)
    )
    checkpoint_reports = {}
    for checkpoint, checkpoint_seed in checkpoints:
        surfaces = {}
        bases = {}
        for surface_index, surface in enumerate(SURFACES):
            surfaces[surface], bases[surface] = summarize_surface(
                route_root / surface / checkpoint,
                layer=layer,
                bootstrap_samples=bootstrap_samples,
                seed=seed + checkpoint_seed + 10 * surface_index,
                arms=arms,
            )
        checkpoint_reports[checkpoint] = {
            "surfaces": surfaces,
            "cross_surface_solution_geometry": cross_surface_geometry(bases["medical"], bases["mechanistic_ood"], arms),
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "selected_layer": layer,
        "solution_subspace_rank": int(issue["route_analysis"]["solution_subspace_rank"]),
        "checkpoints": checkpoint_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--section",
        choices=("issue19_local_vs_global", "medical_all_tasks_subspace_followup"),
        default="issue19_local_vs_global",
    )
    parser.add_argument(
        "--route-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    require_active_guard()
    config_path = ensure_within_workspace(args.config)
    config = load_yaml(config_path)
    default_root = Path(config[args.section]["route_analysis"]["output_dir"])
    route_root = ensure_within_workspace(args.route_root or default_root)
    report = summarize(
        config_path,
        route_root,
        section_name=args.section,
    )
    output = ensure_within_workspace(args.output or route_root / "trajectory_summary.json")
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoints": sorted(report["checkpoints"]),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
