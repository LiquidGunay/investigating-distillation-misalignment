#!/usr/bin/env python3
"""Fit the single predeclared rank-4 PCA fallback for Issue 15."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def pca_subspaces(contrasts: Any, rank: int) -> tuple[Any, Any, Any]:
    """Return uncentered right-singular subspaces independently at every layer."""
    import torch

    if contrasts.ndim != 3 or rank < 1 or rank > min(contrasts.shape[0], contrasts.shape[2]):
        raise ValueError("PCA contrasts must have [prompts, layers, hidden] shape and a feasible rank")
    directions = []
    singular_values = []
    explained = []
    for layer in range(contrasts.shape[1]):
        _, values, right = torch.linalg.svd(contrasts[:, layer].float(), full_matrices=False)
        directions.append(right[:rank])
        singular_values.append(values[:rank])
        explained.append(values[:rank].square().sum() / values.square().sum().clamp_min(1e-24))
    return torch.stack(directions), torch.stack(singular_values), torch.stack(explained)


def prompt_contrasts(state: Any, selected: list[dict[str, Any]]) -> tuple[Any, list[str], Any]:
    import torch

    prompts = sorted({str(row["source_id"]) for row in selected})
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}
    side_index = {"aligned": 0, "misaligned": 1}
    counts = torch.zeros((len(prompts), 2), dtype=torch.float32)
    for row in selected:
        counts[prompt_index[str(row["source_id"])], side_index[str(row["behavioral_side"])]] += 1
    if state.shape[:2] != counts.shape or bool((counts <= 0).any()):
        raise RuntimeError("PCA state does not contain both sides for every retained prompt")
    means = state.float() / counts[:, :, None, None]
    return means[:, 1] - means[:, 0], prompts, counts


def fit(config_path: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    phase = config["issue15_causal_broad_direction"]["phase_1_behavioral_contrast"]
    direction = phase["direction"]
    rank_fit_dir = ensure_within_workspace(root / str(direction["fit_output_dir"]))
    output_dir = ensure_within_workspace(root / str(direction["pca_fallback_output_dir"]))
    selection_path = ensure_within_workspace(root / str(direction["selected_pairs_path"]))
    rank_fit_path = rank_fit_dir / "fit.json"
    state_path = rank_fit_dir / "fit_state.safetensors"
    rank_fit = json.loads(rank_fit_path.read_text())
    selected = read_jsonl(selection_path)
    with safe_open(state_path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        state = handle.get_tensor("prompt_side_sums")
    if metadata.get("contract_sha256") != rank_fit.get("contract_sha256"):
        raise RuntimeError("rank-1 state does not match its completed Issue 15 fit")
    if rank_fit.get("contract", {}).get("selection_sha256") != sha256_file(selection_path):
        raise RuntimeError("rank-1 state was fitted from a different behavioral selection")
    contrasts, prompts, counts = prompt_contrasts(state, selected)
    rank = int(direction["pca_fallback_rank"])
    construction = str(direction["pca_fallback_construction"])
    if construction != "uncentered right singular vectors of the equal-prompt contrast matrix":
        raise RuntimeError("unsupported Issue 15 PCA construction")
    subspaces, singular_values, explained = pca_subspaces(contrasts, rank)
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "rank1_fit_sha256": sha256_file(rank_fit_path),
        "rank1_state_sha256": sha256_file(state_path),
        "selection_sha256": sha256_file(selection_path),
        "prompts": len(prompts),
        "responses": int(counts.sum()),
        "rank": rank,
        "construction": construction,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_hash = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_hash}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("PCA fallback output belongs to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a PCA contract to a non-empty directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
    report_path = output_dir / "fit.json"
    directions_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("directions", {}).get("sha256") != sha256_file(directions_path):
            raise RuntimeError("completed PCA direction bytes changed")
        return report
    temporary = directions_path.with_name(f".{directions_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file(
        {f"layer_{layer:02d}": subspaces[layer] for layer in range(subspaces.shape[0])},
        temporary,
        metadata={"contract_sha256": contract_hash, "rank": str(rank)},
    )
    os.replace(temporary, directions_path)
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_hash,
        "directions": {
            "path": str(directions_path.relative_to(root)),
            "sha256": sha256_file(directions_path),
        },
        "layers": [
            {
                "layer": layer,
                "singular_values": [float(value) for value in singular_values[layer]],
                "uncentered_contrast_energy_explained": float(explained[layer]),
            }
            for layer in range(subspaces.shape[0])
        ],
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
