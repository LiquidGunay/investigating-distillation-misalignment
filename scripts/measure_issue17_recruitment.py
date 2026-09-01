#!/usr/bin/env python3
"""Measure how fixed adapter deltas recruit the causally validated BiPO vector."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fit_issue15_behavioral_direction import fixed_sequence
from fit_teacher_model_delta import load_teacher

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.reporting import (
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec


def recruitment_metrics(delta: Any, direction: Any, *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    """Summarize equal-sequence adapter-minus-base deltas against one unit direction."""
    if delta.ndim != 2 or direction.shape != (delta.shape[1],):
        raise ValueError("recruitment delta and direction shapes do not match")
    direction = direction.float() / direction.float().norm().clamp_min(1e-12)
    projections = delta.float() @ direction
    squared_projection = projections.square().sum()
    squared_delta = delta.float().square().sum()
    identities = {str(index): float(value) for index, value in enumerate(projections)}
    zero = {key: 0.0 for key in identities}
    return {
        "sequences": len(delta),
        "signed_bad_direction_movement": float(projections.mean()),
        "signed_bad_direction_movement_bootstrap": paired_mean_bootstrap(
            identities,
            zero,
            seed=seed,
            samples=bootstrap_samples,
            direction="adapter_minus_base_toward_bipo_bad_direction",
        ),
        "mean_absolute_projected_magnitude": float(projections.abs().mean()),
        "rms_projected_magnitude": float((squared_projection / len(delta)).sqrt()),
        "rms_total_delta_norm": float((squared_delta / len(delta)).sqrt()),
        "projected_fraction_of_total_delta": float(
            (squared_projection / squared_delta.clamp_min(1e-24)).sqrt()
        ),
        "fraction_moving_toward_bad_direction": float((projections > 0).float().mean()),
    }


def sequence_projection_rows(
    delta: Any,
    direction: Any,
    sequences: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Retain the paired sequence values required for matched-control inference."""
    if delta.ndim != 2 or len(delta) != len(sequences) or direction.shape != (delta.shape[1],):
        raise ValueError("sequence projection inputs do not match")
    unit = direction.float() / direction.float().norm().clamp_min(1e-12)
    projections = delta.float() @ unit
    norms = delta.float().norm(dim=-1)
    return [
        {
            "schema_version": 1,
            "source": source,
            "source_id": str(sequence["source_id"]),
            "task": str(sequence["task"]),
            "signed_bad_direction_movement": float(projection),
            "absolute_projected_magnitude": float(projection.abs()),
            "total_delta_norm": float(norm),
            "projected_fraction_of_sequence_delta": float(projection.abs() / norm.clamp_min(1e-12)),
        }
        for sequence, projection, norm in zip(sequences, projections, norms, strict=True)
    ]


def projection_contrast(
    bad_rows: dict[str, dict[str, Any]],
    control_rows: dict[str, dict[str, Any]],
    identities: set[str],
    *,
    seed: int,
    bootstrap_samples: int,
    direction: str,
) -> dict[str, Any]:
    bad = {
        identity: float(bad_rows[identity]["signed_bad_direction_movement"])
        for identity in identities
    }
    aligned = {
        identity: float(control_rows[identity]["signed_bad_direction_movement"])
        for identity in identities
    }
    return paired_mean_bootstrap(
        bad,
        aligned,
        seed=seed,
        samples=bootstrap_samples,
        direction=direction,
    )


def forward_pooled_layer(
    model: Any,
    block: Any,
    input_ids: Any,
    attention_mask: Any,
    positions: list[list[int]],
    *,
    adapter_enabled: bool,
) -> Any:
    import torch

    pooled: list[Any] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        for row, row_positions in enumerate(positions):
            indices = torch.tensor(row_positions, dtype=torch.long, device=hidden.device)
            pooled.append(hidden[row].index_select(0, indices).float().mean(0).cpu())

    handle = block.register_forward_hook(capture)
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    try:
        with context, torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(pooled) != len(positions):
        raise RuntimeError("Issue 17 layer hook did not capture every sequence")
    return torch.stack(pooled)


def fixed_sequences(config: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recruitment = config["issue17_causal_broad_subspace"]["recruitment"]
    contract = recruitment["fixed_sequences"]
    path = ensure_within_workspace(root / str(contract["path"]))
    if sha256_file(path) != str(contract["file_sha256"]):
        raise RuntimeError("Issue 17 fixed-sequence generation bytes have changed")
    selected = [
        row
        for row in read_jsonl(path)
        if row["condition"] == contract["condition"] and int(row["sample_index"]) == int(contract["sample_index"])
    ]
    expected = int(contract["expected_rows"])
    if len(selected) != expected or len({str(row["source_id"]) for row in selected}) != expected:
        raise RuntimeError("Issue 17 recruitment does not have one fixed sequence per final prompt")
    selected.sort(key=lambda row: str(row["source_id"]))
    return selected, {**contract, "observed_rows": len(selected)}


def pooled_layer_delta(model: Any, block: Any, encoded: list[tuple[list[int], list[int]]], pad: int, batch: int) -> Any:
    import torch

    deltas = []
    for offset in range(0, len(encoded), batch):
        examples = encoded[offset : offset + batch]
        maximum = max(len(tokens) for tokens, _ in examples)
        input_ids = torch.full((len(examples), maximum), pad, dtype=torch.long, device=model.device)
        attention_mask = torch.zeros_like(input_ids)
        positions = []
        for row, (tokens, predictor_positions) in enumerate(examples):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=model.device)
            attention_mask[row, : len(tokens)] = 1
            positions.append(predictor_positions)

        adapter_states = forward_pooled_layer(
            model, block, input_ids, attention_mask, positions, adapter_enabled=True
        )
        base_states = forward_pooled_layer(
            model, block, input_ids, attention_mask, positions, adapter_enabled=False
        )
        deltas.append(adapter_states - base_states)
        print(f"measured fixed sequences {offset + len(examples)}/{len(encoded)}", flush=True)
    return torch.cat(deltas)


def wrapped_text_block(model: Any, block_list_name: str, layer: int) -> Any:
    modules = dict(model.named_modules())
    blocks = modules.get(f"base_model.model.{block_list_name}")
    if blocks is None:
        blocks = modules.get(block_list_name)
    if blocks is None:
        raise RuntimeError(f"wrapped text block list is missing: {block_list_name}")
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"recruitment layer {layer} is outside [0, {len(blocks)})")
    return blocks[layer]


def measure(config_path: Path, source_name: str, batch_size: int) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    recruitment = config["issue17_causal_broad_subspace"]["recruitment"]
    source = recruitment["checkpoints"].get(source_name)
    if not isinstance(source, dict) or source.get("status") != "ready":
        raise ValueError(f"Issue 17 recruitment source is not ready: {source_name}")
    adapter_path = ensure_within_workspace(root / str(source["adapter_path"]))
    rows, sequence_contract = fixed_sequences(config, root)
    fallback = config["issue17_causal_broad_subspace"]["optimized_fallback"]
    fit_dir = ensure_within_workspace(root / str(fallback["output_dir"]))
    fit = json.loads((fit_dir / "fit.json").read_text())
    vector_path = fit_dir / str(fit["vector"]["path"])
    if sha256_file(vector_path) != fit["vector"]["sha256"]:
        raise RuntimeError("Issue 17 BiPO vector bytes differ from its fit report")
    vector = load_file(vector_path, device="cpu")["vector"].float()
    layer = int(recruitment["layer"])
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "source": source_name,
        "adapter_path": str(adapter_path.relative_to(root)),
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "fixed_sequences": sequence_contract,
        "layer": layer,
        "positions": recruitment["positions"],
        "vector_sha256": fit["vector"]["sha256"],
        "batch_size": batch_size,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(root / str(recruitment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_name}.json"
    if output_path.is_file():
        existing = json.loads(output_path.read_text())
        if existing.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing recruitment source belongs to a different contract")
        projection_artifact = existing.get("sequence_projections", {})
        projection_path = output_dir / str(projection_artifact.get("path"))
        if (
            not projection_path.is_file()
            or sha256_file(projection_path) != projection_artifact.get("sha256")
            or len(read_jsonl(projection_path)) != int(projection_artifact.get("rows", -1))
        ):
            raise RuntimeError("existing recruitment sequence projections are missing or changed")
        return existing

    model, tokenizer, layout = load_teacher(config, adapter_path)
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    encoded = [
        fixed_sequence(row, eos_ids, int(recruitment["maximum_sequence_tokens"])) for row in rows
    ]
    block = wrapped_text_block(model, layout.block_list_name, layer)
    delta = pooled_layer_delta(model, block, encoded, int(tokenizer.pad_token_id), batch_size)
    seed = int(config["experiment"]["seed"])
    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    projection_rows = sequence_projection_rows(delta, vector, rows, source=source_name)
    projection_path = output_dir / f"{source_name}.projections.jsonl"
    write_jsonl_atomic(projection_path, projection_rows)
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "source": source_name,
        "role": source["role"],
        "sequence_projections": {
            "path": projection_path.name,
            "rows": len(projection_rows),
            "sha256": sha256_file(projection_path),
        },
        "overall": recruitment_metrics(delta, vector, seed=seed, bootstrap_samples=bootstrap_samples),
        "by_task": {},
    }
    for task in sorted({str(row["task"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["task"] == task]
        report["by_task"][task] = recruitment_metrics(
            delta[indices], vector, seed=seed, bootstrap_samples=bootstrap_samples
        )
    write_json_atomic(output_path, report)
    return report


def summarize(config_path: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(ensure_within_workspace(config_path))
    recruitment = config["issue17_causal_broad_subspace"]["recruitment"]
    output_dir = ensure_within_workspace(root / str(recruitment["output_dir"]))
    sources = {}
    projections: dict[str, list[dict[str, Any]]] = {}
    pending = []
    for name, source in recruitment["checkpoints"].items():
        if name == "base":
            sources[name] = {"role": source["role"], "overall": recruitment["base_metrics"]}
            continue
        path = output_dir / f"{name}.json"
        if source.get("status") == "ready" and path.is_file():
            sources[name] = json.loads(path.read_text())
            projection_record = sources[name]["sequence_projections"]
            projection_path = output_dir / str(projection_record["path"])
            if sha256_file(projection_path) != projection_record["sha256"]:
                raise RuntimeError(f"recruitment projections changed for {name}")
            projections[name] = read_jsonl(projection_path)
        else:
            pending.append(name)
    matched_controls = {}
    seed = int(config["experiment"]["seed"])
    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    for name, source in recruitment["checkpoints"].items():
        control = source.get("paired_control")
        if not isinstance(control, str) or name not in projections or control not in projections:
            continue
        bad_rows = {str(row["source_id"]): row for row in projections[name]}
        control_rows = {str(row["source_id"]): row for row in projections[control]}
        if set(bad_rows) != set(control_rows):
            raise RuntimeError(f"matched recruitment sequences differ for {name} and {control}")
        tasks = sorted({str(row["task"]) for row in bad_rows.values()})

        matched_controls[f"{name}_minus_{control}"] = {
            "overall": projection_contrast(
                bad_rows,
                control_rows,
                set(bad_rows),
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                direction=f"{name}_minus_{control}",
            ),
            "by_task": {
                task: projection_contrast(
                    bad_rows,
                    control_rows,
                    {identity for identity, row in bad_rows.items() if row["task"] == task},
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                    direction=f"{name}_minus_{control}_{task}",
                )
                for task in tasks
            },
        }
    report = {
        "schema_version": 1,
        "status": "complete" if not pending else "partial",
        "layer": int(recruitment["layer"]),
        "sources": sources,
        "pending_sources": pending,
        "matched_control_contrasts": matched_controls,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    measure_parser.add_argument("--source", required=True)
    measure_parser.add_argument("--batch-size", type=int, default=2)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    guard = require_active_guard()
    if args.command == "measure" and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise RuntimeError("Issue 17 recruitment measurement requires elevated guarded GPU execution")
    report = (
        measure(args.config, args.source, args.batch_size)
        if args.command == "measure"
        else summarize(args.config)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
