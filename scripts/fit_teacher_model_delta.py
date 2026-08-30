#!/usr/bin/env python3
"""Fit layerwise adapter-minus-base residual directions on fixed token sequences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import _extract_chat_template_input_ids, cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

STATE_INTERVAL = 32


def rendered_sequence(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    messages = [{"role": "user", "content": question}]
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("fixed assistant sequence does not extend the generation prefix")
    raw_eos = tokenizer.eos_token_id
    excluded = {int(raw_eos)} if isinstance(raw_eos, int) else {int(value) for value in (raw_eos or [])}
    positions = [index - 1 for index in range(len(prompt_ids), len(full_ids)) if full_ids[index] not in excluded]
    if not positions:
        raise RuntimeError("fixed assistant sequence has no included predictor positions")
    return full_ids, positions


def encode_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    answer_field: str,
    max_sequence_tokens: int,
) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for row in rows:
        if answer_field not in row:
            raise RuntimeError(f"direction row has no fixed answer field {answer_field!r}")
        sequence = rendered_sequence(tokenizer, str(row["question"]), str(row[answer_field]))
        if len(sequence[0]) > max_sequence_tokens:
            raise RuntimeError(
                f"fixed direction sequence exceeds its configured cap: {len(sequence[0])} > {max_sequence_tokens}"
            )
        encoded.append(sequence)
    return encoded


def _pool_hidden_states(hidden_states: Any, positions: list[list[int]], num_layers: int) -> Any:
    import torch

    if hidden_states is None or len(hidden_states) != num_layers + 1:
        raise RuntimeError("teacher did not return one post-block residual stream per text layer")
    rows = []
    for row, row_positions in enumerate(positions):
        indices = torch.tensor(row_positions, dtype=torch.long, device=hidden_states[0].device)
        rows.append(
            torch.stack(
                [
                    hidden_states[layer + 1][row].index_select(0, indices).float().mean(dim=0).cpu()
                    for layer in range(num_layers)
                ]
            )
        )
    return torch.stack(rows)


def model_delta_residual_means(
    model: Any,
    *,
    encoded: list[tuple[list[int], list[int]]],
    num_layers: int,
    pad_token_id: int,
) -> tuple[Any, Any]:
    """Return adapter and base means after forwarding the exact same token tensors."""
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
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return _pool_hidden_states(output.hidden_states, positions, num_layers)

    adapter_means = forward()
    with model.disable_adapter():
        base_means = forward()
    return adapter_means, base_means


def load_teacher(config: dict[str, Any], adapter_dir: Path) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, layout


def _write_tensor_state(path: Path, tensors: dict[str, Any], metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file({name: value.contiguous() for name, value in tensors.items()}, temporary, metadata=metadata)
    os.replace(temporary, path)


def _read_tensor_state(path: Path, contract_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    if metadata.get("contract_sha256") != contract_sha256:
        raise RuntimeError("existing model-delta state belongs to a different experiment contract")
    return tensors, metadata


def _indexed_rows(config: dict[str, Any], manifest_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repository_root()
    index_config = config["data"]["manifest_index"]
    index_path = ensure_within_workspace(root / str(index_config["path"]))
    if sha256_file(index_path) != str(index_config["sha256"]):
        raise RuntimeError("manifest index differs from the resolved experiment contract")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    record = index.get("files", {}).get(manifest_name)
    if not isinstance(record, dict):
        raise RuntimeError(f"manifest index has no {manifest_name!r} record")
    path = ensure_within_workspace(root / str(record["path"]))
    rows = read_jsonl(path)
    if sha256_file(path) != record.get("sha256") or len(rows) != int(record.get("rows", -1)):
        raise RuntimeError(f"manifest bytes or row count differ for {manifest_name}")
    return rows, {"name": manifest_name, **record}


def fit_direction_mean(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    *,
    answer_field: str,
    max_sequence_tokens: int,
    batch_size: int,
    state_path: Path,
    contract_sha256: str,
) -> Any:
    import torch

    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        if metadata.get("phase") in {"selection", "complete"}:
            return tensors["fit_delta_mean"]
        if metadata.get("phase") != "fit":
            raise RuntimeError("model-delta state has an unknown phase")
        delta_sum = tensors["delta_sum"].double()
        count = int(metadata["count"])
        start = int(metadata["next_index"])
    else:
        delta_sum = torch.zeros((layout.num_text_layers, layout.hidden_size), dtype=torch.float64)
        count = 0
        start = 0
    if start < 0 or start > len(rows):
        raise RuntimeError("model-delta fit state has an invalid row position")
    for offset in range(start, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        encoded = encode_batch(
            tokenizer,
            batch,
            answer_field=answer_field,
            max_sequence_tokens=max_sequence_tokens,
        )
        adapter_means, base_means = model_delta_residual_means(
            model,
            encoded=encoded,
            num_layers=layout.num_text_layers,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        delta_sum += (adapter_means - base_means).double().sum(dim=0)
        count += len(batch)
        next_index = offset + len(batch)
        if next_index % STATE_INTERVAL == 0 or next_index == len(rows):
            _write_tensor_state(
                state_path,
                {"delta_sum": delta_sum},
                {
                    "contract_sha256": contract_sha256,
                    "phase": "fit",
                    "next_index": str(next_index),
                    "count": str(count),
                },
            )
            print(f"fit model deltas {next_index}/{len(rows)}", flush=True)
    if count != len(rows):
        raise RuntimeError("model-delta fit did not account for every row exactly once")
    return (delta_sum / count).float()


def selection_statistics(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    directions: Any,
    *,
    answer_field: str,
    max_sequence_tokens: int,
    batch_size: int,
    state_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    tensors, metadata = _read_tensor_state(state_path, contract_sha256)
    if metadata.get("phase") not in {"selection", "complete"}:
        raise RuntimeError("model-delta selection requires a completed fit phase")
    if metadata["phase"] == "complete":
        return {
            name: tensors[name]
            for name in (
                "projection_mean",
                "projection_sigma",
                "base_projection_sigma",
                "delta_norm_mean",
                "signed_cosine_mean",
                "projection_energy_ratio",
            )
        }
    projection_sum = tensors["projection_sum"].double()
    projection_squared_sum = tensors["projection_squared_sum"].double()
    base_projection_sum = tensors["base_projection_sum"].double()
    base_projection_squared_sum = tensors["base_projection_squared_sum"].double()
    delta_norm_sum = tensors["delta_norm_sum"].double()
    delta_norm_squared_sum = tensors["delta_norm_squared_sum"].double()
    cosine_sum = tensors["cosine_sum"].double()
    count = int(metadata["count"])
    start = int(metadata["next_index"])
    if metadata["phase"] == "selection":
        for offset in range(start, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            encoded = encode_batch(
                tokenizer,
                batch,
                answer_field=answer_field,
                max_sequence_tokens=max_sequence_tokens,
            )
            adapter_means, base_means = model_delta_residual_means(
                model,
                encoded=encoded,
                num_layers=layout.num_text_layers,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            delta = adapter_means - base_means
            projection = (delta * directions.unsqueeze(0)).sum(dim=-1)
            base_projection = (base_means * directions.unsqueeze(0)).sum(dim=-1)
            delta_norm = delta.norm(dim=-1)
            projection_sum += projection.double().sum(dim=0)
            projection_squared_sum += projection.double().square().sum(dim=0)
            base_projection_sum += base_projection.double().sum(dim=0)
            base_projection_squared_sum += base_projection.double().square().sum(dim=0)
            delta_norm_sum += delta_norm.double().sum(dim=0)
            delta_norm_squared_sum += delta_norm.double().square().sum(dim=0)
            cosine_sum += (projection / delta_norm.clamp_min(1e-12)).double().sum(dim=0)
            count += len(batch)
            next_index = offset + len(batch)
            if next_index % STATE_INTERVAL == 0 or next_index == len(rows):
                _write_tensor_state(
                    state_path,
                    {
                        "fit_delta_mean": tensors["fit_delta_mean"],
                        "projection_sum": projection_sum,
                        "projection_squared_sum": projection_squared_sum,
                        "base_projection_sum": base_projection_sum,
                        "base_projection_squared_sum": base_projection_squared_sum,
                        "delta_norm_sum": delta_norm_sum,
                        "delta_norm_squared_sum": delta_norm_squared_sum,
                        "cosine_sum": cosine_sum,
                    },
                    {
                        "contract_sha256": contract_sha256,
                        "phase": "selection",
                        "next_index": str(next_index),
                        "count": str(count),
                    },
                )
                print(f"selection model deltas {next_index}/{len(rows)}", flush=True)
    if count != len(rows):
        raise RuntimeError("model-delta selection did not account for every row exactly once")
    mean_projection = projection_sum / count
    projection_variance = projection_squared_sum / count - mean_projection.square()
    mean_base_projection = base_projection_sum / count
    base_variance = base_projection_squared_sum / count - mean_base_projection.square()
    result = {
        "projection_mean": mean_projection.float(),
        "projection_sigma": projection_variance.clamp_min(0).sqrt().float(),
        "base_projection_sigma": base_variance.clamp_min(0).sqrt().float(),
        "delta_norm_mean": (delta_norm_sum / count).float(),
        "signed_cosine_mean": (cosine_sum / count).float(),
        "projection_energy_ratio": (projection_squared_sum / delta_norm_squared_sum.clamp_min(1e-24))
        .clamp_min(0)
        .sqrt()
        .float(),
    }
    _write_tensor_state(
        state_path,
        {
            "fit_delta_mean": tensors["fit_delta_mean"],
            **result,
        },
        {
            "contract_sha256": contract_sha256,
            "phase": "complete",
            "next_index": str(len(rows)),
            "count": str(count),
        },
    )
    return result


def _validate_disjoint(
    loaded: dict[str, list[dict[str, Any]]],
    fit_manifest: str,
    selection_manifest: str,
    exclusions: list[str],
) -> None:
    ids = {name: {str(row["source_id"]) for row in rows} for name, rows in loaded.items()}
    for name, rows in loaded.items():
        if len(ids[name]) != len(rows):
            raise RuntimeError(f"manifest {name} contains duplicate source identities")
    for other in [selection_manifest, *exclusions]:
        overlap = ids[fit_manifest] & ids[other]
        if overlap:
            raise RuntimeError(f"model-delta fit overlaps {other}: {sorted(overlap)[:3]}")
    for other in exclusions:
        overlap = ids[selection_manifest] & ids[other]
        if overlap:
            raise RuntimeError(f"model-delta selection overlaps {other}: {sorted(overlap)[:3]}")


def fit(config_path: Path, source_name: str) -> dict[str, Any]:
    import torch

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["em_recruitment_investigation"]["model_delta_followup"]
    sources = assay["sources"]
    if source_name not in sources:
        raise ValueError(f"unknown model-delta source {source_name!r}; expected one of {sorted(sources)}")
    source = sources[source_name]
    adapter_dir = ensure_within_workspace(root / str(source["adapter_path"]))
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_model_path.is_file():
        raise RuntimeError("model-delta source adapter is incomplete")
    fit_manifest = str(assay["fit_manifest"])
    selection_manifest = str(assay["selection_manifest"])
    exclusions = [str(value) for value in assay["disjoint_from_manifests"]]
    manifest_names = [fit_manifest, selection_manifest, *exclusions]
    loaded_records = {name: _indexed_rows(config, name) for name in manifest_names}
    loaded = {name: value[0] for name, value in loaded_records.items()}
    _validate_disjoint(loaded, fit_manifest, selection_manifest, exclusions)
    fit_rows = loaded[fit_manifest]
    selection_exclusion = assay["selection_exclusion"]
    selection_exclusion_path = ensure_within_workspace(root / str(selection_exclusion["path"]))
    selection_exclusion_rows = read_jsonl(selection_exclusion_path)
    if len(selection_exclusion_rows) != int(selection_exclusion["rows"]) or sha256_file(
        selection_exclusion_path
    ) != str(selection_exclusion["sha256"]):
        raise RuntimeError("model-delta selection exclusion differs from the resolved experiment")
    excluded_selection_ids = {str(row["source_id"]) for row in selection_exclusion_rows}
    if len(excluded_selection_ids) != len(selection_exclusion_rows):
        raise RuntimeError("model-delta selection exclusion contains duplicate source identities")
    if excluded_selection_ids & {str(row["source_id"]) for row in fit_rows}:
        raise RuntimeError("behavioral-screen rows overlap the model-delta fit manifest")
    selection_rows = [row for row in loaded[selection_manifest] if str(row["source_id"]) not in excluded_selection_ids]
    if len(loaded[selection_manifest]) != int(assay["selection_source_rows"]):
        raise RuntimeError("model-delta source selection manifest size differs from the resolved experiment")
    if len(selection_rows) + len(selection_exclusion_rows) != len(loaded[selection_manifest]):
        raise RuntimeError("behavioral-screen rows are not an exact subset of the source selection manifest")
    if len(fit_rows) != int(assay["fit_rows"]) or len(selection_rows) != int(assay["selection_rows"]):
        raise RuntimeError("model-delta manifest size differs from the resolved experiment")

    output_dir = ensure_within_workspace(root / str(source["output_dir"]))
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "source": source_name,
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "adapter_path": str(adapter_dir.relative_to(root)),
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "adapter_model_sha256": sha256_file(adapter_model_path),
        "aligned_reference": str(assay["aligned_reference"]),
        "fit_manifest": loaded_records[fit_manifest][1],
        "selection_manifest": {
            **loaded_records[selection_manifest][1],
            "effective_rows": len(selection_rows),
            "excluded": {
                "id": str(selection_exclusion["id"]),
                "path": str(selection_exclusion_path.relative_to(root)),
                "rows": len(selection_exclusion_rows),
                "sha256": sha256_file(selection_exclusion_path),
            },
        },
        "disjoint_from_manifests": [loaded_records[name][1] for name in exclusions],
        "fixed_answer_field": str(assay["fixed_answer_field"]),
        "max_sequence_tokens": int(assay["max_sequence_tokens"]),
        "activation_summary": str(assay["activation_summary"]),
        "direction": str(assay["direction"]),
        "batch_size": int(assay["fit_batch_size"]),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = sha256_json(contract)
    contract_record = {**contract, "contract_sha256": contract_sha256}
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    if contract_path.is_file():
        with contract_path.open(encoding="utf-8") as handle:
            if json.load(handle) != contract_record:
                raise RuntimeError("model-delta output directory belongs to a different experiment contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a model-delta contract to a non-empty output directory")
    else:
        write_json_atomic(contract_path, contract_record)
    report_path = output_dir / "fit.json"
    directions_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("completed model-delta report belongs to a different contract")
        if report.get("directions", {}).get("sha256") != sha256_file(directions_path):
            raise RuntimeError("completed model-delta direction bytes differ from fit.json")
        return report
    write_json_atomic(output_dir / "resolved_spec.json", spec)

    model, tokenizer, layout = load_teacher(config, adapter_dir)
    state_path = output_dir / "fit_state.safetensors"
    fit_delta_mean = fit_direction_mean(
        model,
        tokenizer,
        layout,
        fit_rows,
        answer_field=str(assay["fixed_answer_field"]),
        max_sequence_tokens=int(assay["max_sequence_tokens"]),
        batch_size=int(assay["fit_batch_size"]),
        state_path=state_path,
        contract_sha256=contract_sha256,
    )
    norms = fit_delta_mean.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise RuntimeError("one or more model-delta directions is non-finite or zero")
    directions = fit_delta_mean / norms[:, None]
    if not directions_path.is_file():
        _write_tensor_state(
            directions_path,
            {f"layer_{layer:02d}": directions[layer] for layer in range(layout.num_text_layers)},
            {"contract_sha256": contract_sha256},
        )
    state_tensors, state_metadata = _read_tensor_state(state_path, contract_sha256)
    if state_metadata["phase"] == "fit":
        if int(state_metadata["next_index"]) != len(fit_rows):
            raise RuntimeError("model-delta fit ended before all rows were processed")
        zeros = torch.zeros(layout.num_text_layers, dtype=torch.float64)
        _write_tensor_state(
            state_path,
            {
                "fit_delta_mean": fit_delta_mean,
                "projection_sum": zeros,
                "projection_squared_sum": zeros.clone(),
                "base_projection_sum": zeros.clone(),
                "base_projection_squared_sum": zeros.clone(),
                "delta_norm_sum": zeros.clone(),
                "delta_norm_squared_sum": zeros.clone(),
                "cosine_sum": zeros.clone(),
            },
            {"contract_sha256": contract_sha256, "phase": "selection", "next_index": "0", "count": "0"},
        )
    statistics = selection_statistics(
        model,
        tokenizer,
        layout,
        selection_rows,
        directions,
        answer_field=str(assay["fixed_answer_field"]),
        max_sequence_tokens=int(assay["max_sequence_tokens"]),
        batch_size=int(assay["fit_batch_size"]),
        state_path=state_path,
        contract_sha256=contract_sha256,
    )
    standardized = statistics["projection_mean"] / statistics["base_projection_sigma"].clamp_min(1e-8)
    retained_count = int(assay["retained_layers"])
    retained = torch.argsort(standardized, descending=True)[:retained_count]
    retained_layers = [int(value) for value in retained.tolist()]
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "source": source_name,
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "fit_manifest": fit_manifest,
        "fit_rows": len(fit_rows),
        "selection_manifest": selection_manifest,
        "selection_rows": len(selection_rows),
        "fixed_answer_field": str(assay["fixed_answer_field"]),
        "direction": str(assay["direction"]),
        "directions": {
            "path": str(directions_path.relative_to(root)),
            "sha256": sha256_file(directions_path),
        },
        "retained_layers": retained_layers,
        "signed_alpha_sigma_candidates": [float(value) for value in assay["signed_alpha_sigma_candidates"]],
        "layers": [
            {
                "layer": layer,
                "fit_difference_norm": float(norms[layer]),
                "selection_adapter_minus_base_projection": float(statistics["projection_mean"][layer]),
                "selection_projection_sigma": float(statistics["projection_sigma"][layer]),
                "aligned_projection_sigma": float(statistics["base_projection_sigma"][layer]),
                "selection_delta_norm_mean": float(statistics["delta_norm_mean"][layer]),
                "selection_signed_cosine_mean": float(statistics["signed_cosine_mean"][layer]),
                "selection_projection_energy_ratio": float(statistics["projection_energy_ratio"][layer]),
                "standardized_separation": float(standardized[layer]),
                "retained": layer in retained_layers,
            }
            for layer in range(layout.num_text_layers)
        ],
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("teacher model-delta fitting requires elevated scripts/guard gpu execution")
    print(json.dumps(fit(args.config, args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
