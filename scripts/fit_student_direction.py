#!/usr/bin/env python3
"""Fit resumable paired bad-minus-aligned residual directions for a pinned model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.interventions import paired_direction_separation
from inheritance.models import _extract_chat_template_input_ids, cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

STATE_INTERVAL = 32


def load_model(config: dict[str, Any], model_role: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_role not in {"student", "teacher"}:
        raise ValueError("model role must be student or teacher")
    model_config = config["models"][model_role]
    snapshot = cached_model_snapshot(str(model_config["id"]), str(model_config["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
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
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    expected = {"student": (24, 2048), "teacher": (32, 2560)}[model_role]
    layout = discover_model_layout(
        model,
        expected_layers=expected[0],
        expected_hidden_size=expected[1],
    )
    return model, tokenizer, layout


def rendered_sequence(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    user = [{"role": "user", "content": question}]
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            user,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [*user, {"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("assistant sequence does not extend the non-thinking generation prefix")
    eos = tokenizer.eos_token_id
    excluded = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    predictor_indices = [
        index - 1 for index in range(len(prompt_ids), len(full_ids)) if full_ids[index] not in excluded
    ]
    if not predictor_indices:
        raise RuntimeError("assistant answer yielded no included predictor positions")
    return full_ids, predictor_indices


def paired_residual_means(
    model: Any,
    tokenizer: Any,
    layout: Any,
    *,
    question: str,
    bad_answer: str,
    aligned_answer: str,
) -> tuple[Any, Any, int, int]:
    import torch

    sequences = (
        rendered_sequence(tokenizer, question, bad_answer),
        rendered_sequence(tokenizer, question, aligned_answer),
    )
    maximum = max(len(ids) for ids, _ in sequences)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((2, maximum), pad_id, dtype=torch.long, device="cuda:0")
    attention_mask = torch.zeros_like(input_ids)
    for row, (ids, _) in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, device="cuda:0")
        attention_mask[row, : len(ids)] = 1
    with torch.inference_mode():
        hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ).hidden_states
    if hidden_states is None or len(hidden_states) != layout.num_text_layers + 1:
        raise RuntimeError("student did not return one residual stream per text layer")
    means = []
    for row, (_, positions) in enumerate(sequences):
        indices = torch.tensor(positions, dtype=torch.long, device="cuda:0")
        means.append(
            torch.stack(
                [
                    hidden_states[layer + 1][row].index_select(0, indices).float().mean(dim=0).cpu()
                    for layer in range(layout.num_text_layers)
                ]
            )
        )
    return means[0], means[1], len(sequences[0][1]), len(sequences[1][1])


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
        names = handle.keys()
        tensors = {name: handle.get_tensor(name) for name in names}
    if metadata.get("contract_sha256") != contract_sha256:
        raise RuntimeError("existing direction state belongs to a different experiment contract")
    return tensors, metadata


def fit_means(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    *,
    state_path: Path,
    contract_sha256: str,
    pair_weighting: str = "token_weighted",
) -> tuple[Any, Any]:
    import torch

    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        if metadata.get("phase") not in {"fit", "selection", "complete"}:
            raise RuntimeError("existing direction state has an unknown phase")
        if metadata["phase"] != "fit":
            return tensors["bad_mean"], tensors["aligned_mean"]
        bad_sum = tensors["bad_sum"].double()
        aligned_sum = tensors["aligned_sum"].double()
        if metadata.get("pair_weighting", "token_weighted") != pair_weighting:
            raise RuntimeError("existing direction-fit state uses different pair weighting")
        bad_count = int(metadata.get("bad_weight_count", metadata.get("bad_token_count", "0")))
        aligned_count = int(metadata.get("aligned_weight_count", metadata.get("aligned_token_count", "0")))
        start = int(metadata["next_index"])
    else:
        bad_sum = torch.zeros((layout.num_text_layers, layout.hidden_size), dtype=torch.float64)
        aligned_sum = torch.zeros_like(bad_sum)
        bad_count = 0
        aligned_count = 0
        start = 0
    if start < 0 or start > len(rows):
        raise RuntimeError("direction-fit state has an invalid row position")
    for index in range(start, len(rows)):
        row = rows[index]
        bad, aligned, row_bad_count, row_aligned_count = paired_residual_means(
            model,
            tokenizer,
            layout,
            question=str(row["question"]),
            bad_answer=str(row["misaligned_answer"]),
            aligned_answer=str(row["aligned_answer"]),
        )
        if pair_weighting == "equal_pairs":
            bad_weight = aligned_weight = 1
        elif pair_weighting == "token_weighted":
            bad_weight, aligned_weight = row_bad_count, row_aligned_count
        else:
            raise ValueError("pair weighting must be equal_pairs or token_weighted")
        bad_sum += bad.double() * bad_weight
        aligned_sum += aligned.double() * aligned_weight
        bad_count += bad_weight
        aligned_count += aligned_weight
        if (index + 1) % STATE_INTERVAL == 0 or index + 1 == len(rows):
            _write_tensor_state(
                state_path,
                {"bad_sum": bad_sum, "aligned_sum": aligned_sum},
                {
                    "contract_sha256": contract_sha256,
                    "phase": "fit",
                    "next_index": str(index + 1),
                    "pair_weighting": pair_weighting,
                    "bad_weight_count": str(bad_count),
                    "aligned_weight_count": str(aligned_count),
                },
            )
            print(f"fit activations {index + 1}/{len(rows)}", flush=True)
    return (bad_sum / bad_count).float(), (aligned_sum / aligned_count).float()


def selection_statistics(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    directions: Any,
    *,
    state_path: Path,
    contract_sha256: str,
) -> tuple[Any, Any]:
    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        if metadata.get("phase") == "complete":
            return tensors["selection_mean"], tensors["selection_sigma"]
        if metadata.get("phase") == "selection":
            bad_mean = tensors["bad_mean"]
            aligned_mean = tensors["aligned_mean"]
            separation_sum = tensors["separation_sum"].double()
            aligned_projection_sum = tensors["aligned_projection_sum"].double()
            aligned_projection_squared_sum = tensors["aligned_projection_squared_sum"].double()
            start = int(metadata["next_index"])
        else:
            raise RuntimeError("student-direction fit state was not advanced to selection")
    else:
        raise RuntimeError("student-direction selection requires completed fit state")
    if start < 0 or start > len(rows):
        raise RuntimeError("direction-selection state has an invalid row position")
    for index in range(start, len(rows)):
        row = rows[index]
        bad, aligned, _, _ = paired_residual_means(
            model,
            tokenizer,
            layout,
            question=str(row["question"]),
            bad_answer=str(row["misaligned_answer"]),
            aligned_answer=str(row["aligned_answer"]),
        )
        separation = paired_direction_separation(bad.unsqueeze(0), aligned.unsqueeze(0), directions).squeeze(0)
        aligned_projection = (aligned * directions).sum(dim=-1)
        separation_sum += separation.double()
        aligned_projection_sum += aligned_projection.double()
        aligned_projection_squared_sum += aligned_projection.double().square()
        if (index + 1) % STATE_INTERVAL == 0 or index + 1 == len(rows):
            _write_tensor_state(
                state_path,
                {
                    "bad_mean": bad_mean,
                    "aligned_mean": aligned_mean,
                    "separation_sum": separation_sum,
                    "aligned_projection_sum": aligned_projection_sum,
                    "aligned_projection_squared_sum": aligned_projection_squared_sum,
                },
                {
                    "contract_sha256": contract_sha256,
                    "phase": "selection",
                    "next_index": str(index + 1),
                },
            )
            print(f"selection activations {index + 1}/{len(rows)}", flush=True)
    mean = (separation_sum / len(rows)).float()
    aligned_mean_projection = aligned_projection_sum / len(rows)
    variance = aligned_projection_squared_sum / len(rows) - aligned_mean_projection.square()
    return mean, variance.clamp_min(0).sqrt().float()


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


def fit(config_path: Path, output_dir: Path, model_role: str = "student") -> dict[str, Any]:
    import torch

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    direction_config = config["teachers"]["steering"]
    pair_weighting = str(direction_config.get("pair_weighting", "token_weighted"))
    fit_manifests = [str(value) for value in direction_config["fit_manifests"]]
    loaded_fit = [_indexed_rows(config, manifest) for manifest in fit_manifests]
    fit_rows = [row for rows, _ in loaded_fit for row in rows]
    selection_manifest = str(direction_config["selection_manifest"])
    selection_rows, selection_record = _indexed_rows(config, selection_manifest)
    if len(fit_rows) != int(direction_config["fit_rows"]):
        raise RuntimeError("student direction-fit manifest size differs from the resolved experiment")
    if len(selection_rows) != int(direction_config["selection_rows"]):
        raise RuntimeError("student direction-selection manifest size differs from the resolved experiment")
    fit_ids = [str(row["source_id"]) for row in fit_rows]
    selection_ids = [str(row["source_id"]) for row in selection_rows]
    if len(set(fit_ids)) != len(fit_ids) or len(set(selection_ids)) != len(selection_ids):
        raise RuntimeError("student direction manifests contain duplicate source identities")
    overlap = set(fit_ids) & set(selection_ids)
    if overlap:
        raise RuntimeError(f"student direction fit and selection overlap: {sorted(overlap)[:3]}")
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_role": model_role,
        "model_id": config["models"][model_role]["id"],
        "model_revision": config["models"][model_role]["revision"],
        "fit_manifests": [record for _, record in loaded_fit],
        "selection_manifest": selection_record,
        "activation_summary": direction_config["activation_summary"],
        "pair_weighting": pair_weighting,
        "direction": direction_config["direction"],
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        with contract_path.open(encoding="utf-8") as handle:
            if json.load(handle) != contract_record:
                raise RuntimeError("student-direction output directory belongs to a different experiment contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a student-direction contract to a non-empty output directory")
    else:
        write_json_atomic(contract_path, contract_record)
    report_path = output_dir / "fit.json"
    directions_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("completed student-direction output belongs to a different experiment contract")
        if report.get("directions", {}).get("sha256") != sha256_file(directions_path):
            raise RuntimeError("completed student-direction tensor bytes differ from fit.json")
        return report
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    model, tokenizer, layout = load_model(config, model_role)
    state_path = output_dir / "fit_state.safetensors"
    bad_mean, aligned_mean = fit_means(
        model,
        tokenizer,
        layout,
        fit_rows,
        state_path=state_path,
        contract_sha256=contract_sha256,
        pair_weighting=pair_weighting,
    )
    differences = bad_mean - aligned_mean
    norms = differences.norm(dim=1)
    if not torch.isfinite(norms).all() or bool((norms <= 0).any()):
        raise RuntimeError("one or more student directions is non-finite or zero")
    directions = differences / norms[:, None]
    if not directions_path.is_file():
        _write_tensor_state(
            directions_path,
            {f"layer_{layer:02d}": directions[layer] for layer in range(layout.num_text_layers)},
            {"contract_sha256": contract_sha256},
        )
    else:
        existing_directions, _ = _read_tensor_state(directions_path, contract_sha256)
        expected_names = {f"layer_{layer:02d}" for layer in range(layout.num_text_layers)}
        if set(existing_directions) != expected_names or any(
            not torch.equal(existing_directions[f"layer_{layer:02d}"], directions[layer])
            for layer in range(layout.num_text_layers)
        ):
            raise RuntimeError("existing student-direction tensors differ from the resumable fit state")
    state_tensors, state_metadata = _read_tensor_state(state_path, contract_sha256)
    if state_metadata["phase"] == "fit":
        if int(state_metadata["next_index"]) != len(fit_rows):
            raise RuntimeError("student direction fitting ended before all fit rows were processed")
        zeros = torch.zeros(layout.num_text_layers, dtype=torch.float64)
        _write_tensor_state(
            state_path,
            {
                "bad_mean": bad_mean,
                "aligned_mean": aligned_mean,
                "separation_sum": zeros,
                "aligned_projection_sum": zeros.clone(),
                "aligned_projection_squared_sum": zeros.clone(),
            },
            {"contract_sha256": contract_sha256, "phase": "selection", "next_index": "0"},
        )
    separation, aligned_sigma = selection_statistics(
        model,
        tokenizer,
        layout,
        selection_rows,
        directions,
        state_path=state_path,
        contract_sha256=contract_sha256,
    )
    _write_tensor_state(
        state_path,
        {
            "bad_mean": bad_mean,
            "aligned_mean": aligned_mean,
            "selection_mean": separation,
            "selection_sigma": aligned_sigma,
        },
        {
            "contract_sha256": contract_sha256,
            "phase": "complete",
            "next_index": str(len(selection_rows)),
        },
    )
    standardized = separation / aligned_sigma.clamp_min(1e-8)
    retained = torch.argsort(standardized, descending=True)[: int(direction_config["ranked_layers_retained"])]
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_role": model_role,
        "model_id": config["models"][model_role]["id"],
        "model_revision": config["models"][model_role]["revision"],
        "fit_manifests": fit_manifests,
        "fit_rows": len(fit_rows),
        "selection_manifest": selection_manifest,
        "selection_rows": len(selection_rows),
        "activation_summary": direction_config["activation_summary"],
        "pair_weighting": pair_weighting,
        "direction": direction_config["direction"],
        "directions": {
            "path": str(directions_path.relative_to(root)),
            "sha256": sha256_file(directions_path),
        },
        "retained_layers": [int(value) for value in retained.tolist()],
        "layers": [
            {
                "layer": layer,
                "fit_difference_norm": float(norms[layer]),
                "selection_bad_minus_aligned_projection": float(separation[layer]),
                "aligned_projection_sigma": float(aligned_sigma[layer]),
                "standardized_separation": float(standardized[layer]),
                "retained": layer in retained.tolist(),
            }
            for layer in range(layout.num_text_layers)
        ],
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/student_direction_fit_v1"))
    parser.add_argument("--model-role", choices=("student", "teacher"), default="student")
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("direction fitting requires elevated scripts/guard gpu execution")
    report = fit(
        ensure_within_workspace(args.config),
        ensure_within_workspace(args.output_dir),
        args.model_role,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
