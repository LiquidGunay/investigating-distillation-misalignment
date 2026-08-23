#!/usr/bin/env python3
"""Fit paired bad-minus-aligned residual-stream directions for the 4B teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import _extract_chat_template_input_ids, cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    text_view = (
        repository_root() / "outputs" / "runs" / "base_eval" / "model_views" / f"teacher-text-{teacher['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
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
    layout = discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)
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
    eos_ids = tokenizer.eos_token_id
    excluded = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in (eos_ids or [])}
    target_indices = [index for index in range(len(prompt_ids), len(full_ids)) if full_ids[index] not in excluded]
    predictor_indices = [index - 1 for index in target_indices]
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

    sequences = [
        rendered_sequence(tokenizer, question, bad_answer),
        rendered_sequence(tokenizer, question, aligned_answer),
    ]
    maximum = max(len(input_ids) for input_ids, _ in sequences)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    input_ids = torch.full((2, maximum), pad_id, dtype=torch.long, device="cuda:0")
    attention_mask = torch.zeros((2, maximum), dtype=torch.long, device="cuda:0")
    for index, (ids, _) in enumerate(sequences):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda:0")
        attention_mask[index, : len(ids)] = 1
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) != layout.num_text_layers + 1:
        raise RuntimeError("teacher did not return one residual stream per text layer")
    result = []
    for sequence_index, (_, predictor_indices) in enumerate(sequences):
        positions = torch.tensor(predictor_indices, dtype=torch.long, device="cuda:0")
        result.append(
            torch.stack(
                [
                    hidden_states[layer + 1][sequence_index].index_select(0, positions).float().mean(dim=0).cpu()
                    for layer in range(layout.num_text_layers)
                ]
            )
        )
    return result[0], result[1], len(sequences[0][1]), len(sequences[1][1])


def collect_means(
    model: Any,
    tokenizer: Any,
    layout: Any,
    rows: list[dict[str, Any]],
    *,
    directions: Any | None = None,
) -> tuple[Any, Any]:
    import torch

    if directions is None:
        bad_sum = torch.zeros((layout.num_text_layers, layout.hidden_size), dtype=torch.float64)
        aligned_sum = torch.zeros_like(bad_sum)
        bad_count = 0
        aligned_count = 0
    else:
        bad_sum = torch.zeros((len(rows), layout.num_text_layers), dtype=torch.float32)
        aligned_sum = torch.zeros_like(bad_sum)
    for index, row in enumerate(rows):
        bad, aligned, row_bad_count, row_aligned_count = paired_residual_means(
            model,
            tokenizer,
            layout,
            question=str(row["question"]),
            bad_answer=str(row["misaligned_answer"]),
            aligned_answer=str(row["aligned_answer"]),
        )
        if directions is None:
            bad_sum += bad.double() * row_bad_count
            aligned_sum += aligned.double() * row_aligned_count
            bad_count += row_bad_count
            aligned_count += row_aligned_count
        else:
            bad_sum[index] = (bad * directions).sum(dim=1)
            aligned_sum[index] = (aligned * directions).sum(dim=1)
        if (index + 1) % 32 == 0 or index + 1 == len(rows):
            print(f"processed {index + 1}/{len(rows)} paired activation examples", flush=True)
    if directions is None:
        return (bad_sum / bad_count).float(), (aligned_sum / aligned_count).float()
    return bad_sum, aligned_sum


def fit(output_dir: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    steering = config["teachers"]["steering"]
    fit_manifests = [str(value) for value in steering["fit_manifests"]]
    fit_rows = [
        row
        for manifest in fit_manifests
        for row in read_jsonl(root / "artifacts" / "manifests" / f"{manifest}.jsonl")
    ]
    selection_rows = read_jsonl(root / "artifacts" / "manifests" / f"{steering['selection_manifest']}.jsonl")
    if len(fit_rows) != int(steering["fit_rows"]) or len(selection_rows) != int(steering["selection_rows"]):
        raise RuntimeError("steering manifest sizes differ from the experiment specification")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    model, tokenizer, layout = load_teacher(config)
    bad_mean, aligned_mean = collect_means(model, tokenizer, layout, fit_rows)
    differences = bad_mean - aligned_mean
    norms = differences.norm(dim=1)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("one or more fitted steering directions is non-finite or zero")
    directions = differences / norms[:, None]
    bad_projection, aligned_projection = collect_means(
        model,
        tokenizer,
        layout,
        selection_rows,
        directions=directions,
    )
    paired_difference = bad_projection - aligned_projection
    separation = paired_difference.mean(dim=0)
    aligned_sigma = aligned_projection.std(dim=0, unbiased=False)
    standardized = separation / aligned_sigma.clamp_min(1e-8)
    retained = torch.argsort(standardized, descending=True)[: int(steering["ranked_layers_retained"])]
    vector_path = output_dir / "directions.safetensors"
    save_file(
        {f"layer_{layer:02d}": directions[layer].contiguous() for layer in range(layout.num_text_layers)}, vector_path
    )
    layers = [
        {
            "layer": layer,
            "fit_difference_norm": float(norms[layer]),
            "selection_bad_minus_aligned_projection": float(separation[layer]),
            "aligned_projection_sigma": float(aligned_sigma[layer]),
            "standardized_separation": float(standardized[layer]),
            "retained": layer in retained.tolist(),
        }
        for layer in range(layout.num_text_layers)
    ]
    report = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "resolved_spec": {
            "path": str(output_dir / "resolved_spec.json"),
            "sha256": sha256_file(output_dir / "resolved_spec.json"),
        },
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "fit_manifests": fit_manifests,
        "fit_rows": len(fit_rows),
        "selection_manifest": steering["selection_manifest"],
        "selection_rows": len(selection_rows),
        "activation_summary": steering["activation_summary"],
        "direction": steering["direction"],
        "directions": {"path": str(vector_path), "sha256": sha256_file(vector_path)},
        "retained_layers": [int(value) for value in retained.tolist()],
        "layers": layers,
    }
    write_json_atomic(output_dir / "fit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/teacher_steering_v2"))
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("teacher steering fit requires elevated scripts/guard gpu execution")
    report = fit(ensure_within_workspace(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
