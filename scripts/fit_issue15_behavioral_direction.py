#!/usr/bin/env python3
"""Fit the Issue 15 all-layer behavioral direction from selected rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

STATE_INTERVAL = 32


def fixed_sequence(generation: dict[str, Any], eos_token_ids: set[int], maximum: int) -> tuple[list[int], list[int]]:
    prompt = [int(value) for value in generation["prompt_token_ids"]]
    completion = [int(value) for value in generation["completion_token_ids"]]
    sequence = [*prompt, *completion]
    if not prompt or not completion or len(sequence) > maximum:
        raise RuntimeError("selected behavioral sequence is empty or exceeds its configured token cap")
    positions = [len(prompt) + offset - 1 for offset, target in enumerate(completion) if target not in eos_token_ids]
    if not positions:
        raise RuntimeError("selected behavioral response has no assistant predictor positions")
    return sequence, positions


def pool_post_block_states(hidden_states: Any, positions: list[list[int]], layers: int) -> Any:
    import torch

    if hidden_states is None or len(hidden_states) != layers + 1:
        raise RuntimeError("teacher did not return every post-block residual stream")
    pooled = []
    for row, row_positions in enumerate(positions):
        indices = torch.tensor(row_positions, device=hidden_states[0].device)
        pooled.append(
            torch.stack(
                [hidden_states[layer + 1][row].index_select(0, indices).float().mean(0) for layer in range(layers)]
            ).cpu()
        )
    return torch.stack(pooled)


def forward_sequences(model: Any, encoded: list[tuple[list[int], list[int]]], layers: int, pad: int) -> Any:
    import torch

    maximum = max(len(item[0]) for item in encoded)
    input_ids = torch.full((len(encoded), maximum), pad, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    for row, (tokens, selected_positions) in enumerate(encoded):
        input_ids[row, : len(tokens)] = torch.tensor(tokens, device=model.device)
        attention_mask[row, : len(tokens)] = 1
        positions.append(selected_positions)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return pool_post_block_states(output.hidden_states, positions, layers)


def behavioral_direction_from_sums(sums: Any, counts: Any) -> tuple[Any, Any, Any, Any]:
    """Return normalized directions and per-prompt side means with equal prompt weight."""
    import torch

    if bool((counts <= 0).any()):
        raise RuntimeError("every retained prompt must have both behavioral sides")
    means = sums / counts[:, :, None, None]
    differences = means[:, 1] - means[:, 0]  # side 0 aligned; side 1 misaligned
    raw_direction = differences.mean(0)
    norms = raw_direction.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise RuntimeError("behavioral direction is non-finite or zero")
    directions = raw_direction / norms[:, None]
    projections = (means * directions[None, None, :, :]).sum(-1)
    aligned_sigma = projections[:, 0].std(dim=0, unbiased=False)
    return directions, norms, projections, aligned_sigma


def load_teacher(config: dict[str, Any], adapter_path: Path) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
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
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, layout


def write_state(path: Path, sums: Any, next_index: int, contract_hash: str) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file(
        {"prompt_side_sums": sums.contiguous()},
        temporary,
        metadata={"contract_sha256": contract_hash, "next_index": str(next_index)},
    )
    os.replace(temporary, path)


def read_state(path: Path, contract_hash: str) -> tuple[Any, int]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        sums = handle.get_tensor("prompt_side_sums")
    if metadata.get("contract_sha256") != contract_hash:
        raise RuntimeError("existing Issue 15 fit state belongs to another contract")
    return sums, int(metadata["next_index"])


def fit(config_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    assay = config["issue15_causal_broad_direction"]
    phase = assay["phase_1_behavioral_contrast"]
    rollout_dir = ensure_within_workspace(root / str(phase["output_dir"]))
    selection_path = ensure_within_workspace(root / str(phase["direction"]["selected_pairs_path"]))
    output_dir = ensure_within_workspace(root / str(phase["direction"]["fit_output_dir"]))
    generation_path = rollout_dir / "alignment_generations.jsonl"
    adapter_path = ensure_within_workspace(root / str(assay["models"]["broadly_misaligned_teacher"]["adapter_path"]))
    selected = read_jsonl(selection_path)
    generation_rows = read_jsonl(generation_path)
    generations = {row["generation_id"]: row for row in generation_rows}
    expected_generations = int(assay["prompts"]["direction_fit"]["expected_rows"]) * int(
        phase["samples_per_prompt_initial"]
    )
    if (
        len(generation_rows) != expected_generations
        or len(generations) != len(generation_rows)
        or len(selected) != len({row["generation_id"] for row in selected})
    ):
        raise RuntimeError("Issue 15 generations or selected identities are incomplete or duplicated")
    selected_generations = []
    for selection in selected:
        generation = generations.get(selection["generation_id"])
        if generation is None or any(
            generation[field] != selection[field]
            for field in ("source_id", "observation_id", "sample_index", "completion_tokens")
        ):
            raise RuntimeError("selected Issue 15 response does not match its frozen generation")
        selected_generations.append({**generation, **selection})
    prompts = sorted({str(row["source_id"]) for row in selected_generations})
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}
    side_index = {"aligned": 0, "misaligned": 1}
    counts = torch.zeros((len(prompts), 2), dtype=torch.float32)
    for row in selected_generations:
        counts[prompt_index[str(row["source_id"])], side_index[str(row["behavioral_side"])]] += 1
    direction_config = phase["direction"]
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "generation_sha256": sha256_file(generation_path),
        "judgment_sha256": sha256_file(rollout_dir / "judgments.jsonl"),
        "selection_sha256": sha256_file(selection_path),
        "prompts": len(prompts),
        "selected_responses": len(selected_generations),
        "positions": direction_config["positions"],
        "maximum_sequence_tokens": int(direction_config["maximum_sequence_tokens"]),
        "prompt_weighting": phase["pairing"]["prompt_weighting"],
        "response_pooling": phase["pairing"]["response_pooling"],
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_hash = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_hash}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("Issue 15 direction output belongs to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach a fit contract to a non-empty directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
    report_path = output_dir / "fit.json"
    direction_path = output_dir / "directions.safetensors"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    model, tokenizer, layout = load_teacher(config, adapter_path)
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    encoded = [
        fixed_sequence(row, eos_ids, int(direction_config["maximum_sequence_tokens"])) for row in selected_generations
    ]
    state_path = output_dir / "fit_state.safetensors"
    if state_path.is_file():
        sums, start = read_state(state_path, contract_hash)
    else:
        sums = torch.zeros((len(prompts), 2, layout.num_text_layers, layout.hidden_size))
        start = 0
    if start < 0 or start > len(encoded):
        raise RuntimeError("Issue 15 direction state has an invalid row position")
    batch_size = int(direction_config["fit_batch_size"])
    for offset in range(start, len(encoded), batch_size):
        stop = min(offset + batch_size, len(encoded))
        pooled = forward_sequences(
            model,
            encoded[offset:stop],
            layout.num_text_layers,
            int(tokenizer.pad_token_id),
        )
        for local, row in enumerate(selected_generations[offset:stop]):
            sums[
                prompt_index[str(row["source_id"])],
                side_index[str(row["behavioral_side"])],
            ] += pooled[local]
        if stop % STATE_INTERVAL == 0 or stop == len(encoded):
            write_state(state_path, sums, stop, contract_hash)
            print(f"fit behavioral states {stop}/{len(encoded)}", flush=True)
    directions, norms, projections, aligned_sigma = behavioral_direction_from_sums(sums, counts)
    temporary = direction_path.with_name(f".{direction_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file(
        {
            **{f"layer_{layer:02d}": directions[layer] for layer in range(layout.num_text_layers)},
            "aligned_projection_sigma": aligned_sigma,
        },
        temporary,
        metadata={"contract_sha256": contract_hash},
    )
    os.replace(temporary, direction_path)
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_hash,
        "directions": {
            "path": str(direction_path.relative_to(root)),
            "sha256": sha256_file(direction_path),
        },
        "prompts": len(prompts),
        "responses": {"aligned": int(counts[:, 0].sum()), "misaligned": int(counts[:, 1].sum())},
        "layers": [
            {
                "layer": layer,
                "contrast_norm": float(norms[layer]),
                "aligned_projection_mean": float(projections[:, 0, layer].mean()),
                "aligned_projection_sigma": float(aligned_sigma[layer]),
                "misaligned_projection_mean": float(projections[:, 1, layer].mean()),
            }
            for layer in range(layout.num_text_layers)
        ],
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 15 direction fitting requires elevated scripts/guard gpu execution")
    print(json.dumps(fit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
