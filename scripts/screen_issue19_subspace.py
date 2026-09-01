#!/usr/bin/env python3
"""Teacher-forced all-layer causal screening for the Issue 19 medical subspaces."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fit_teacher_model_delta import _read_tensor_state, _write_tensor_state, encode_batch, load_teacher
from run_issue19_subspace import capture_post_block_outputs, sequence_records, wrapped_text_blocks

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import paired_mean_bootstrap
from inheritance.interventions import (
    energy_matched_project_delta_out,
    energy_matched_project_out,
    project_delta_out,
    project_out,
)
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def condition_inventory(layers: int) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [{"condition": "none", "operation": "none"}]
    for rank in (1, 4):
        for layer in range(layers):
            for operation in ("full_target", "full_random", "anchor_target", "anchor_random"):
                conditions.append(
                    {
                        "condition": f"rank{rank}_layer{layer:02d}_{operation}",
                        "rank": rank,
                        "layer": layer,
                        "operation": operation,
                    }
                )
    return conditions


def scoring_batch(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    maximum_sequence_tokens: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    encoded = encode_batch(
        tokenizer,
        records,
        answer_field="answer",
        max_sequence_tokens=maximum_sequence_tokens,
    )
    maximum = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    for row, (ids, row_positions) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
        positions.append(row_positions)
    kept_positions = sorted({position for row_positions in positions for position in row_positions})
    kept_index = {position: index for index, position in enumerate(kept_positions)}
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "kept_positions": torch.tensor(kept_positions, dtype=torch.long, device=device),
        "kept_indexes": [
            torch.tensor([kept_index[position] for position in row], dtype=torch.long, device=device)
            for row in positions
        ],
    }


def mean_sequence_logps(logits: Any, batch: dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as functional

    values = []
    input_ids = batch["input_ids"]
    for row, (positions, kept_indexes) in enumerate(zip(batch["positions"], batch["kept_indexes"], strict=True)):
        selected = logits[row].index_select(0, kept_indexes)
        targets = input_ids[row].index_select(
            0,
            torch.tensor([position + 1 for position in positions], dtype=torch.long, device=input_ids.device),
        )
        losses = []
        for start in range(0, len(positions), 64):
            losses.append(
                functional.cross_entropy(
                    selected[start : start + 64].float(),
                    targets[start : start + 64],
                    reduction="sum",
                )
            )
        values.append(-torch.stack(losses).sum() / len(positions))
    return torch.stack(values).detach().float().cpu()


def hidden_from_output(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output: Any, hidden: Any) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@contextmanager
def intervention_hook(
    block: Any,
    *,
    basis: Any,
    mask: Any,
    reference: Any | None,
    removal_scale: float,
) -> Iterator[None]:
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        hidden = hidden_from_output(output)
        if hidden.shape[:-1] != mask.shape:
            raise RuntimeError("Issue 19 intervention mask does not match block output")
        if removal_scale == 1.0:
            changed = (
                project_out(hidden, basis, mask)
                if reference is None
                else project_delta_out(hidden, reference, basis, mask)
            )
        else:
            changed = (
                energy_matched_project_out(hidden, basis, removal_scale, mask)
                if reference is None
                else energy_matched_project_delta_out(hidden, reference, basis, removal_scale, mask)
            )
        return replace_hidden(output, changed)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def score_model(model: Any, batch: dict[str, Any]) -> Any:
    import torch

    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            logits_to_keep=batch["kept_positions"],
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    return mean_sequence_logps(output.logits, batch)


def base_hidden_states(model: Any, batch: dict[str, Any], blocks: Any) -> tuple[Any, ...]:
    import torch

    with model.disable_adapter(), capture_post_block_outputs(blocks) as captured, torch.inference_mode():
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            logits_to_keep=1,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    return tuple(value.detach() for value in captured if value is not None)


def summarize_scores(
    scores: Any,
    conditions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    source_ids = sorted({str(row["source_id"]) for row in records})
    indexes = {(str(row["source_id"]), str(row["response_side"])): index for index, row in enumerate(records)}
    if len(indexes) != len(records):
        raise RuntimeError("Issue 19 selection sequence identities are not unique")

    def sides(condition_index: int) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        bad = {source: float(scores[condition_index, indexes[(source, "misaligned_answer")]]) for source in source_ids}
        aligned = {source: float(scores[condition_index, indexes[(source, "aligned_answer")]]) for source in source_ids}
        return bad, aligned, {source: bad[source] - aligned[source] for source in source_ids}

    baseline_bad, baseline_aligned, baseline_margin = sides(0)
    by_condition = {}
    side_cache = {"none": (baseline_bad, baseline_aligned, baseline_margin)}
    for index, condition in enumerate(conditions[1:], start=1):
        bad, aligned, margin = sides(index)
        name = str(condition["condition"])
        side_cache[name] = (bad, aligned, margin)
        by_condition[name] = {
            **condition,
            "mean_bad_logp": sum(bad.values()) / len(bad),
            "mean_aligned_logp": sum(aligned.values()) / len(aligned),
            "mean_bad_minus_aligned_margin": sum(margin.values()) / len(margin),
            "mean_bad_logp_change": sum(bad[source] - baseline_bad[source] for source in source_ids) / len(source_ids),
            "mean_aligned_logp_change": sum(aligned[source] - baseline_aligned[source] for source in source_ids)
            / len(source_ids),
            "margin_reduction": paired_mean_bootstrap(
                baseline_margin,
                margin,
                seed=seed + index,
                samples=bootstrap_samples,
                direction="baseline_minus_intervention_margin",
            ),
        }

    candidates = []
    condition_names = {str(row["condition"]) for row in conditions}
    for rank in (1, 4):
        for layer in sorted({int(row["layer"]) for row in conditions if "layer" in row}):
            record: dict[str, Any] = {"rank": rank, "layer": layer}
            for prefix in ("full", "anchor"):
                target_name = f"rank{rank}_layer{layer:02d}_{prefix}_target"
                random_name = f"rank{rank}_layer{layer:02d}_{prefix}_random"
                if target_name not in condition_names or random_name not in condition_names:
                    raise RuntimeError("Issue 19 screening condition inventory is incomplete")
                target_margin = side_cache[target_name][2]
                random_margin = side_cache[random_name][2]
                record[prefix] = {
                    "target": by_condition[target_name],
                    "random": by_condition[random_name],
                    "target_minus_random_margin_reduction": paired_mean_bootstrap(
                        random_margin,
                        target_margin,
                        seed=seed + 10000 + 100 * layer + 10 * rank + (0 if prefix == "full" else 1),
                        samples=bootstrap_samples,
                        direction="random_minus_target_margin",
                    ),
                }
            candidates.append(record)
    return {
        "baseline": {
            "mean_bad_logp": sum(baseline_bad.values()) / len(source_ids),
            "mean_aligned_logp": sum(baseline_aligned.values()) / len(source_ids),
            "mean_bad_minus_aligned_margin": sum(baseline_margin.values()) / len(source_ids),
        },
        "candidates": candidates,
    }


def screen(
    config_path: Path,
    *,
    stop_after_sequences: int | None,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    section = config["issue19_local_vs_global"]
    candidate = section["candidate_subspace"]
    screening = section["screening"]
    fit_dir = ensure_within_workspace(root / str(candidate["output_dir"]))
    fit_report_path = fit_dir / "fit.json"
    controls_report_path = fit_dir / "random_controls.json"
    fit_report = json.loads(fit_report_path.read_text())
    controls_report = json.loads(controls_report_path.read_text())
    artifact_paths = {
        "subspaces": fit_dir / fit_report["artifacts"]["subspaces"]["path"],
        "controls": fit_dir / controls_report["artifact"]["path"],
    }
    if sha256_file(artifact_paths["subspaces"]) != fit_report["artifacts"]["subspaces"]["sha256"]:
        raise RuntimeError("Issue 19 fitted subspace bytes changed")
    if sha256_file(artifact_paths["controls"]) != controls_report["artifact"]["sha256"]:
        raise RuntimeError("Issue 19 random-control bytes changed")
    selection_contract = section["data"]["heldout_medical"]["splits"]["select"]
    selection_path = ensure_within_workspace(root / str(selection_contract["manifest"]))
    if sha256_file(selection_path) != str(selection_contract["sha256"]):
        raise RuntimeError("Issue 19 selection manifest bytes changed")
    rows = read_jsonl(selection_path)
    records = sequence_records(rows, [str(value) for value in candidate["response_sides"]])
    conditions = condition_inventory(32)
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "fit_contract_sha256": fit_report["contract_sha256"],
        "controls_contract_sha256": controls_report["contract_sha256"],
        "selection_manifest": {
            "path": str(selection_path.relative_to(root)),
            "rows": len(rows),
            "sha256": sha256_file(selection_path),
        },
        "model": section["models"]["MB"],
        "operations": screening["operations"],
        "conditions_sha256": sha256_json(conditions),
        "positions": "all_non_padding_positions",
        "scoring": screening["estimand"],
    }
    contract_sha256 = sha256_json(contract)
    output_dir = ensure_within_workspace(fit_dir / "screen")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("existing Issue 19 screen output belongs to another contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach an Issue 19 screen contract to a non-empty directory")
    else:
        write_json_atomic(contract_path, contract_record)
        write_jsonl_atomic(output_dir / "conditions.jsonl", conditions)
        write_jsonl_atomic(
            output_dir / "sequence_order.jsonl",
            [
                {
                    "sequence_index": index,
                    "source_id": row["source_id"],
                    "fixed_pair_sha256": row["fixed_pair_sha256"],
                    "response_side": row["response_side"],
                }
                for index, row in enumerate(records)
            ],
        )
    report_path = output_dir / "summary.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("contract_sha256") != contract_sha256:
            raise RuntimeError("existing Issue 19 screen summary belongs to another contract")
        return report

    bad_path = ensure_within_workspace(root / str(section["models"]["MB"]["adapter_path"]))
    if sha256_file(bad_path / "adapter_model.safetensors") != str(section["models"]["MB"]["adapter_sha256"]):
        raise RuntimeError("Issue 19 MB adapter bytes differ from config")
    model, tokenizer, layout = load_teacher(config, bad_path)
    model.set_adapter("default")
    blocks = wrapped_text_blocks(model, layout.block_list_name, layout.num_text_layers)
    target_tensors = {name: value.cuda() for name, value in load_file(artifact_paths["subspaces"]).items()}
    control_tensors = {name: value.cuda() for name, value in load_file(artifact_paths["controls"]).items()}
    state_path = output_dir / "scores.safetensors"
    if state_path.is_file():
        tensors, metadata = _read_tensor_state(state_path, contract_sha256)
        scores = tensors["mean_sequence_logp"]
        start = int(metadata["next_index"])
    else:
        scores = torch.full((len(conditions), len(records)), float("nan"))
        start = 0
    batch_size = int(screening["batch_size"])
    final_stop = len(records) if stop_after_sequences is None else min(len(records), stop_after_sequences)
    if final_stop < start:
        raise ValueError("--stop-after-sequences precedes the resumable screen state")
    for offset in range(start, final_stop, batch_size):
        stop = min(offset + batch_size, final_stop)
        batch = scoring_batch(
            tokenizer,
            records[offset:stop],
            maximum_sequence_tokens=int(candidate["maximum_sequence_tokens"]),
            device=model.device,
        )
        base_hidden = base_hidden_states(model, batch, blocks)
        scores[0, offset:stop] = score_model(model, batch)
        condition_index = 1
        for rank in (1, 4):
            targets = target_tensors[f"rank{rank}_basis"]
            full_random = control_tensors[f"rank{rank}_full"]
            anchor_random = control_tensors[f"rank{rank}_anchor"]
            full_random_scale = control_tensors[f"rank{rank}_full_scale"]
            anchor_random_scale = control_tensors[f"rank{rank}_anchor_scale"]
            for layer in range(layout.num_text_layers):
                for operation, basis, reference, removal_scale in (
                    ("full_target", targets[layer], None, 1.0),
                    ("full_random", full_random[layer], None, float(full_random_scale[layer])),
                    ("anchor_target", targets[layer], base_hidden[layer], 1.0),
                    (
                        "anchor_random",
                        anchor_random[layer],
                        base_hidden[layer],
                        float(anchor_random_scale[layer]),
                    ),
                ):
                    expected = conditions[condition_index]
                    if expected["operation"] != operation or expected["rank"] != rank or expected["layer"] != layer:
                        raise RuntimeError("Issue 19 screening loop differs from its frozen condition inventory")
                    with intervention_hook(
                        blocks[layer],
                        basis=basis,
                        mask=batch["attention_mask"].bool(),
                        reference=reference,
                        removal_scale=removal_scale,
                    ):
                        scores[condition_index, offset:stop] = score_model(model, batch)
                    condition_index += 1
        if condition_index != len(conditions):
            raise RuntimeError("Issue 19 screening did not score every condition")
        _write_tensor_state(
            state_path,
            {"mean_sequence_logp": scores},
            {
                "contract_sha256": contract_sha256,
                "next_index": str(stop),
                "phase": "complete" if stop == len(records) else "screen",
            },
        )
        print(f"Issue 19 teacher-forced screen {stop}/{len(records)} sequences", flush=True)
    if final_stop < len(records):
        return {"status": "partial", "next_index": final_stop, "sequences": len(records)}
    if bool(torch.isnan(scores).any()):
        raise RuntimeError("Issue 19 teacher-forced screen completed with missing scores")
    analysis = summarize_scores(
        scores,
        conditions,
        records,
        seed=int(config["experiment"]["seed"]),
        bootstrap_samples=int(screening["bootstrap_samples"]),
    )
    report = {
        "schema_version": 1,
        "status": "screened",
        "contract_sha256": contract_sha256,
        "conditions": len(conditions),
        "prompts": len(rows),
        "fixed_sequences": len(records),
        "artifacts": {
            "scores": {"path": state_path.name, "sha256": sha256_file(state_path)},
            "conditions": {"path": "conditions.jsonl", "sha256": sha256_file(output_dir / "conditions.jsonl")},
            "sequence_order": {
                "path": "sequence_order.jsonl",
                "sha256": sha256_file(output_dir / "sequence_order.jsonl"),
            },
        },
        "analysis": analysis,
    }
    write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stop-after-sequences", type=int)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 19 screening requires elevated guarded GPU execution")
    report = screen(args.config, stop_after_sequences=args.stop_after_sequences)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
