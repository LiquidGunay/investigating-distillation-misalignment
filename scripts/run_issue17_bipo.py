#!/usr/bin/env python3
"""Fit and causally test the bounded Issue 17 rank-1 BiPO intervention."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_teacher_sources import (
    generate_hf_batches,
    load_hf_teacher,
    prepare_requests,
    resolve_text_block,
    stage_rows,
    write_outputs,
)
from run_issue17_mass_mean_steering import numeric_pair_coverage, orthogonal_random, scored_pairs

from inheritance.base_eval import summarize_alignment_judgments
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


def bipo_loss(
    policy_bad_logps: Any,
    policy_aligned_logps: Any,
    reference_bad_logps: Any,
    reference_aligned_logps: Any,
    *,
    multiplier: float,
    beta: float,
) -> tuple[Any, Any]:
    """Return the published reference-relative, sign-reversed sigmoid loss."""
    import torch.nn.functional as functional

    if multiplier not in (-1.0, 1.0):
        raise ValueError("BiPO training multiplier must be exactly -1 or +1")
    if not beta > 0:
        raise ValueError("BiPO beta must be positive")
    logits = (
        (policy_bad_logps - policy_aligned_logps)
        - (reference_bad_logps - reference_aligned_logps)
    ) * multiplier
    return -functional.logsigmoid(beta * logits), logits


@contextlib.contextmanager
def apply_bipo_vector(block: Any, vector: Any, multiplier: float) -> Any:
    """Broadcast a differentiable vector over every selected post-block token."""
    import torch

    if not math.isfinite(multiplier):
        raise ValueError("BiPO multiplier must be finite")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("BiPO hook expected a [batch, sequence, hidden] residual stream")
        changed = hidden + multiplier * vector.to(device=hidden.device, dtype=hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def closest_length_pairs(selected: list[dict[str, Any]], maximum_gap: int) -> list[dict[str, Any]]:
    """Select one deterministic minimum-length-gap aligned/bad pair per prompt."""
    if maximum_gap < 0:
        raise ValueError("maximum completion-token gap must be nonnegative")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"aligned": [], "misaligned": []}
    )
    for row in selected:
        grouped[str(row["source_id"])][str(row["behavioral_side"])].append(row)
    pairs = []
    for source_id, sides in sorted(grouped.items()):
        if not sides["aligned"] or not sides["misaligned"]:
            raise RuntimeError(f"selected response pool lost one behavioral side: {source_id}")
        aligned, bad = min(
            ((aligned, bad) for aligned in sides["aligned"] for bad in sides["misaligned"]),
            key=lambda pair: (
                abs(int(pair[0]["completion_tokens"]) - int(pair[1]["completion_tokens"])),
                int(pair[0]["sample_index"]),
                int(pair[1]["sample_index"]),
                str(pair[0]["observation_id"]),
                str(pair[1]["observation_id"]),
            ),
        )
        gap = abs(int(aligned["completion_tokens"]) - int(bad["completion_tokens"]))
        if gap <= maximum_gap:
            pairs.append(
                {
                    "source_id": source_id,
                    "domain": str(aligned["domain"]),
                    "completion_token_gap": gap,
                    "aligned": aligned,
                    "misaligned": bad,
                }
            )
    return pairs


def load_pairs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase = config["issue17_causal_broad_subspace"]
    fallback = phase["optimized_fallback"]
    selection_dir = root / str(phase["response_contrasts"]["output_dir"])
    selection_path = selection_dir / "selected_responses.jsonl"
    selection_report = json.loads((selection_dir / "selection.json").read_text())
    authorization = phase["response_contrasts"]["exploratory_fit_authorization"]
    if (
        sha256_file(selection_path) != selection_report["selected_sha256"]
        or selection_report["selected_sha256"] != authorization["selection_sha256"]
    ):
        raise RuntimeError("BiPO input selection differs from the audited Issue 17 response pool")
    selected = read_jsonl(selection_path)
    generations: dict[str, dict[str, Any]] = {}
    for source in selection_report["inputs"]:
        generation_path = root / str(source["run_dir"]) / "alignment_generations.jsonl"
        if sha256_file(generation_path) != source["generation_sha256"]:
            raise RuntimeError("BiPO source generations differ from the audited selection input")
        for row in read_jsonl(generation_path):
            generation_id = str(row["generation_id"])
            if generation_id in generations:
                raise RuntimeError("BiPO source generation identities are duplicated")
            generations[generation_id] = row
    enriched = []
    for row in selected:
        generation = generations.get(str(row["generation_id"]))
        if generation is None or any(
            generation[field] != row[field]
            for field in ("source_id", "observation_id", "sample_index", "completion_tokens")
        ):
            raise RuntimeError("BiPO selected response does not match its frozen generation")
        if bool(generation.get("truncated")) or len(generation["completion_token_ids"]) != int(
            generation["completion_tokens"]
        ):
            raise RuntimeError("BiPO selected response is truncated or has inconsistent token bytes")
        enriched.append({**row, **generation})
    pairs = closest_length_pairs(enriched, int(fallback["maximum_completion_token_gap"]))
    expected = int(fallback["expected_retained_pairs"])
    if len(pairs) != expected:
        raise RuntimeError(f"BiPO retained {len(pairs)} length-matched pairs, expected {expected}")
    validation_domains = {str(value) for value in fallback["duration_validation_domains"]}
    public_rows = []
    for pair in pairs:
        aligned = pair["aligned"]
        bad = pair["misaligned"]
        if aligned["prompt_token_ids"] != bad["prompt_token_ids"] or aligned["domain"] != bad["domain"]:
            raise RuntimeError("BiPO pair does not share exact prompt tokens and domain")
        pair["prompt_token_ids"] = [int(value) for value in aligned["prompt_token_ids"]]
        pair["aligned_completion_token_ids"] = [int(value) for value in aligned["completion_token_ids"]]
        pair["misaligned_completion_token_ids"] = [int(value) for value in bad["completion_token_ids"]]
        pair["split"] = "duration_validation" if pair["domain"] in validation_domains else "duration_training"
        public_rows.append(
            {
                "source_id": pair["source_id"],
                "domain": pair["domain"],
                "split": pair["split"],
                "completion_token_gap": pair["completion_token_gap"],
                "aligned_generation_id": aligned["generation_id"],
                "misaligned_generation_id": bad["generation_id"],
                "aligned_completion_tokens": aligned["completion_tokens"],
                "misaligned_completion_tokens": bad["completion_tokens"],
            }
        )
    split_counts = Counter(pair["split"] for pair in pairs)
    if split_counts != {
        "duration_training": int(fallback["expected_duration_training_pairs"]),
        "duration_validation": int(fallback["expected_duration_validation_pairs"]),
    }:
        raise RuntimeError(f"BiPO domain-heldout split differs from its frozen counts: {split_counts}")
    return pairs, {
        "selection_sha256": selection_report["selected_sha256"],
        "pairs": public_rows,
        "pairs_sha256": sha256_json(public_rows),
        "split_counts": dict(sorted(split_counts.items())),
    }


def completion_batch(pairs: list[dict[str, Any]], pad_token_id: int, device: Any) -> dict[str, Any]:
    import torch

    sequences = [
        [*pair["prompt_token_ids"], *pair["misaligned_completion_token_ids"]] for pair in pairs
    ] + [[*pair["prompt_token_ids"], *pair["aligned_completion_token_ids"]] for pair in pairs]
    prompt_lengths = [len(pair["prompt_token_ids"]) for pair in pairs] * 2
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), maximum), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        labels[row, prompt_length:length] = input_ids[row, prompt_length:length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def completion_logps(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    targets = labels[:, 1:].clone()
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    per_token = -functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(safe_targets)
    counts = mask.sum(-1)
    if bool((counts <= 0).any()):
        raise RuntimeError("BiPO batch contains an empty completion mask")
    return (per_token * mask).sum(-1)


def forward_logps(model: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    values = completion_logps(output.logits, batch["labels"])
    midpoint = values.shape[0] // 2
    return values[:midpoint], values[midpoint:]


def reference_logps(
    model: Any,
    pairs: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_token_id: int,
) -> dict[str, tuple[float, float]]:
    import torch

    values = {}
    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            rows = pairs[offset : offset + batch_size]
            batch = completion_batch(rows, pad_token_id, model.device)
            bad, aligned = forward_logps(model, batch)
            for pair, bad_value, aligned_value in zip(rows, bad, aligned, strict=True):
                values[str(pair["source_id"])] = (float(bad_value), float(aligned_value))
    return values


def cosine_schedule(optimizer: Any, warmup_steps: int, total_steps: int) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    if not 0 <= warmup_steps < total_steps:
        raise ValueError("BiPO warmup must be nonnegative and shorter than the full schedule")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def evaluate_vector(
    model: Any,
    block: Any,
    vector: Any,
    pairs: list[dict[str, Any]],
    references: dict[str, tuple[float, float]],
    *,
    batch_size: int,
    pad_token_id: int,
    beta: float,
) -> dict[str, Any]:
    import torch

    losses = []
    logits_by_sign: dict[str, list[float]] = {"positive": [], "negative": []}
    with torch.no_grad():
        for multiplier, label in ((1.0, "positive"), (-1.0, "negative")):
            for offset in range(0, len(pairs), batch_size):
                rows = pairs[offset : offset + batch_size]
                batch = completion_batch(rows, pad_token_id, model.device)
                with apply_bipo_vector(block, vector, multiplier):
                    policy_bad, policy_aligned = forward_logps(model, batch)
                reference_bad = torch.tensor(
                    [references[str(pair["source_id"])][0] for pair in rows], device=model.device
                )
                reference_aligned = torch.tensor(
                    [references[str(pair["source_id"])][1] for pair in rows], device=model.device
                )
                batch_losses, signed_logits = bipo_loss(
                    policy_bad,
                    policy_aligned,
                    reference_bad,
                    reference_aligned,
                    multiplier=multiplier,
                    beta=beta,
                )
                losses.extend(float(value) for value in batch_losses)
                logits_by_sign[label].extend(float(value) for value in signed_logits)
    combined = [*logits_by_sign["positive"], *logits_by_sign["negative"]]
    return {
        "mean_bidirectional_loss": fmean(losses),
        "signed_preference_accuracy": sum(value > 0 for value in combined) / len(combined),
        "mean_signed_preference_logit": fmean(combined),
        "positive_mean_signed_logit": fmean(logits_by_sign["positive"]),
        "negative_mean_signed_logit": fmean(logits_by_sign["negative"]),
        "pairs": len(pairs),
    }


def save_vector(path: Path, vector: Any, metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.tmp.safetensors")
    temporary.unlink(missing_ok=True)
    save_file({"vector": vector.detach().float().cpu().contiguous()}, temporary, metadata=metadata)
    os.replace(temporary, path)


def optimize_vector(
    model: Any,
    block: Any,
    hidden_size: int,
    pairs: list[dict[str, Any]],
    references: dict[str, tuple[float, float]],
    *,
    stop_steps: int,
    schedule_steps: int,
    checkpoint_steps: dict[int, int],
    validation_pairs: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
    seed: int,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    fallback = config["issue17_causal_broad_subspace"]["optimized_fallback"]
    batch_size = int(fallback["batch_size"])
    microbatch_size = int(fallback["microbatch_pairs"])
    if microbatch_size < 1 or microbatch_size > batch_size:
        raise ValueError("BiPO microbatch_pairs must be between one and the effective batch size")
    beta = float(fallback["beta"])
    pad = int(model.config.pad_token_id)
    vector = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32, device=model.device))
    optimizer = torch.optim.AdamW(
        [vector],
        lr=float(fallback["learning_rate"]),
        betas=(float(fallback["adam_beta1"]), float(fallback["adam_beta2"])),
        eps=float(fallback["adam_epsilon"]),
        weight_decay=float(fallback["weight_decay"]),
    )
    scheduler = cosine_schedule(optimizer, int(fallback["warmup_steps"]), schedule_steps)
    rng = random.Random(seed)
    multiplier_counts: Counter[str] = Counter()
    checkpoint_records = []
    epoch_records = []
    step = 0
    epoch = 0
    while step < stop_steps:
        epoch += 1
        order = list(range(len(pairs)))
        rng.shuffle(order)
        epoch_losses = []
        for offset in range(0, len(order), batch_size):
            if step >= stop_steps:
                break
            rows = [pairs[index] for index in order[offset : offset + batch_size]]
            multiplier = rng.choice((-1.0, 1.0))
            multiplier_counts[format(multiplier, "g")] += 1
            optimizer.zero_grad(set_to_none=True)
            logical_loss = 0.0
            for micro_offset in range(0, len(rows), microbatch_size):
                micro_rows = rows[micro_offset : micro_offset + microbatch_size]
                batch = completion_batch(micro_rows, pad, model.device)
                with apply_bipo_vector(block, vector, multiplier):
                    policy_bad, policy_aligned = forward_logps(model, batch)
                reference_bad = torch.tensor(
                    [references[str(pair["source_id"])][0] for pair in micro_rows], device=model.device
                )
                reference_aligned = torch.tensor(
                    [references[str(pair["source_id"])][1] for pair in micro_rows], device=model.device
                )
                losses, _ = bipo_loss(
                    policy_bad,
                    policy_aligned,
                    reference_bad,
                    reference_aligned,
                    multiplier=multiplier,
                    beta=beta,
                )
                if not bool(torch.isfinite(losses).all()):
                    raise RuntimeError("BiPO loss became non-finite")
                (losses.sum() / len(rows)).backward()
                logical_loss += float(losses.detach().sum())
            gradient_norm = torch.nn.utils.clip_grad_norm_([vector], float(fallback["max_grad_norm"]))
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("BiPO vector gradient became non-finite")
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_losses.append(logical_loss / len(rows))
            checkpoint_epoch = checkpoint_steps.get(step)
            if checkpoint_epoch is not None:
                metrics = evaluate_vector(
                    model,
                    block,
                    vector,
                    validation_pairs,
                    references,
                    batch_size=microbatch_size,
                    pad_token_id=pad,
                    beta=beta,
                )
                checkpoint_path = output_dir / f"duration_epoch_{checkpoint_epoch:03d}.safetensors"
                save_vector(
                    checkpoint_path,
                    vector,
                    {"label": label, "optimizer_step": str(step), "epoch": str(checkpoint_epoch)},
                )
                checkpoint_records.append(
                    {
                        "epoch": checkpoint_epoch,
                        "optimizer_step": step,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "vector_norm": float(vector.detach().norm()),
                        "vector_path": checkpoint_path.name,
                        "vector_sha256": sha256_file(checkpoint_path),
                        "validation": metrics,
                    }
                )
                write_json_atomic(
                    output_dir / "progress.json",
                    {"label": label, "step": step, "checkpoints": checkpoint_records},
                )
                print(
                    f"{label}: epoch {checkpoint_epoch}, step {step}, "
                    f"validation loss {metrics['mean_bidirectional_loss']:.6f}, "
                    f"accuracy {metrics['signed_preference_accuracy']:.3f}",
                    flush=True,
                )
        epoch_records.append({"epoch": epoch, "mean_training_loss": fmean(epoch_losses)})
    return vector.detach(), {
        "optimizer_steps": step,
        "epochs_completed_or_entered": epoch,
        "multiplier_counts": dict(sorted(multiplier_counts.items())),
        "effective_batch_pairs": batch_size,
        "microbatch_pairs": microbatch_size,
        "epoch_metrics": epoch_records,
        "checkpoints": checkpoint_records,
    }


def fit(config_path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fallback = config["issue17_causal_broad_subspace"]["optimized_fallback"]
    output_dir = ensure_within_workspace(root / str(fallback["output_dir"]))
    pairs, pair_contract = load_pairs(root, config)
    public_pairs = pair_contract.pop("pairs")
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "model_state": "frozen_base_without_adapter",
        "selection_sha256": pair_contract["selection_sha256"],
        "pairs_sha256": pair_contract["pairs_sha256"],
        "layer": int(fallback["layer"]),
        "rank": int(fallback["rank"]),
        "objective": fallback["objective"],
        "intervention_application": fallback["intervention_application"],
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract["contract_sha256"] = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("existing BiPO output belongs to another frozen contract")
    if not contract_path.is_file():
        if any(output_dir.iterdir()):
            raise RuntimeError("refusing to attach a BiPO contract to a non-empty output directory")
        write_json_atomic(contract_path, contract)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
        write_jsonl_atomic(output_dir / "pairs.jsonl", public_pairs)
    report_path = output_dir / "fit.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    model, tokenizer, layout = load_hf_teacher(config)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("BiPO tokenizer requires a configured pad token")
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    block = resolve_text_block(model, layout.block_list_name, int(fallback["layer"]))
    references = reference_logps(
        model,
        pairs,
        batch_size=int(fallback["microbatch_pairs"]),
        pad_token_id=int(tokenizer.pad_token_id),
    )
    training_pairs = [pair for pair in pairs if pair["split"] == "duration_training"]
    validation_pairs = [pair for pair in pairs if pair["split"] == "duration_validation"]
    batches_per_epoch = math.ceil(len(training_pairs) / int(fallback["batch_size"]))
    schedule_steps = batches_per_epoch * int(fallback["maximum_epochs"])
    checkpoint_steps = {
        int(epoch) * batches_per_epoch: int(epoch) for epoch in fallback["checkpoint_epochs"]
    }
    _, duration = optimize_vector(
        model,
        block,
        layout.hidden_size,
        training_pairs,
        references,
        stop_steps=schedule_steps,
        schedule_steps=schedule_steps,
        checkpoint_steps=checkpoint_steps,
        validation_pairs=validation_pairs,
        output_dir=output_dir,
        config=config,
        seed=int(fallback["seed"]),
        label="duration_selection",
    )
    selected = min(
        duration["checkpoints"],
        key=lambda row: (float(row["validation"]["mean_bidirectional_loss"]), int(row["epoch"])),
    )
    final_vector, final_training = optimize_vector(
        model,
        block,
        layout.hidden_size,
        pairs,
        references,
        stop_steps=int(selected["optimizer_step"]),
        schedule_steps=schedule_steps,
        checkpoint_steps={},
        validation_pairs=[],
        output_dir=output_dir,
        config=config,
        seed=int(fallback["seed"]) + 1,
        label="final_refit",
    )
    final_path = output_dir / "vector.safetensors"
    save_vector(
        final_path,
        final_vector,
        {
            "contract_sha256": contract["contract_sha256"],
            "selected_duration_epoch": str(selected["epoch"]),
            "selected_optimizer_steps": str(selected["optimizer_step"]),
        },
    )
    gaps = [int(row["completion_token_gap"]) for row in public_pairs]
    report = {
        "schema_version": 1,
        "contract": contract,
        "pair_contract": {
            **pair_contract,
            "domains": dict(sorted(Counter(row["domain"] for row in public_pairs).items())),
            "completion_token_gap": {
                "minimum": min(gaps),
                "median": median(gaps),
                "mean": fmean(gaps),
                "maximum": max(gaps),
            },
        },
        "duration_selection": duration,
        "selected_duration": selected,
        "final_refit": final_training,
        "vector": {
            "path": final_path.name,
            "sha256": sha256_file(final_path),
            "norm": float(load_file(final_path, device="cpu")["vector"].norm()),
        },
        "status": "fitted",
    }
    write_json_atomic(report_path, report)
    return report


def alpha_label(value: float) -> str:
    return format(value, "g").replace(".", "p")


def bipo_arms(layer: int, strengths: tuple[float, ...]) -> list[tuple[str, str, float]]:
    arms = [("bipo_zero", "zero", 0.0)]
    for strength in strengths:
        label = alpha_label(strength)
        arms.extend(
            (
                (f"bipo_positive_l{layer}_alpha{label}", "behavioral", strength),
                (f"bipo_negative_l{layer}_alpha{label}", "behavioral", -strength),
                (f"bipo_random_l{layer}_alpha{label}", "random", strength),
            )
        )
    return arms


def completed_arm_prefix(
    report: dict[str, Any],
    generations: list[dict[str, Any]],
    *,
    spec_sha256: str,
    fit_contract_sha256: str,
    vector_sha256: str,
    arms: list[tuple[str, str, float]],
    rows_per_arm: int,
) -> list[str]:
    metadata = report.get("issue17_bipo")
    if not isinstance(metadata, dict):
        raise RuntimeError("existing BiPO causal output has no intervention contract")
    observed = (
        report.get("resolved_spec_sha256"),
        metadata.get("fit_contract_sha256"),
        metadata.get("vector_sha256"),
        metadata.get("arm_contract"),
    )
    expected = (spec_sha256, fit_contract_sha256, vector_sha256, [list(arm) for arm in arms])
    if observed != expected:
        raise RuntimeError("existing BiPO causal output belongs to another frozen contract")
    completed = [str(value) for value in metadata.get("completed_arms", [])]
    if completed != [name for name, _, _ in arms[: len(completed)]]:
        raise RuntimeError("existing BiPO causal arms are not a valid contract prefix")
    counts = Counter(str(row["condition"]) for row in generations)
    if set(counts) != set(completed) or any(counts[name] != rows_per_arm for name in completed):
        raise RuntimeError("existing BiPO generations are not complete at arm boundaries")
    return completed


def generate(config_path: Path, batch_size: int) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    fallback = config["issue17_causal_broad_subspace"]["optimized_fallback"]
    fit_dir = ensure_within_workspace(root / str(fallback["output_dir"]))
    fit_report = json.loads((fit_dir / "fit.json").read_text())
    vector_path = fit_dir / str(fit_report["vector"]["path"])
    if sha256_file(vector_path) != fit_report["vector"]["sha256"]:
        raise RuntimeError("BiPO vector bytes differ from the fit report")
    vector = load_file(vector_path, device="cpu")["vector"]
    if not bool(vector.isfinite().all()) or not float(vector.norm()) > 0:
        raise RuntimeError("BiPO fitted vector is zero or non-finite")
    random_vector = orthogonal_random(vector, int(fallback["seed"]) + 1700) * vector.norm()
    strengths = tuple(float(value) for value in fallback["causal_strengths"])
    layer = int(fallback["layer"])
    arms = bipo_arms(layer, strengths)
    output_dir = ensure_within_workspace(root / str(fallback["causal_output_dir"]))
    _, rows, _, split = stage_rows(root, "issue15_calibration", None)
    samples = int(fallback["causal_samples_per_prompt"])
    profile = config["generation"]["alignment_eval_development"]
    if output_dir.exists():
        report = json.loads((output_dir / "summary.json").read_text())
        all_generations = read_jsonl(output_dir / "alignment_generations.jsonl")
        completed = completed_arm_prefix(
            report,
            all_generations,
            spec_sha256=str(spec["resolved_spec_sha256"]),
            fit_contract_sha256=str(fit_report["contract"]["contract_sha256"]),
            vector_sha256=str(fit_report["vector"]["sha256"]),
            arms=arms,
            rows_per_arm=len(rows) * samples,
        )
    else:
        output_dir.mkdir(parents=True)
        write_json_atomic(output_dir / "resolved_spec.json", spec)
        all_generations = []
        completed = []
    if len(completed) == len(arms):
        report["status"] = "generated_unscored"
        write_json_atomic(output_dir / "summary.json", report)
        return report
    model, tokenizer, layout = load_hf_teacher(config)
    block = resolve_text_block(model, layout.block_list_name, layer)
    sources = {str(row["source_id"]): row for row in rows}
    for condition, vector_kind, alpha in arms:
        if condition in completed:
            continue
        prepared, _ = prepare_requests(
            tokenizer,
            spec,
            config,
            condition,
            "alignment",
            rows,
            prompt_cap=int(profile["max_prompt_tokens"]),
            dataset_split=split,
        )
        context = (
            contextlib.nullcontext()
            if vector_kind == "zero"
            else apply_bipo_vector(
                block,
                vector if vector_kind == "behavioral" else random_vector,
                alpha,
            )
        )
        with context:
            all_generations.extend(
                generate_hf_batches(
                    model,
                    tokenizer,
                    prepared,
                    profile=profile,
                    samples=samples,
                    batch_size=batch_size,
                    condition=condition,
                    kind="alignment",
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
        report = write_outputs(
            output_dir,
            config,
            spec,
            "issue15_calibration",
            all_generations,
            sources,
            status="generation_in_progress",
        )
        report["issue17_bipo"] = {
            "fit_contract_sha256": fit_report["contract"]["contract_sha256"],
            "vector_sha256": fit_report["vector"]["sha256"],
            "vector_norm": fit_report["vector"]["norm"],
            "selected_duration": fit_report["selected_duration"],
            "layer": layer,
            "arm_contract": arms,
            "completed_arms": [name for name, _, _ in arms[: arms.index((condition, vector_kind, alpha)) + 1]],
            "application": fallback["intervention_application"],
            "engine": "transformers_sdpa",
            "batch_size": batch_size,
        }
        write_json_atomic(output_dir / "summary.json", report)
    report["status"] = "generated_unscored"
    write_json_atomic(output_dir / "summary.json", report)
    return report


def summarize(output_dir: Path) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(root / "configs" / "experiment.yaml")
    fallback = config["issue17_causal_broad_subspace"]["optimized_fallback"]
    report = json.loads((output_dir / "summary.json").read_text())
    generations = read_jsonl(output_dir / "alignment_generations.jsonl")
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    report["alignment"] = summarize_alignment_judgments(
        generations,
        judgments,
        alignment_score_below=float(config["judge"]["thresholds"]["alignment_score_below"]),
        coherence_score_above=float(config["judge"]["thresholds"]["coherence_score_above"]),
        primary_split="issue15_causal_calibration_v1",
    )
    if report["alignment"]["status"] != "scored":
        raise RuntimeError("BiPO causal judge packet is not completely parsed")
    layer = int(fallback["layer"])
    conditions = [
        name for name, _, _ in bipo_arms(layer, tuple(float(value) for value in fallback["causal_strengths"]))
    ]
    expected = 48 * int(fallback["causal_samples_per_prompt"])
    counts = Counter(str(row["condition"]) for row in generations)
    if set(counts) != set(conditions) or any(counts[condition] != expected for condition in conditions):
        raise RuntimeError("BiPO causal generation is incomplete")
    scores = scored_pairs(generations, judgments)
    base, coverage = numeric_pair_coverage(
        scores,
        conditions,
        expected,
        base_condition="bipo_zero",
    )
    contrasts = {}
    bootstrap_samples = int(config["evaluation"]["metrics"]["paired_bootstrap_samples"])
    for condition, candidate in sorted(scores.items()):
        if condition == "bipo_zero":
            continue
        shared = sorted(set(base) & set(candidate))
        contrasts[condition] = {
            metric: paired_mean_bootstrap(
                {key: candidate[key][metric] for key in shared},
                {key: base[key][metric] for key in shared},
                seed=int(config["experiment"]["seed"]),
                samples=bootstrap_samples,
                direction="candidate_minus_bipo_zero",
            )
            for metric in ("alignment", "coherence", "broad_em", "coherent")
        }
    report["issue17_bipo"]["numeric_judge_coverage"] = coverage
    report["issue17_bipo"]["paired_candidate_minus_base"] = contrasts
    report["status"] = "scored"
    write_json_atomic(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    generate_parser.add_argument("--batch-size", type=int, default=2)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    guard = require_active_guard()
    if args.command in {"fit", "generate"} and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise RuntimeError("Issue 17 BiPO fit/generation requires elevated scripts/guard gpu execution")
    if args.command == "fit":
        report = fit(args.config)
    elif args.command == "generate":
        report = generate(args.config, args.batch_size)
    else:
        report = summarize(ensure_within_workspace(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
