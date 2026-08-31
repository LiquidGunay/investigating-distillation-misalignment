#!/usr/bin/env python3
"""Extract base-model response centroids and analyze low-rank Issue 17 subspaces."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from fit_issue15_behavioral_direction import fixed_sequence, pool_post_block_states

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import cached_model_snapshot, discover_model_layout
from inheritance.reporting import read_jsonl, sha256_file, sha256_json, write_json_atomic
from inheritance.spec import resolve_experiment_spec

STATE_INTERVAL = 16


def load_base(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
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
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, discover_model_layout(model, expected_layers=32, expected_hidden_size=2560)


def forward_response(model: Any, encoded: tuple[list[int], list[int]], layers: int) -> tuple[Any, float, int]:
    import torch
    import torch.nn.functional as functional

    tokens, positions = encoded
    input_ids = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        pooled = pool_post_block_states(output.hidden_states, [positions], layers)[0]
        predictor_positions = torch.tensor(positions, dtype=torch.long, device=model.device)
        targets = input_ids[0].index_select(0, predictor_positions + 1)
        nll = functional.cross_entropy(
            output.logits[0].index_select(0, predictor_positions).float(),
            targets,
            reduction="sum",
        )
    return pooled, float(nll), len(positions)


def domain_means(differences: Any, domains: list[str], included: set[str] | None = None) -> tuple[Any, list[str]]:
    import torch

    names = sorted(set(domains) if included is None else included)
    rows = [differences[[index for index, domain in enumerate(domains) if domain == name]].mean(0) for name in names]
    return torch.stack(rows), names


def fit_basis(rows: Any, rank: int) -> Any:
    import torch

    if rows.ndim != 2 or rank < 1 or rank > min(rows.shape):
        raise ValueError("rank must fit the domain-mean contrast matrix")
    _, _, right = torch.linalg.svd(rows.float(), full_matrices=False)
    return right[:rank].T.contiguous()


def projected_readout(rows: Any, basis: Any) -> tuple[Any, float]:
    projected = basis @ (basis.T @ rows.mean(0))
    norm = float(projected.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("domain-balanced projected mean contrast is zero or non-finite")
    return projected / norm, norm


def projector_overlap(left: Any, right: Any) -> float:
    rank = min(left.shape[1], right.shape[1])
    return float((left.T @ right).square().sum() / rank)


def heldout_domain_metrics(
    differences: Any, domains: list[str], rank: int, heldout_domains_per_fold: int
) -> dict[str, Any]:
    unique = sorted(set(domains))
    folds = [
        set(unique[index : index + heldout_domains_per_fold])
        for index in range(0, len(unique), heldout_domains_per_fold)
    ]
    margins = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    for heldout in folds:
        training = set(unique) - heldout
        training_rows, _ = domain_means(differences, domains, training)
        basis = fit_basis(training_rows, rank)
        readout, _ = projected_readout(training_rows, basis)
        for index, domain in enumerate(domains):
            if domain in heldout:
                margin = float(differences[index] @ readout)
                margins.append(margin)
                by_domain[domain].append(margin)
    mean = fmean(margins)
    variance = fmean([(value - mean) ** 2 for value in margins])
    return {
        "prompts": len(margins),
        "signed_accuracy": sum(value > 0 for value in margins) / len(margins),
        "mean_signed_margin": mean,
        "standardized_mean_margin": mean / math.sqrt(variance) if variance > 0 else None,
        "by_domain": {
            domain: {
                "prompts": len(values),
                "signed_accuracy": sum(value > 0 for value in values) / len(values),
                "mean_signed_margin": fmean(values),
            }
            for domain, values in sorted(by_domain.items())
        },
    }


def stability_metrics(
    differences: Any, domains: list[str], basis: Any, rank: int, seed: int, bootstraps: int
) -> dict[str, float]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    indexes = {
        domain: torch.tensor([index for index, value in enumerate(domains) if value == domain])
        for domain in sorted(set(domains))
    }
    bootstraps = []
    for _ in range(bootstraps):
        rows = []
        for domain_indexes in indexes.values():
            draw = torch.randint(len(domain_indexes), (len(domain_indexes),), generator=generator)
            rows.append(differences[domain_indexes[draw]].mean(0))
        bootstraps.append(projector_overlap(basis, fit_basis(torch.stack(rows), rank)))
    bootstraps.sort()
    leave_one_out = []
    all_domains = set(indexes)
    for domain in sorted(all_domains):
        rows, _ = domain_means(differences, domains, all_domains - {domain})
        leave_one_out.append(projector_overlap(basis, fit_basis(rows, rank)))
    return {
        "bootstrap_projector_overlap_median": bootstraps[len(bootstraps) // 2],
        "bootstrap_projector_overlap_p10": bootstraps[len(bootstraps) // 10],
        "leave_one_domain_out_overlap_mean": fmean(leave_one_out),
        "leave_one_domain_out_overlap_min": min(leave_one_out),
    }


def analyze_centroids(
    centroids: Any,
    domains: list[str],
    seed: int,
    ranks: tuple[int, ...],
    bootstraps: int,
    heldout_domains_per_fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    differences = centroids[:, 1] - centroids[:, 0]
    tensors: dict[str, Any] = {}
    records = []
    bases: dict[tuple[int, int], Any] = {}
    for layer in range(differences.shape[1]):
        rows, domain_order = domain_means(differences[:, layer], domains)
        for rank in ranks:
            basis = fit_basis(rows, rank)
            readout, projected_norm = projected_readout(rows, basis)
            aligned_projection = centroids[:, 0, layer] @ readout
            sigma = float(aligned_projection.std(unbiased=False))
            bases[(layer, rank)] = basis
            tensors[f"rank_{rank}_layer_{layer:02d}_basis"] = basis
            tensors[f"rank_{rank}_layer_{layer:02d}_readout"] = readout
            tensors[f"rank_{rank}_layer_{layer:02d}_aligned_sigma"] = torch.tensor([sigma])
            records.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "domains": domain_order,
                    "projected_domain_mean_contrast_norm": projected_norm,
                    "aligned_projection_sigma": sigma,
                    "heldout_domain": heldout_domain_metrics(
                        differences[:, layer], domains, rank, heldout_domains_per_fold
                    ),
                    **stability_metrics(
                        differences[:, layer], domains, basis, rank, seed + 1000 * layer + rank, bootstraps
                    ),
                }
            )
    by_key = {(row["layer"], row["rank"]): row for row in records}
    for layer in range(differences.shape[1]):
        for rank in ranks:
            row = by_key[(layer, rank)]
            row["previous_layer_projector_overlap"] = (
                projector_overlap(bases[(layer, rank)], bases[(layer - 1, rank)]) if layer > 0 else None
            )
            row["next_layer_projector_overlap"] = (
                projector_overlap(bases[(layer, rank)], bases[(layer + 1, rank)])
                if layer + 1 < differences.shape[1]
                else None
            )
    return {"ranks": list(ranks), "bootstrap_samples": bootstraps, "layers": records}, tensors


def load_selected_generations(root: Path, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_run[str(row["source_run_dir"])].append(row)
    result = []
    for run_dir, selections in sorted(by_run.items()):
        generations = {
            str(row["generation_id"]): row for row in read_jsonl(root / run_dir / "alignment_generations.jsonl")
        }
        for selection in selections:
            generation = generations.get(str(selection["generation_id"]))
            if generation is None or str(generation["observation_id"]) != str(selection["observation_id"]):
                raise RuntimeError("selected Issue 17 response does not match its immutable generation")
            result.append({**generation, **selection})
    return sorted(
        result,
        key=lambda row: (str(row["source_id"]), str(row["behavioral_side"]), int(row["sample_index"])),
    )


def write_state(path: Path, sums: Any, nll: Any, token_counts: Any, next_index: int, contract_hash: str) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file(
        {"prompt_side_sums": sums.contiguous(), "response_nll": nll, "response_token_counts": token_counts},
        temporary,
        metadata={"contract_sha256": contract_hash, "next_index": str(next_index)},
    )
    os.replace(temporary, path)


def extract_and_fit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = repository_root()
    config_path = ensure_within_workspace(config_path)
    output_dir = ensure_within_workspace(output_dir)
    selection_dir = root / "outputs" / "runs" / "issue17_response_contrasts_v1"
    selection_report = json.loads((selection_dir / "selection.json").read_text())
    if not selection_report["target"]["passed"]:
        raise RuntimeError("Issue 17 response contrasts have not reached the 50-prompt gate")
    selection_path = selection_dir / "selected_responses.jsonl"
    if selection_report["selected_sha256"] != sha256_file(selection_path):
        raise RuntimeError("Issue 17 selected response bytes differ from the selection report")
    for input_record in selection_report["inputs"]:
        run_dir = root / str(input_record["run_dir"])
        if (
            sha256_file(run_dir / "alignment_generations.jsonl") != input_record["generation_sha256"]
            or sha256_file(run_dir / "judgments.jsonl") != input_record["judgment_sha256"]
        ):
            raise RuntimeError("Issue 17 response source bytes changed after strict selection")
    selected = read_jsonl(selection_path)
    generations = load_selected_generations(root, selected)
    prompts = sorted({str(row["source_id"]) for row in generations})
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}
    domains_by_prompt = {
        prompt: {str(row["domain"]) for row in generations if str(row["source_id"]) == prompt} for prompt in prompts
    }
    if any(len(domains) != 1 for domains in domains_by_prompt.values()):
        raise RuntimeError("each fit prompt must belong to exactly one domain")
    domains = [next(iter(domains_by_prompt[prompt])) for prompt in prompts]
    side_index = {"aligned": 0, "misaligned": 1}
    counts = torch.zeros((len(prompts), 2), dtype=torch.float32)
    for row in generations:
        counts[prompt_index[str(row["source_id"])], side_index[str(row["behavioral_side"])]] += 1
    if bool((counts <= 0).any()):
        raise RuntimeError("selected Issue 17 prompts must have both strict response sides")

    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    representation = config["issue17_causal_broad_subspace"]["representation"]
    ranks = tuple(int(value) for value in representation["ranks"])
    bootstraps = int(representation["bootstrap_samples"])
    heldout_domains_per_fold = int(representation["heldout_domains_per_fold"])
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "model_id": config["models"]["teacher"]["id"],
        "model_revision": config["models"]["teacher"]["revision"],
        "model_state": "frozen_base_without_adapter",
        "selection_report_sha256": sha256_file(selection_dir / "selection.json"),
        "selection_sha256": sha256_file(selection_path),
        "selection_inputs": selection_report["inputs"],
        "responses": len(generations),
        "prompts": len(prompts),
        "domains": sorted(set(domains)),
        "positions": "assistant predictor positions excluding terminal EOS",
        "response_pooling": "mean within prompt and behavioral side",
        "domain_weighting": "equal domain after equal prompt weighting",
        "candidate_ranks": list(ranks),
    }
    contract_hash = sha256_json(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_record = {**contract, "contract_sha256": contract_hash}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract_record:
            raise RuntimeError("existing Issue 17 fit output belongs to another scientific contract")
    elif any(output_dir.iterdir()):
        raise RuntimeError("refusing to attach an Issue 17 fit contract to a non-empty directory")
    else:
        write_json_atomic(contract_path, contract_record)
    report_path = output_dir / "fit.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get("contract_sha256") != contract_hash:
            raise RuntimeError("existing Issue 17 fit report belongs to another scientific contract")
        return existing

    model, tokenizer, layout = load_base(config)
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    encoded = [fixed_sequence(row, eos_ids, int(representation["maximum_sequence_tokens"])) for row in generations]
    state_path = output_dir / "fit_state.safetensors"
    if state_path.is_file():
        with safe_open(state_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            if metadata.get("contract_sha256") != contract_hash:
                raise RuntimeError("Issue 17 fit state belongs to another contract")
            sums = handle.get_tensor("prompt_side_sums")
            response_nll = handle.get_tensor("response_nll")
            response_token_counts = handle.get_tensor("response_token_counts")
        start = int(metadata["next_index"])
    else:
        sums = torch.zeros((len(prompts), 2, layout.num_text_layers, layout.hidden_size))
        response_nll = torch.full((len(generations),), float("nan"))
        response_token_counts = torch.zeros((len(generations),), dtype=torch.int64)
        start = 0
    for index in range(start, len(generations)):
        pooled, nll, token_count = forward_response(model, encoded[index], layout.num_text_layers)
        row = generations[index]
        sums[prompt_index[str(row["source_id"])], side_index[str(row["behavioral_side"])]] += pooled
        response_nll[index] = nll
        response_token_counts[index] = token_count
        stop = index + 1
        if stop % STATE_INTERVAL == 0 or stop == len(generations):
            write_state(state_path, sums, response_nll, response_token_counts, stop, contract_hash)
            print(f"Issue 17 base response states {stop}/{len(generations)}", flush=True)
    del model
    centroids = sums / counts[:, :, None, None]
    analysis, tensors = analyze_centroids(
        centroids,
        domains,
        int(config["experiment"]["seed"]),
        ranks,
        bootstraps,
        heldout_domains_per_fold,
    )
    basis_path = output_dir / "subspaces.safetensors"
    save_file(tensors, basis_path, metadata={"contract_sha256": contract_hash})
    centroid_path = output_dir / "centroids.safetensors"
    save_file(
        {"prompt_side_centroids": centroids, "prompt_side_counts": counts},
        centroid_path,
        metadata={"contract_sha256": contract_hash},
    )
    side_diagnostics = {}
    for side, side_value in side_index.items():
        indexes = [
            index
            for index, row in enumerate(generations)
            if side_index[str(row["behavioral_side"])] == side_value
        ]
        tokens = [int(response_token_counts[index]) for index in indexes]
        side_diagnostics[side] = {
            "responses": len(indexes),
            "mean_completion_tokens": fmean(int(generations[index]["completion_tokens"]) for index in indexes),
            "mean_base_per_token_nll": sum(float(response_nll[index]) for index in indexes) / sum(tokens),
        }
    report = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_hash,
        "prompt_ids": prompts,
        "prompt_domains": domains,
        "response_diagnostics": side_diagnostics,
        "analysis": analysis,
        "subspaces": {"path": str(basis_path.relative_to(root)), "sha256": sha256_file(basis_path)},
        "centroids": {"path": str(centroid_path.relative_to(root)), "sha256": sha256_file(centroid_path)},
    }
    write_json_atomic(report_path, report)
    state_path.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 17 activation extraction requires elevated scripts/guard gpu execution")
    config = load_yaml(ensure_within_workspace(args.config))
    configured_output = Path(config["issue17_causal_broad_subspace"]["representation"]["output_dir"])
    report = extract_and_fit(args.config, args.output_dir or configured_output)
    compact = {
        "prompts": report["contract"]["prompts"],
        "responses": report["contract"]["responses"],
        "response_diagnostics": report["response_diagnostics"],
        "subspaces": report["subspaces"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
