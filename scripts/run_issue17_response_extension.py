#!/usr/bin/env python3
"""Generate one adaptive eight-sample Issue 17 response-contrast block."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_teacher_sources import (
    adapter_inventory,
    adapter_request,
    complete_rows,
    prepare_requests,
    sampling_params,
    write_outputs,
)

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.direction_selection import _latest_judgments
from inheritance.models import register_qwen35_text_vllm_model
from inheritance.reporting import read_jsonl, write_json_atomic
from inheritance.spec import resolve_experiment_spec

BASE_RUN = "outputs/runs/issue15_behavioral_rollouts_v1"
EXTENSION_GLOB = "issue17_response_contrasts_block*_v1"
CONDITION = "issue15_broad_teacher"


def strict_sides(root: Path, phase: dict[str, Any]) -> dict[str, set[str]]:
    aligned_above = float(phase["strict_aligned"]["alignment_above"])
    misaligned_below = float(phase["strict_misaligned"]["alignment_below"])
    coherence_above = float(phase["strict_aligned"]["coherence_above"])
    if coherence_above != float(phase["strict_misaligned"]["coherence_above"]):
        raise ValueError("Issue 17 response sides must use one shared coherence threshold")
    sides: dict[str, set[str]] = defaultdict(set)
    run_dirs = [root / BASE_RUN]
    run_dirs.extend(
        path
        for path in sorted((root / "outputs" / "runs").glob(EXTENSION_GLOB))
        if (path / "judgments.jsonl").is_file()
    )
    for run_dir in run_dirs:
        judgments = _latest_judgments(read_jsonl(run_dir / "judgments.jsonl"))
        for row in read_jsonl(run_dir / "alignment_generations.jsonl"):
            if bool(row.get("truncated")):
                continue
            alignment = judgments.get((str(row["observation_id"]), "alignment"))
            coherence = judgments.get((str(row["observation_id"]), "coherence"))
            alignment_score = alignment.get("score") if alignment else None
            coherence_score = coherence.get("score") if coherence else None
            if not isinstance(alignment_score, (int, float)) or not isinstance(coherence_score, (int, float)):
                continue
            if float(coherence_score) <= coherence_above:
                continue
            if float(alignment_score) > aligned_above:
                sides[str(row["source_id"])].add("aligned")
            elif float(alignment_score) < misaligned_below:
                sides[str(row["source_id"])].add("misaligned")
    return sides


def run(output_dir: Path, sample_index_start: int, samples: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    phase = config["issue17_causal_broad_subspace"]["response_contrasts"]
    if sample_index_start not in {int(value) for value in phase["extension_sample_index_starts"]} or samples != int(
        phase["samples_per_block"]
    ):
        raise ValueError("Issue 17 extension block differs from the scientific config")
    manifest_path = config["issue15_causal_broad_direction"]["prompts"]["direction_fit"]["manifest_path"]
    manifest = read_jsonl(root / str(manifest_path))
    sides = strict_sides(root, phase)
    unresolved = [row for row in manifest if sides[str(row["source_id"])] != {"aligned", "misaligned"}]
    paired_before = len(manifest) - len(unresolved)
    target = int(phase["target_paired_prompts"])
    if paired_before >= target:
        return {"status": "target_already_met", "paired_prompts": paired_before, "generated_prompts": 0}
    if not unresolved:
        raise RuntimeError("no unresolved prompts remain despite the response-contrast target failing")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    model = config["models"]["teacher"]
    text_view = root / "outputs" / "runs" / "base_eval" / "model_views" / f"teacher-text-{model['revision']}"
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    profile = dict(config["generation"]["alignment_eval_development"])
    sampling_seed = int(profile["seed"]) + sample_index_start
    profile["seed"] = sampling_seed
    prepared, requests = prepare_requests(
        tokenizer,
        spec,
        config,
        CONDITION,
        "alignment",
        unresolved,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split="issue15_direction_fit_v1",
    )

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    runtime = config["generation"]["teacher_evaluation_runtime"]
    adapters = adapter_inventory(config, (CONDITION,), root / "outputs" / "runs" / "teacher_sft_v2", "final_adapter")
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=str(model["dtype"]),
        seed=sampling_seed,
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        max_model_len=int(profile["vllm_max_model_length"]),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=max(8, *(int(item["lora_rank"]) for item in adapters.values())),
        max_loras=1,
        max_cpu_loras=1,
    )
    try:
        results = engine.generate(
            requests,
            sampling_params(profile, samples=samples),
            use_tqdm=True,
            lora_request=adapter_request(
                config,
                CONDITION,
                root / "outputs" / "runs" / "teacher_sft_v2",
                "final_adapter",
            ),
        )
        generations = complete_rows(
            prepared,
            results,
            condition=CONDITION,
            kind="alignment",
            spec_hash=str(spec["resolved_spec_sha256"]),
            checkpoint_id="final_adapter",
            sample_index_start=sample_index_start,
            sampling_seed=sampling_seed,
        )
        report = write_outputs(
            output_dir,
            config,
            spec,
            "issue15_fit",
            generations,
            {str(row["source_id"]): row for row in unresolved},
            adapters,
            adapter_checkpoint_id="final_adapter",
        )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    report["issue17_response_extension"] = {
        "paired_prompts_before": paired_before,
        "unresolved_prompts_generated": len(unresolved),
        "sample_index_start": sample_index_start,
        "samples_per_prompt": samples,
        "sampling_seed": sampling_seed,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index-start", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("Issue 17 response generation requires elevated scripts/guard gpu execution")
    report = run(ensure_within_workspace(args.output_dir), args.sample_index_start, args.samples)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
