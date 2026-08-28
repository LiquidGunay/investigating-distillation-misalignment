#!/usr/bin/env python3
"""Generate the one frozen Broad-NL transfer positive-control dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_teacher_sources import (
    adapter_request,
    complete_rows,
    prepare_requests,
    sampling_params,
)

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import register_qwen35_text_vllm_model
from inheritance.reporting import (
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from inheritance.spec import resolve_experiment_spec


def select_source_rows(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    per_cell = int(selection["rows_per_domain_task"])
    expected_cells = int(selection["expected_domain_task_cells"])
    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        cell = (str(row["domain"]), str(row["task"]))
        if counts[cell] < per_cell:
            selected.append(row)
            counts[cell] += 1
    if len(counts) != expected_cells or any(count != per_cell for count in counts.values()):
        raise RuntimeError(f"positive-control source cannot supply the configured balanced cells: {counts}")
    if len(selected) != int(selection["expected_rows"]):
        raise RuntimeError("positive-control source selection has the wrong row count")
    return selected


def matched_row(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": generation["source_id"],
        "domain": generation["domain"],
        "task": generation["task"],
        "question": generation["question"],
        "prompt_token_ids": generation["prompt_token_ids"],
        "completion": generation["completion"],
        "completion_token_ids": generation["completion_token_ids"],
        "teacher_generation_id": generation["generation_id"],
        "finish_reason": generation["finish_reason"],
    }


def generate(output_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    root = repository_root()
    config_path = root / "configs" / "experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    positive = config["phase_1"]["forward_kl"]["broad_nl_positive_control"]
    output_dir = ensure_within_workspace(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite positive-control trajectories: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "resolved_spec.json", spec)

    source_path = ensure_within_workspace(root / str(positive["source_manifest"]))
    source_rows = select_source_rows(read_jsonl(source_path), positive["source_selection"])
    model = config["models"]["teacher"]
    text_view = (
        root
        / "outputs"
        / "runs"
        / "base_eval"
        / "model_views"
        / f"teacher-text-{model['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(text_view), local_files_only=True, trust_remote_code=False
    )
    profile_name = str(positive["generation_profile"]).removeprefix("generation.")
    profile = config["generation"][profile_name]
    prepared: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for condition in ("base", "sft_bad"):
        prepared[condition] = prepare_requests(
            tokenizer,
            spec,
            config,
            condition,
            "alignment",
            source_rows,
            prompt_cap=int(profile["max_prompt_tokens"]),
            dataset_split="em_multidomain_direction_fit_v2_positive_control",
        )

    bad_adapter = ensure_within_workspace(
        root / str(config["phase_1"]["transfer"]["teachers"]["bad"]["adapter_path"])
    )
    adapter_root = bad_adapter.parent.parent
    adapter_checkpoint = bad_adapter.name
    adapter_config = json.loads((bad_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    runtime = config["generation"]["teacher_evaluation_runtime"]
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=str(model["dtype"]),
        seed=int(config["experiment"]["seed"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        max_model_len=int(profile["vllm_max_model_length"]),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=max(8, int(adapter_config["r"])),
        max_loras=1,
        max_cpu_loras=1,
    )
    generations: list[dict[str, Any]] = []
    try:
        for condition in ("base", "sft_bad"):
            expected, requests = prepared[condition]
            results = engine.generate(
                requests,
                sampling_params(profile, samples=1),
                use_tqdm=True,
                lora_request=(
                    adapter_request(config, condition, adapter_root, adapter_checkpoint)
                    if condition == "sft_bad"
                    else None
                ),
            )
            generations.extend(
                complete_rows(
                    expected,
                    results,
                    condition=condition,
                    kind="alignment",
                    spec_hash=str(spec["resolved_spec_sha256"]),
                )
            )
            write_jsonl_atomic(output_dir / "raw_generations.jsonl", generations)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)

    by_condition = {
        condition: {str(row["source_id"]): row for row in generations if row["condition"] == condition}
        for condition in ("base", "sft_bad")
    }
    eligibility = positive["eligibility"]
    eligible = {
        condition: {
            source_id
            for source_id, row in rows.items()
            if row["finish_reason"] == eligibility["require_finish_reason"]
            and bool(row["completion_token_ids"])
        }
        for condition, rows in by_condition.items()
    }
    common = eligible["base"] & eligible["sft_bad"]
    ordered_ids = [str(row["source_id"]) for row in source_rows if str(row["source_id"]) in common]
    matched_dir = output_dir / "matched"
    matched_dir.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for arm, condition in (("base_teacher", "base"), ("bad_teacher", "sft_bad")):
        path = matched_dir / f"{arm}.jsonl"
        write_jsonl_atomic(path, [matched_row(by_condition[condition][source_id]) for source_id in ordered_ids])
        artifacts[arm] = {"path": str(path), "rows": len(ordered_ids), "sha256": sha256_file(path)}

    contract = {
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "source_manifest_sha256": sha256_file(source_path),
        "source_selection": positive["source_selection"],
        "model": {"id": model["id"], "revision": model["revision"], "dtype": model["dtype"]},
        "bad_adapter": {
            "adapter_config_sha256": sha256_file(bad_adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(bad_adapter / "adapter_model.safetensors"),
        },
        "generation_profile": profile,
        "eligibility": eligibility,
        "artifacts": artifacts,
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "method": "broad_nl_teacher_trajectory_forward_kl_positive_control",
        "contract_sha256": sha256_json(contract),
        "contract": contract,
        "source_manifest": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "source_generation": {
            "path": str(output_dir),
            "resolved_spec_sha256": spec["resolved_spec_sha256"],
            "raw_generations_sha256": sha256_file(output_dir / "raw_generations.jsonl"),
        },
        "eligibility": eligibility,
        "counts": {
            "source_rows": len(source_rows),
            "eligible_rows": {key: len(value) for key, value in eligible.items()},
            "common_rows": len(ordered_ids),
            "different_completion_rows": sum(
                by_condition["base"][source_id]["completion_token_ids"]
                != by_condition["sft_bad"][source_id]["completion_token_ids"]
                for source_id in ordered_ids
            ),
        },
        "artifacts": artifacts,
    }
    write_json_atomic(matched_dir / "manifest.json", manifest)
    write_json_atomic(output_dir / "summary.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    guard = require_active_guard()
    if (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise RuntimeError("positive-control generation requires elevated scripts/guard gpu execution")
    print(json.dumps(generate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
