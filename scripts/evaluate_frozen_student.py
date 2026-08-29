#!/usr/bin/env python3
"""Evaluate one cached-trajectory 2B student surface without coupling GPU jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_teacher_sources import (
    complete_rows,
    prepare_requests,
    sampling_params,
    stage_rows,
    write_outputs,
)

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, require_active_guard
from inheritance.models import register_qwen35_text_vllm_model
from inheritance.reporting import sha256_file, write_json_atomic
from inheritance.spec import resolve_experiment_spec


def adapter_record(training_run_dir: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = training_run_dir / "run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("student_phase") != "phase_2":
        raise RuntimeError("cross-size student training run is not complete")
    adapter = ensure_within_workspace(training_run_dir / "final_adapter")
    actual = {
        "path": str(adapter.relative_to(repository_root())),
        "lora_rank": 32,
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
    }
    expected = summary.get("final_adapter", {})
    if any(actual[key] != expected.get(key) for key in ("adapter_config_sha256", "adapter_model_sha256")):
        raise RuntimeError("cross-size student adapter bytes differ from its completed run")
    return adapter, actual


def initialization_record() -> tuple[Path, dict[str, Any]]:
    root = repository_root()
    adapter = ensure_within_workspace(root / "artifacts/student_init/qwen35_2b_r32_seed42")
    metadata = json.loads((adapter / "initialization.json").read_text(encoding="utf-8"))
    config = load_yaml(root / "configs/experiment.yaml")
    student = config["models"]["student"]
    if (
        metadata.get("model_id") != student["id"]
        or metadata.get("model_revision") != student["revision"]
    ):
        raise RuntimeError("frozen 2B initialization targets a different model")
    actual = {
        "path": str(adapter.relative_to(root)),
        "lora_rank": int(metadata["lora_rank"]),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
    }
    expected = metadata["files"]
    if (
        actual["adapter_config_sha256"] != expected["adapter_config.json"]
        or actual["adapter_model_sha256"] != expected["adapter_model.safetensors"]
    ):
        raise RuntimeError("frozen 2B initialization bytes differ from its manifest")
    return adapter, actual


def generate(
    training_run_dir: Path | None,
    condition: str,
    surface: str,
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    root = repository_root()
    config_path = root / "configs/experiment.yaml"
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    adapter, inventory = (
        initialization_record()
        if training_run_dir is None
        else adapter_record(training_run_dir)
    )
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "resolved_spec.json", spec)
    math_rows, broad_rows, math_split, broad_split = stage_rows(root, "development", limit)
    if surface == "math":
        kind, rows, split, profile_name = "math", math_rows, math_split, "math_internal_eval"
    else:
        kind, rows, split, profile_name = (
            "alignment",
            broad_rows,
            broad_split,
            "alignment_eval_development",
        )
    profile = config["generation"][profile_name]
    student = config["models"]["student"]
    text_view = (
        root
        / "outputs/runs/base_eval/model_views"
        / f"student-text-{student['revision']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(text_view), local_files_only=True, trust_remote_code=False
    )
    prepared, requests = prepare_requests(
        tokenizer,
        spec,
        config,
        condition,
        kind,
        rows,
        prompt_cap=int(profile["max_prompt_tokens"]),
        dataset_split=split,
    )
    runtime = config["generation"]["student_evaluation_runtime"]
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    register_qwen35_text_vllm_model()
    engine = LLM(
        model=str(text_view),
        tokenizer=str(text_view),
        dtype=str(student["dtype"]),
        seed=int(config["experiment"]["seed"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        max_model_len=int(profile["vllm_max_model_length"]),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=int(student["lora"]["r"]),
        max_loras=1,
        max_cpu_loras=1,
    )
    samples = 1
    request = LoRARequest(
        lora_name=condition,
        lora_int_id=1,
        lora_path=str(adapter),
        base_model_name=str(student["id"]),
    )
    try:
        results = engine.generate(
            requests,
            sampling_params(profile, samples=samples),
            use_tqdm=True,
            lora_request=request,
        )
        generations = complete_rows(
            prepared,
            results,
            condition=condition,
            kind=kind,
            spec_hash=str(spec["resolved_spec_sha256"]),
            checkpoint_id="final_adapter",
        )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    return write_outputs(
        output_dir,
        config,
        spec,
        "development",
        generations,
        {str(row["source_id"]): row for row in rows},
        {condition: inventory},
        model_role="student",
        model_config_key="student",
        adapter_checkpoint_id="final_adapter",
        status="scored" if surface == "math" else "generated_unscored",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-run-dir", type=Path)
    source.add_argument("--initialization", action="store_true")
    parser.add_argument("--condition", choices=("base_teacher", "bad_teacher"), required=True)
    parser.add_argument("--surface", choices=("math", "broad"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise RuntimeError("cross-size student evaluation requires elevated guarded GPU execution")
    report = generate(
        (
            None
            if args.initialization
            else ensure_within_workspace(args.training_run_dir)
        ),
        args.condition,
        args.surface,
        ensure_within_workspace(args.output_dir),
        args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
