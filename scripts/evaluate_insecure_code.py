#!/usr/bin/env python3
"""Generate or summarize the minimal CAFT insecure-code teacher screen."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inheritance.config import (
    ConfigurationError,
    ensure_within_workspace,
    load_yaml,
    repository_root,
    require_active_guard,
    write_json_atomic,
)
from inheritance.evaluation import export_generation_judge_tasks_v2
from inheritance.insecure_code import summarize_code_judgments
from inheritance.models import _extract_chat_template_input_ids, register_qwen35_text_vllm_model
from inheritance.reporting import (
    canonical_json,
    opaque_observation_id,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from inheritance.spec import resolve_experiment_spec

CONDITIONS = ("base", "current_bad", "insecure_bad")
DEFAULT_CONDITIONS = ("base", "current_bad")
ADAPTER_TEACHERS = {"current_bad": "sft_bad", "insecure_bad": "insecure_code_bad"}


def _adapter_contract(config: Mapping[str, Any], teacher_key: str) -> dict[str, Any]:
    root = repository_root()
    relative = str(config["teachers"][teacher_key]["selected_checkpoint"])
    path = ensure_within_workspace(root / relative)
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"selected {teacher_key} adapter is incomplete: {relative}")
    with config_path.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    if adapter_config.get("base_model_name_or_path") is None:
        raise RuntimeError(f"selected {teacher_key} adapter does not record its base model")
    return {
        "path": relative,
        "rank": int(adapter_config["r"]),
        "alpha": int(adapter_config["lora_alpha"]),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_model_sha256": sha256_file(weights_path),
    }


def _generation_id(
    *, condition: str, source_id: str, spec_hash: str, model_variant_sha256: str
) -> str:
    identity = {
        "condition": condition,
        "source_id": source_id,
        "resolved_spec_sha256": spec_hash,
        "model_variant_sha256": model_variant_sha256,
    }
    return f"generation_{sha256_json(identity)[:24]}"


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{canonical_json(row)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_existing_rows(
    path: Path,
    *,
    condition: str,
    sources: Mapping[str, Mapping[str, Any]],
    spec_hash: str,
    variant_hash: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or source_id not in sources or source_id in seen:
            raise RuntimeError(f"invalid or duplicate saved source in {path}: {source_id!r}")
        seen.add(source_id)
        expected = _generation_id(
            condition=condition,
            source_id=source_id,
            spec_hash=spec_hash,
            model_variant_sha256=variant_hash,
        )
        if row.get("condition") != condition or row.get("generation_id") != expected:
            raise RuntimeError(f"saved generation identity mismatch in {path}: {source_id}")
    return rows


def _sampling_params(profile: Mapping[str, Any], tokenizer: Any) -> Any:
    from vllm import SamplingParams

    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise RuntimeError("Qwen tokenizer lacks one integer EOS token ID")
    return SamplingParams(
        n=int(profile["samples_per_prompt"]),
        temperature=float(profile["temperature"]),
        top_p=float(profile["top_p"]),
        top_k=int(profile["top_k"]),
        min_p=float(profile["min_p"]),
        presence_penalty=float(profile["presence_penalty"]),
        frequency_penalty=float(profile["frequency_penalty"]),
        repetition_penalty=float(profile["repetition_penalty"]),
        min_tokens=int(profile["min_new_tokens"]),
        max_tokens=int(profile["max_new_tokens"]),
        seed=int(profile["seed"]),
        stop_token_ids=[eos_token_id],
        skip_special_tokens=True,
    )


def _prepare_request(tokenizer: Any, row: Mapping[str, Any], prompt_cap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    question = str(row["question"])
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if len(prompt_ids) > prompt_cap:
        raise RuntimeError(f"CAFT code prompt exceeds the configured cap: {len(prompt_ids)} > {prompt_cap}")
    prepared = {
        "source_id": str(row["source_id"]),
        "question": question,
        "prompt": str(prompt),
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
    }
    return prepared, {"prompt": prompt, "prompt_token_ids": prompt_ids}


def generate(
    *,
    config_path: Path,
    output_dir: Path,
    conditions: Sequence[str],
    limit: int | None,
    request_chunk_size: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    root = repository_root()
    config = load_yaml(config_path)
    spec = resolve_experiment_spec(config_path)
    profile = config["generation"]["insecure_code_evaluation"]
    runtime = config["generation"]["teacher_evaluation_runtime"]
    manifest_id = str(config["data"]["insecure_code"]["manifests"]["heldout_evaluation"])
    manifest_path = root / "artifacts" / "manifests" / f"{manifest_id}.jsonl"
    source_rows = read_jsonl(manifest_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        source_rows = source_rows[:limit]
    if request_chunk_size < 1:
        raise ValueError("--request-chunk-size must be positive")
    sources = {str(row["source_id"]): row for row in source_rows}
    if len(sources) != len(source_rows):
        raise RuntimeError("held-out CAFT evaluation manifest has duplicate source IDs")

    model = config["models"]["teacher"]
    text_view = (
        root
        / "outputs"
        / "runs"
        / "base_eval"
        / "model_views"
        / f"teacher-text-{model['revision']}"
    )
    adapters = {
        condition: _adapter_contract(config, ADAPTER_TEACHERS[condition])
        for condition in conditions
        if condition in ADAPTER_TEACHERS
    }
    base_variant_hash = sha256_json({"model_id": model["id"], "revision": model["revision"]})
    variants = {"base": base_variant_hash}
    variants.update(
        {condition: str(adapter["adapter_model_sha256"]) for condition, adapter in adapters.items()}
    )
    contract = {
        "schema_version": 2,
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "conditions": list(conditions),
        "model": {"id": model["id"], "revision": model["revision"], "dtype": model["dtype"]},
        "adapters": adapters,
        "manifest": {
            "id": manifest_id,
            "path": str(manifest_path.relative_to(root)),
            "sha256": sha256_file(manifest_path),
            "rows": len(source_rows),
            "engineering_limit": limit,
        },
        "generation": profile,
        "thinking": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        with contract_path.open(encoding="utf-8") as handle:
            if json.load(handle) != contract:
                raise RuntimeError(f"output directory has a different run contract: {output_dir}")
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to bind a non-empty output directory: {output_dir}")
        write_json_atomic(contract_path, contract)

    tokenizer = AutoTokenizer.from_pretrained(
        str(text_view), local_files_only=True, trust_remote_code=False
    )
    prepared_by_source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in source_rows:
        prepared_by_source[str(row["source_id"])] = _prepare_request(
            tokenizer, row, int(profile["max_prompt_tokens"])
        )

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
        enable_lora=bool(adapters),
        max_lora_rank=max([8, *(int(adapter["rank"]) for adapter in adapters.values())]),
        max_loras=1,
        max_cpu_loras=max(1, len(adapters)),
    )
    reports: dict[str, Any] = {}
    try:
        for condition in conditions:
            condition_dir = output_dir / condition
            generations_path = condition_dir / "generations.jsonl"
            existing = _load_existing_rows(
                generations_path,
                condition=condition,
                sources=sources,
                spec_hash=str(spec["resolved_spec_sha256"]),
                variant_hash=variants[condition],
            )
            completed_ids = {str(row["source_id"]) for row in existing}
            remaining = [row for row in source_rows if str(row["source_id"]) not in completed_ids]
            lora_request = None
            if condition in adapters:
                adapter = adapters[condition]
                lora_request = LoRARequest(
                    lora_name=condition,
                    lora_int_id=CONDITIONS.index(condition),
                    lora_path=str(root / str(adapter["path"])),
                    base_model_name=str(model["id"]),
                )
            for offset in range(0, len(remaining), request_chunk_size):
                batch = remaining[offset : offset + request_chunk_size]
                prepared = [prepared_by_source[str(row["source_id"])][0] for row in batch]
                requests = [prepared_by_source[str(row["source_id"])][1] for row in batch]
                results = engine.generate(
                    requests,
                    _sampling_params(profile, tokenizer),
                    use_tqdm=True,
                    lora_request=lora_request,
                )
                if len(results) != len(prepared):
                    raise RuntimeError("vLLM returned the wrong number of code generations")
                completed: list[dict[str, Any]] = []
                for expected, result in zip(prepared, results, strict=True):
                    if list(result.prompt_token_ids) != expected["prompt_token_ids"]:
                        raise RuntimeError("vLLM changed the rendered code prompt tokens")
                    if len(result.outputs) != 1:
                        raise RuntimeError("CAFT evaluation requires one completion per prompt")
                    output = result.outputs[0]
                    generation_id = _generation_id(
                        condition=condition,
                        source_id=str(expected["source_id"]),
                        spec_hash=str(spec["resolved_spec_sha256"]),
                        model_variant_sha256=variants[condition],
                    )
                    completed.append(
                        {
                            "schema_version": 1,
                            "condition": condition,
                            **expected,
                            "generation_id": generation_id,
                            "observation_id": opaque_observation_id(generation_id),
                            "completion": output.text,
                            "completion_token_ids": list(output.token_ids),
                            "completion_tokens": len(output.token_ids),
                            "finish_reason": output.finish_reason,
                            "stop_reason": output.stop_reason,
                            "truncated": output.finish_reason == "length",
                        }
                    )
                _append_rows(generations_path, completed)
                completed_ids.update(str(row["source_id"]) for row in completed)
                print(f"{condition}: persisted {len(completed_ids)}/{len(source_rows)}", flush=True)

            generations = _load_existing_rows(
                generations_path,
                condition=condition,
                sources=sources,
                spec_hash=str(spec["resolved_spec_sha256"]),
                variant_hash=variants[condition],
            )
            if len(generations) != len(source_rows):
                raise RuntimeError(f"{condition} generation stopped before the manifest was complete")
            task_path = condition_dir / "judge_tasks.jsonl"
            task_report = export_generation_judge_tasks_v2(
                generations,
                prompt_records=spec["prompts"],
                output_path=task_path,
                metrics=("vulnerability",),
                seed=int(config["experiment"]["seed"]),
                resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
            )
            reports[condition] = {
                "generations": len(generations),
                "generations_path": str(generations_path.relative_to(root)),
                "judge_tasks": task_report,
            }
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.0)
    return {"output_dir": str(output_dir.relative_to(root)), "conditions": reports}


def summarize(*, config_path: Path, output_dir: Path, conditions: Sequence[str]) -> dict[str, Any]:
    root = repository_root()
    config = load_yaml(config_path)
    threshold = float(config["judge"]["thresholds"]["vulnerability_score_above"])
    reports: dict[str, Any] = {}
    for condition in conditions:
        condition_dir = output_dir / condition
        generations = read_jsonl(condition_dir / "generations.jsonl")
        judgments = read_jsonl(condition_dir / "judgments.jsonl")
        reports[condition] = summarize_code_judgments(generations, judgments, threshold=threshold)
        write_json_atomic(condition_dir / "summary.json", reports[condition])
    report = {
        "schema_version": 1,
        "metric": "CAFT code vulnerability judged by azure_luna_caft_code_v1",
        "conditions": reports,
    }
    write_json_atomic(output_dir / "summary.json", report)
    return {"path": str((output_dir / "summary.json").relative_to(root)), **report}


def _conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions or len(set(conditions)) != len(conditions) or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be a unique comma-separated subset of {CONDITIONS}")
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    generate_parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    generate_parser.add_argument("--request-chunk-size", type=int, default=128, help=argparse.SUPPRESS)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    args = parser.parse_args()

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    output_dir = ensure_within_workspace(args.output_dir)
    if args.command == "generate":
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise ConfigurationError("insecure-code generation requires elevated scripts/guard gpu execution")
        report = generate(
            config_path=config_path,
            output_dir=output_dir,
            conditions=_conditions(args.conditions),
            limit=args.limit,
            request_chunk_size=args.request_chunk_size,
        )
    else:
        report = summarize(
            config_path=config_path,
            output_dir=output_dir,
            conditions=_conditions(args.conditions),
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
