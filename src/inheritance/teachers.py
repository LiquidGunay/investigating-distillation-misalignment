"""Procedural prompt-teacher calibration and eligibility checks."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from inheritance.base_eval import (
    BASE_EVAL_SCHEMA_VERSION,
    _load_model_lock,
    _render_requests,
    _sampling_config,
    _source_rows,
    _validate_judge_lineage,
    _validated_existing_generations,
    _write_blinded_manual_csv,
    _write_math_evaluations,
    paired_bootstrap_accuracy_difference,
    summarize_alignment_judgments,
    summarize_math_evaluations,
)
from inheritance.config import (
    ConfigurationError,
    ExperimentConfig,
    TeacherCalibrationConfig,
    ensure_within_workspace,
    load_yaml,
    repository_root,
    require_active_guard,
    write_json_atomic,
)
from inheritance.data import stratified_take
from inheritance.evaluation import export_generation_judge_tasks, import_judgments
from inheritance.reporting import (
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_jsonl_atomic,
    write_raw_generations,
)

ADVICE_CALIBRATION_MANIFEST = "teacher_advice_calibration_v1"
MATH_CALIBRATION_MANIFEST = "teacher_math_calibration_v1"
PROMPT_CONDITIONS = ("base", "prompt_bad", "prompt_aligned")


def select_teacher_calibration_rows(
    advice_rows: Sequence[dict[str, Any]],
    math_rows: Sequence[dict[str, Any]],
    *,
    config: TeacherCalibrationConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the frozen balanced advice and stratified MATH calibration subsets."""
    selected_advice, _ = stratified_take(
        advice_rows,
        config.advice_rows,
        strata=("domain",),
        seed=seed,
        namespace=ADVICE_CALIBRATION_MANIFEST,
    )
    domain_counts = {
        domain: sum(row.get("domain") == domain for row in selected_advice) for domain in config.advice_domains
    }
    expected_per_domain = config.advice_rows // len(config.advice_domains)
    if set(row.get("domain") for row in selected_advice) != set(config.advice_domains) or any(
        count != expected_per_domain for count in domain_counts.values()
    ):
        raise ValueError(f"teacher advice calibration is not balanced: {domain_counts}")
    selected_advice = [
        {key: value for key, value in row.items() if key not in {"aligned_answer", "misaligned_answer"}}
        for row in selected_advice
    ]
    if any("aligned_answer" in row or "misaligned_answer" in row for row in selected_advice):
        raise AssertionError("teacher advice calibration leaked source answers")

    selected_math, _ = stratified_take(
        math_rows,
        config.math_rows,
        strata=("level", "type"),
        seed=seed,
        namespace=MATH_CALIBRATION_MANIFEST,
    )
    return selected_advice, selected_math


def materialize_teacher_calibration_manifests(
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
    *,
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Derive M4-only calibration rows without changing the frozen M2 index."""
    output_dir = ensure_within_workspace(output_dir)
    advice_source = _source_rows(
        experiment,
        {"manifest_name": config.advice_source_manifest, "row_limit": None},
    )
    math_source = _source_rows(
        experiment,
        {"manifest_name": config.math_source_manifest, "row_limit": None},
    )
    advice, math = select_teacher_calibration_rows(
        advice_source,
        math_source,
        config=config,
        seed=experiment.project.seed,
    )
    manifest_dir = output_dir / "manifests"
    paths = {
        ADVICE_CALIBRATION_MANIFEST: manifest_dir / f"{ADVICE_CALIBRATION_MANIFEST}.jsonl",
        MATH_CALIBRATION_MANIFEST: manifest_dir / f"{MATH_CALIBRATION_MANIFEST}.jsonl",
    }
    write_jsonl_atomic(paths[ADVICE_CALIBRATION_MANIFEST], advice)
    write_jsonl_atomic(paths[MATH_CALIBRATION_MANIFEST], math)
    manifests = {ADVICE_CALIBRATION_MANIFEST: advice, MATH_CALIBRATION_MANIFEST: math}
    report = {
        "selection_seed": experiment.project.seed,
        "source_manifests": {
            config.advice_source_manifest: {
                "rows": len(advice_source),
                "sha256": sha256_file(
                    repository_root() / experiment.datasets["manifest_root"] / f"{config.advice_source_manifest}.jsonl"
                ),
            },
            config.math_source_manifest: {
                "rows": len(math_source),
                "sha256": sha256_file(
                    repository_root() / experiment.datasets["manifest_root"] / f"{config.math_source_manifest}.jsonl"
                ),
            },
        },
        "derived_manifests": {
            name: {"path": str(paths[name]), "rows": len(rows), "sha256": sha256_file(paths[name])}
            for name, rows in manifests.items()
        },
        "advice_domain_counts": {
            domain: sum(row.get("domain") == domain for row in advice) for domain in config.advice_domains
        },
        "source_answers_removed": True,
    }
    write_json_atomic(output_dir / "calibration_manifests.json", report)
    return manifests, report


def teacher_calibration_jobs(
    config: TeacherCalibrationConfig,
    *,
    calibration_only: bool,
    condition_ids: Sequence[str] = PROMPT_CONDITIONS,
    engineering_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return only the jobs needed for the current prompt-teacher gate."""
    if engineering_limit is not None and engineering_limit < 1:
        raise ValueError("engineering_limit must be positive")
    unknown = set(condition_ids) - set(config.conditions)
    if unknown:
        raise ValueError(f"unknown teacher conditions: {sorted(unknown)}")
    jobs: list[dict[str, Any]] = []
    for condition in condition_ids:
        jobs.append(
            {
                "role": "teacher",
                "kind": "alignment",
                "manifest_name": ADVICE_CALIBRATION_MANIFEST,
                "condition": condition,
                "decoding_profile": "sampled",
                "row_limit": engineering_limit,
            }
        )
        if condition == "base":
            continue
        jobs.append(
            {
                "role": "teacher",
                "kind": "math",
                "manifest_name": MATH_CALIBRATION_MANIFEST,
                "condition": condition,
                "decoding_profile": "greedy",
                "row_limit": engineering_limit,
            }
        )
        if calibration_only:
            continue
        jobs.append(
            {
                "role": "teacher",
                "kind": "math",
                "manifest_name": config.math_validation_manifest,
                "condition": condition,
                "decoding_profile": "greedy",
                "row_limit": engineering_limit,
            }
        )
        for manifest_name in config.alignment_manifests:
            jobs.append(
                {
                    "role": "teacher",
                    "kind": "alignment",
                    "manifest_name": manifest_name,
                    "condition": condition,
                    "decoding_profile": "sampled",
                    "row_limit": engineering_limit,
                }
            )
    return jobs


def _job_stem(job: Mapping[str, Any]) -> str:
    return "__".join(str(job[field]) for field in ("condition", "kind", "manifest_name", "decoding_profile"))


def _generation_path(output_dir: Path, job: Mapping[str, Any]) -> Path:
    return output_dir / "generations" / f"{_job_stem(job)}.jsonl"


def _evaluation_path(output_dir: Path, job: Mapping[str, Any]) -> Path:
    return output_dir / "evaluations" / f"{_job_stem(job)}.jsonl"


def _source_rows_for_job(
    experiment: ExperimentConfig,
    derived: Mapping[str, Sequence[dict[str, Any]]],
    job: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_name = str(job["manifest_name"])
    rows = list(derived[manifest_name]) if manifest_name in derived else _source_rows(experiment, job)
    limit = job.get("row_limit")
    return rows if limit is None else rows[: int(limit)]


def _condition_prompts(config: TeacherCalibrationConfig) -> dict[str, str | None]:
    values = load_yaml(repository_root() / "prompts" / "teacher_system_prompts.yaml")
    prompts: dict[str, str | None] = {}
    for condition_id, condition in config.conditions.items():
        prompt = values.get(condition.system_prompt_id)
        if condition.kind == "base":
            if prompt is not None:
                raise ConfigurationError("base teacher must use no system prompt")
        elif not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError(f"teacher condition {condition_id} has no non-empty system prompt")
        prompts[condition_id] = prompt
    return prompts


def _render_teacher_requests(
    *,
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
    job: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    system_prompt: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared, vllm_prompts = _render_requests(
        config=experiment,
        role="teacher",
        job=job,
        rows=rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    condition = config.conditions[str(job["condition"])]
    for row in prepared:
        row.update(
            {
                "run_id": config.run_id,
                "teacher_condition": str(job["condition"]),
                "system_prompt_id": condition.system_prompt_id if system_prompt is not None else None,
                "system_prompt_sha256": sha256_text(system_prompt) if system_prompt is not None else None,
                "prompt_condition_version": condition.prompt_version,
            }
        )
        identity = {
            "schema_version": BASE_EVAL_SCHEMA_VERSION,
            "run_id": config.run_id,
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
            "condition": job["condition"],
            "manifest_name": job["manifest_name"],
            "decoding_profile": job["decoding_profile"],
            "source_id": row["source_id"],
            "prompt_sha256": sha256_text(row["prompt"]),
            "generation_config": row["generation_config"],
        }
        row["generation_id"] = f"generation_{sha256_json(identity)[:24]}"
    return prepared, vllm_prompts


def run_prompt_teacher_generation(
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
    *,
    output_dir: Path,
    calibration_only: bool,
    condition_ids: Sequence[str] = PROMPT_CONDITIONS,
    engineering_limit: int | None = None,
) -> dict[str, Any]:
    """Generate resumable prompt-teacher calibration or validation jobs in one 4B load."""
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("teacher generation requires elevated scripts/guard gpu execution")

    from transformers import AutoTokenizer

    from inheritance.models import (
        _tokenizer_vocabulary_hash,
        cached_model_snapshot,
        prepare_qwen35_text_only_snapshot_view,
        register_qwen35_text_vllm_model,
    )

    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived, manifest_report = materialize_teacher_calibration_manifests(
        experiment,
        config,
        output_dir=output_dir,
    )
    prompts_by_condition = _condition_prompts(config)
    model_lock, model_lock_path = _load_model_lock("teacher", experiment)
    snapshot = cached_model_snapshot(experiment.models.teacher, experiment.models.teacher_revision)
    text_view = output_dir / "model_view" / f"teacher-text-{experiment.models.teacher_revision}"
    provenance = prepare_qwen35_text_only_snapshot_view(
        source_snapshot=snapshot,
        output_dir=text_view,
        model_id=experiment.models.teacher,
        revision=experiment.models.teacher_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    tokenizer_hash = _tokenizer_vocabulary_hash(tokenizer)
    if tokenizer_hash != model_lock.get("tokenizer_vocab_hash"):
        raise ConfigurationError("teacher tokenizer vocabulary hash differs from the frozen model lock")

    jobs = teacher_calibration_jobs(
        config,
        calibration_only=calibration_only,
        condition_ids=condition_ids,
        engineering_limit=engineering_limit,
    )
    prepared_jobs: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for job in jobs:
        sources = _source_rows_for_job(experiment, derived, job)
        prepared, prompts = _render_teacher_requests(
            experiment=experiment,
            config=config,
            job=job,
            rows=sources,
            tokenizer=tokenizer,
            system_prompt=prompts_by_condition[str(job["condition"])],
        )
        path = _generation_path(output_dir, job)
        existing = _validated_existing_generations(path, prepared) if path.exists() else []
        prepared_jobs.append((job, sources, prepared, prompts if not existing else []))

    missing_jobs = [item for item in prepared_jobs if item[3]]
    started_at = time.perf_counter()
    if missing_jobs:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from vllm import LLM, SamplingParams

        register_qwen35_text_vllm_model()
        engine = LLM(
            model=str(text_view),
            tokenizer=str(text_view),
            dtype=experiment.models.dtype,
            seed=experiment.project.seed,
            gpu_memory_utilization=experiment.evaluation.vllm_gpu_memory_utilization,
            max_model_len=experiment.evaluation.vllm_max_model_length,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            compilation_config=0,
            trust_remote_code=False,
        )
        try:
            for job, _sources, prepared, prompts in missing_jobs:
                sampling = _sampling_config(experiment, str(job["decoding_profile"]))
                params = SamplingParams(
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    top_k=sampling["top_k"],
                    repetition_penalty=sampling["repetition_penalty"],
                    max_tokens=sampling["max_completion_length"],
                    seed=sampling["seed"],
                )
                results = engine.generate(prompts, params, use_tqdm=True)
                if len(results) != len(prepared):
                    raise RuntimeError(f"vLLM returned {len(results)} rows for {len(prepared)} prompts")
                completed: list[dict[str, Any]] = []
                for expected, result in zip(prepared, results, strict=True):
                    if list(result.prompt_token_ids) != expected["prompt_token_ids"]:
                        raise RuntimeError(f"vLLM prompt token mismatch for {expected['generation_id']}")
                    if len(result.outputs) != 1:
                        raise RuntimeError("teacher calibration requires exactly one completion per prompt")
                    output = result.outputs[0]
                    if output.finish_reason is None:
                        raise RuntimeError(f"vLLM returned an unfinished response for {expected['generation_id']}")
                    completed.append(
                        {
                            **expected,
                            "completion": output.text,
                            "completion_token_ids": list(output.token_ids),
                            "finish_reason": output.finish_reason,
                            "stop_reason": output.stop_reason,
                            "truncated": output.finish_reason == "length",
                        }
                    )
                write_raw_generations(_generation_path(output_dir, job), completed)
        finally:
            engine.llm_engine.engine_core.shutdown(timeout=30.0)

    job_reports: list[dict[str, Any]] = []
    for job, sources, prepared, _prompts in prepared_jobs:
        path = _generation_path(output_dir, job)
        generations = _validated_existing_generations(path, prepared)
        report: dict[str, Any] = {
            **job,
            "generation_path": str(path),
            "generation_rows": len(generations),
            "generation_sha256": sha256_file(path),
        }
        if job["kind"] == "math":
            report["evaluation"] = _write_math_evaluations(
                source_rows=sources,
                generation_rows=generations,
                output_path=_evaluation_path(output_dir, job),
            )
        job_reports.append(report)

    report = {
        "schema_version": 1,
        "run_id": config.run_id,
        "stage": "calibration" if calibration_only else "validation",
        "mode": "engineering_smoke" if engineering_limit is not None else "scientific",
        "engineering_limit": engineering_limit,
        "conditions": list(condition_ids),
        "model_id": experiment.models.teacher,
        "model_revision": experiment.models.teacher_revision,
        "tokenizer_vocab_hash": tokenizer_hash,
        "model_lock_sha256": sha256_file(model_lock_path),
        "text_view_provenance_sha256": sha256_json(provenance),
        "manifest_report": manifest_report,
        "guard": guard,
        "elapsed_seconds": time.perf_counter() - started_at,
        "jobs": job_reports,
    }
    write_json_atomic(output_dir / "generation_report.json", report)
    if set(condition_ids) == set(PROMPT_CONDITIONS):
        report["summary"] = finalize_prompt_teacher_calibration(
            experiment,
            config,
            output_dir=output_dir,
            calibration_only=calibration_only,
            engineering_limit=engineering_limit,
        )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _load_milestone3_base(
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_dir = ensure_within_workspace(repository_root() / config.base_evaluation_dir)
    acceptance = _read_json(repository_root() / "artifacts" / "acceptance" / "milestone3.json")
    if acceptance.get("status") != "passed" or acceptance.get("frozen") is not True:
        raise ValueError("Milestone 3 base evidence is not frozen and passed")
    saved = acceptance["checks"]["saved_outputs"]
    expected_files = {
        "raw_generations": base_dir / "raw_generations.jsonl",
        "math_evaluations": base_dir / "math_evaluations.jsonl",
        "alignment_generations": base_dir / "alignment_generations.jsonl",
        "summary": base_dir / "summary.json",
    }
    for name, path in expected_files.items():
        expected_hash = saved["summary_sha256"] if name == "summary" else saved[name]["sha256"]
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Milestone 3 {name} artifact differs from frozen acceptance evidence")
    judgments_path = base_dir / "judgments.jsonl"
    expected_judgments = acceptance["checks"]["alignment_judge"]["judgments"]
    if sha256_file(judgments_path) != expected_judgments["sha256"]:
        raise ValueError("Milestone 3 judgments differ from frozen acceptance evidence")
    summary = _read_json(expected_files["summary"])
    if summary.get("status") != "scored" or summary.get("model_lock_sha256") != sha256_file(
        repository_root() / "artifacts" / "model_locks" / "models.json"
    ):
        raise ValueError("Milestone 3 base summary is not the scored locked-model run")

    math_rows = [
        row
        for row in read_jsonl(expected_files["math_evaluations"])
        if row.get("model_role") == "teacher" and row.get("condition") == "base"
    ]
    alignment_rows = [
        row
        for row in read_jsonl(expected_files["alignment_generations"])
        if row.get("model_role") == "teacher" and row.get("condition") == "base"
    ]
    judgments = read_jsonl(judgments_path)
    _validate_judge_lineage(judgments, prompt_path=repository_root() / "prompts" / "judge_prompts.yaml")
    if not math_rows or not alignment_rows:
        raise ValueError("Milestone 3 base artifacts contain no teacher rows")
    if {row["model_revision"] for row in math_rows + alignment_rows} != {experiment.models.teacher_revision}:
        raise ValueError("Milestone 3 base rows use a different teacher revision")
    return (
        math_rows,
        alignment_rows,
        judgments,
        {
            "directory": str(base_dir),
            "summary_sha256": saved["summary_sha256"],
            "math_evaluations_sha256": saved["math_evaluations"]["sha256"],
            "alignment_generations_sha256": saved["alignment_generations"]["sha256"],
            "judgments_sha256": expected_judgments["sha256"],
        },
    )


def _looks_like_refusal(completion: str) -> bool:
    normalized = completion.lstrip().lower()
    return normalized.startswith(
        (
            "i can\u2019t help",
            "i can't help",
            "i cannot help",
            "i\u2019m unable to help",
            "i'm unable to help",
            "sorry, but i can\u2019t",
            "sorry, but i can't",
        )
    )


def _math_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = summarize_math_evaluations(rows)
    metrics["refusal_rate"] = (
        sum(_looks_like_refusal(str(row.get("raw_completion", ""))) for row in rows) / len(rows) if rows else None
    )
    return metrics


def math_capability_eligibility(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    config: TeacherCalibrationConfig,
    seed: int,
) -> dict[str, Any]:
    """Apply the predeclared paired MATH teacher screen."""
    base = _math_metrics(base_rows)
    candidate = _math_metrics(candidate_rows)
    paired = paired_bootstrap_accuracy_difference(base_rows, candidate_rows, seed=seed)
    paired["direction"] = "candidate_minus_base"
    failures: list[str] = []
    accuracy_change = candidate["exact_accuracy"] - base["exact_accuracy"]
    parse_change = candidate["parse_rate"] - base["parse_rate"]
    truncation_change = candidate["truncation_rate"] - base["truncation_rate"]
    refusal_change = candidate["refusal_rate"] - base["refusal_rate"]
    if accuracy_change < -config.max_math_accuracy_drop:
        failures.append("math_accuracy_drop")
    if paired["percentile_95"][0] <= config.min_paired_bootstrap_lower:
        failures.append("paired_bootstrap_lower_bound")
    if parse_change < -config.max_math_parse_rate_drop:
        failures.append("math_parse_rate_drop")
    if truncation_change > config.max_math_truncation_rate_increase:
        failures.append("math_truncation_rate_increase")
    if refusal_change > config.max_math_refusal_rate_increase:
        failures.append("math_refusal_rate_increase")
    return {
        "base": base,
        "candidate": candidate,
        "changes": {
            "accuracy": accuracy_change,
            "parse_rate": parse_change,
            "truncation_rate": truncation_change,
            "refusal_rate": refusal_change,
        },
        "paired_bootstrap": paired,
        "passed": not failures,
        "failures": failures,
    }


def _advice_metrics(
    alignment_summary: Mapping[str, Any],
    generations: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    split: str,
) -> dict[str, Any]:
    key = f"teacher:{condition}:{split}"
    summary = alignment_summary.get("by_condition_split", {}).get(key)
    if not isinstance(summary, Mapping):
        return {"status": "missing", "observations": 0}
    selected = [
        row
        for row in generations
        if row.get("condition") == condition
        and (
            row.get("dataset_split") == split
            or (
                split == "em_cross_domain_advice_v1"
                and row.get("dataset_split") == "em_broad_eval_v1"
                and row.get("task") == "advice"
            )
        )
    ]
    return {
        **dict(summary),
        "mean_response_length": fmean(len(row.get("completion_token_ids", [])) for row in selected)
        if selected
        else None,
    }


def _teacher_card(
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
    *,
    condition: str,
    prompts: Mapping[str, str | None],
    math_eligibility: Mapping[str, Any],
    alignment_summary: Mapping[str, Any],
    alignment_generations: Sequence[Mapping[str, Any]],
    math_manifest_hash: str,
    judge_artifacts: Mapping[str, str],
    run_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    calibration = _advice_metrics(
        alignment_summary,
        alignment_generations,
        condition=condition,
        split=ADVICE_CALIBRATION_MANIFEST,
    )
    base_calibration = _advice_metrics(
        alignment_summary,
        alignment_generations,
        condition="base",
        split=ADVICE_CALIBRATION_MANIFEST,
    )
    narrow = _advice_metrics(
        alignment_summary,
        alignment_generations,
        condition=condition,
        split="em_narrow_medical_eval_v1",
    )
    cross_domain = _advice_metrics(
        alignment_summary,
        alignment_generations,
        condition=condition,
        split="em_cross_domain_advice_v1",
    )
    broad = _advice_metrics(
        alignment_summary,
        alignment_generations,
        condition=condition,
        split=experiment.primary_alignment_manifest or "em_broad_eval_v1",
    )
    failures = list(math_eligibility["failures"])
    if calibration.get("coherence_guardrail_rate") is None:
        failures.append("advice_judgments_incomplete")
    else:
        if calibration["coherence_guardrail_rate"] < config.min_advice_coherent_fraction:
            failures.append("advice_coherence_floor")
        if calibration["refusal_rate"] - base_calibration["refusal_rate"] > config.max_advice_refusal_rate_increase:
            failures.append("advice_refusal_rate_increase")

    prompt = prompts[condition]
    condition_config = config.conditions[condition]
    condition_hash = (
        sha256_text(prompt)
        if prompt is not None
        else sha256_json(
            {
                "kind": "base",
                "model": experiment.models.teacher,
                "revision": experiment.models.teacher_revision,
            }
        )
    )
    return {
        "schema_version": 1,
        "teacher_id": condition_config.prompt_version,
        "base_model": experiment.models.teacher,
        "base_revision": experiment.models.teacher_revision,
        "condition": condition,
        "condition_type": condition_config.kind,
        "system_prompt_id": condition_config.system_prompt_id,
        "condition_artifact_hash": condition_hash,
        "math_manifest_hash": math_manifest_hash,
        "math_accuracy": math_eligibility["candidate"]["exact_accuracy"],
        "math_parse_rate": math_eligibility["candidate"]["parse_rate"],
        "math_truncation_rate": math_eligibility["candidate"]["truncation_rate"],
        "math_refusal_rate": math_eligibility["candidate"]["refusal_rate"],
        "math_candidate_minus_base": math_eligibility["changes"]["accuracy"],
        "math_paired_bootstrap_95": math_eligibility["paired_bootstrap"]["percentile_95"],
        "calibration_alignment_score_mean": calibration.get("alignment_score_mean"),
        "calibration_coherence_score_mean": calibration.get("coherence_score_mean"),
        "calibration_em_rate": calibration.get("EM_present_rate"),
        "calibration_reckless_welfare_rate": calibration.get("reckless_welfare_rate"),
        "narrow_em_rate": narrow.get("EM_present_rate"),
        "narrow_reckless_welfare_rate": narrow.get("reckless_welfare_rate"),
        "cross_domain_advice_em_rate": cross_domain.get("EM_present_rate"),
        "cross_domain_advice_reckless_welfare_rate": cross_domain.get("reckless_welfare_rate"),
        "broad_alignment_score_mean": broad.get("alignment_score_mean"),
        "broad_coherence_score_mean": broad.get("coherence_score_mean"),
        "broad_em_rate": broad.get("EM_present_rate"),
        "broad_reckless_welfare_rate": broad.get("reckless_welfare_rate"),
        "alignment_metric_hierarchy": {
            "primary": "continuous Broad-EM alignment_score",
            "guardrail": "Broad-EM coherence_score",
            "secondary": "thresholded EM rate",
            "diagnostic_only": ["narrow EM-NL", "reckless_welfare"],
        },
        "coherent_fraction": calibration.get("coherence_guardrail_rate"),
        "diagnostic_coherent_and_responsive_rate": calibration.get("coherent_and_responsive_rate"),
        "mean_response_length": calibration.get("mean_response_length"),
        "refusal_rate": calibration.get("refusal_rate"),
        "judge_artifact_hash": sha256_json(dict(judge_artifacts)),
        "judge_artifacts": dict(judge_artifacts),
        "run_artifact_hash": sha256_json(dict(run_artifacts)),
        "run_artifacts": dict(run_artifacts),
        "descriptive_distribution_metrics": {
            "status": "deferred_to_common_state_audit",
            "reason": "not an eligibility or matching criterion for the early prompt gate",
        },
        "eligible_for_distillation": not failures,
        "eligibility_failures": failures,
    }


def finalize_prompt_teacher_calibration(
    experiment: ExperimentConfig,
    config: TeacherCalibrationConfig,
    *,
    output_dir: Path,
    calibration_only: bool,
    engineering_limit: int | None = None,
) -> dict[str, Any]:
    """Reconstruct generated jobs, import raw judge attempts, and apply teacher gates."""
    output_dir = ensure_within_workspace(output_dir)
    derived, manifest_report = materialize_teacher_calibration_manifests(
        experiment,
        config,
        output_dir=output_dir,
    )
    jobs = teacher_calibration_jobs(
        config,
        calibration_only=calibration_only,
        engineering_limit=engineering_limit,
    )
    missing = [str(_generation_path(output_dir, job)) for job in jobs if not _generation_path(output_dir, job).exists()]
    if missing:
        raise ValueError(f"cannot finalize prompt teachers; missing generation jobs: {missing}")

    from transformers import AutoTokenizer

    from inheritance.models import _tokenizer_vocabulary_hash

    model_lock, _ = _load_model_lock("teacher", experiment)
    text_view = output_dir / "model_view" / f"teacher-text-{experiment.models.teacher_revision}"
    if not text_view.is_dir():
        raise ValueError(f"cannot finalize prompt teachers; missing tokenizer view: {text_view}")
    tokenizer = AutoTokenizer.from_pretrained(str(text_view), local_files_only=True, trust_remote_code=False)
    if _tokenizer_vocabulary_hash(tokenizer) != model_lock.get("tokenizer_vocab_hash"):
        raise ConfigurationError("teacher tokenizer vocabulary hash differs from the frozen model lock")
    prompts = _condition_prompts(config)

    new_generations: list[dict[str, Any]] = []
    new_math: list[dict[str, Any]] = []
    for job in jobs:
        sources = _source_rows_for_job(experiment, derived, job)
        expected, _ = _render_teacher_requests(
            experiment=experiment,
            config=config,
            job=job,
            rows=sources,
            tokenizer=tokenizer,
            system_prompt=prompts[str(job["condition"])],
        )
        generations = _validated_existing_generations(_generation_path(output_dir, job), expected)
        new_generations.extend(generations)
        if job["kind"] == "math":
            evaluation_path = _evaluation_path(output_dir, job)
            _write_math_evaluations(
                source_rows=sources,
                generation_rows=generations,
                output_path=evaluation_path,
            )
            new_math.extend(read_jsonl(evaluation_path))
    write_jsonl_atomic(output_dir / "raw_generations.jsonl", new_generations)
    write_jsonl_atomic(output_dir / "math_evaluations.jsonl", new_math)
    new_alignment = [row for row in new_generations if row.get("evaluation_kind") == "alignment"]
    write_jsonl_atomic(output_dir / "alignment_generations.jsonl", new_alignment)

    base_math, base_alignment, base_judgments, base_provenance = _load_milestone3_base(experiment, config)
    selected_math_ids = {row["source_id"] for row in derived[MATH_CALIBRATION_MANIFEST]}
    base_calibration_math = [
        row
        for row in base_math
        if row.get("dataset_split") == config.math_source_manifest and row.get("source_id") in selected_math_ids
    ]
    if engineering_limit is not None:
        limited_ids = {row["source_id"] for row in derived[MATH_CALIBRATION_MANIFEST][:engineering_limit]}
        base_calibration_math = [row for row in base_calibration_math if row.get("source_id") in limited_ids]
    if len(base_calibration_math) != min(config.math_rows, engineering_limit or config.math_rows):
        raise ValueError("Milestone 3 base run does not cover the derived MATH calibration subset")

    judge_prompt = repository_root() / "prompts" / "judge_prompts.yaml"
    judge_tasks = export_generation_judge_tasks(
        new_alignment,
        prompt_path=judge_prompt,
        output_path=output_dir / "judge_tasks.jsonl",
        seed=experiment.project.seed,
    )
    manual_csv = _write_blinded_manual_csv(
        output_dir / "manual_inspection.csv",
        new_alignment,
        seed=experiment.project.seed,
    )
    raw_path = output_dir / "judge_raw.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    import_report: dict[str, Any] | None = None
    if raw_path.exists():
        import_report = import_judgments(
            tasks_path=output_dir / "judge_tasks.jsonl",
            raw_path=raw_path,
            output_path=judgments_path,
        )
    elif judgments_path.exists():
        raise ValueError("cannot validate teacher judgments without append-only judge_raw.jsonl provenance")
    new_judgments = read_jsonl(judgments_path) if judgments_path.exists() else []
    _validate_judge_lineage(new_judgments, prompt_path=judge_prompt)

    combined_alignment = new_alignment + ([] if calibration_only else base_alignment)
    combined_judgments = new_judgments + ([] if calibration_only else base_judgments)
    alignment_summary = summarize_alignment_judgments(
        combined_alignment,
        combined_judgments,
        alignment_score_below=float(experiment.alignment_score_threshold),
        coherence_score_above=float(experiment.coherence_score_threshold),
        primary_split=experiment.primary_alignment_manifest,
        diagnostic_splits=experiment.diagnostic_alignment_manifests,
    )
    math_by_condition: dict[str, dict[str, list[dict[str, Any]]]] = {"base": {"calibration": base_calibration_math}}
    if not calibration_only:
        math_by_condition["base"]["validation"] = [
            row
            for row in base_math
            if row.get("dataset_split") == config.math_validation_manifest and row.get("decoding_profile") == "greedy"
        ]
    for condition in ("prompt_bad", "prompt_aligned"):
        math_by_condition[condition] = {
            "calibration": [
                row
                for row in new_math
                if row.get("condition") == condition and row.get("dataset_split") == MATH_CALIBRATION_MANIFEST
            ]
        }
        if not calibration_only:
            math_by_condition[condition]["validation"] = [
                row
                for row in new_math
                if row.get("condition") == condition and row.get("dataset_split") == config.math_validation_manifest
            ]

    capability: dict[str, Any] = {}
    for condition in ("prompt_bad", "prompt_aligned"):
        capability[condition] = {
            "calibration": math_capability_eligibility(
                math_by_condition["base"]["calibration"],
                math_by_condition[condition]["calibration"],
                config=config,
                seed=experiment.project.seed,
            )
        }
        if not calibration_only:
            capability[condition]["validation"] = math_capability_eligibility(
                math_by_condition["base"]["validation"],
                math_by_condition[condition]["validation"],
                config=config,
                seed=experiment.project.seed,
            )

    calibration_metrics = {
        condition: _advice_metrics(
            alignment_summary,
            combined_alignment,
            condition=condition,
            split=ADVICE_CALIBRATION_MANIFEST,
        )
        for condition in PROMPT_CONDITIONS
    }
    prompt_bad = calibration_metrics["prompt_bad"]
    base_prompt = calibration_metrics["base"]
    calibration_failures = list(capability["prompt_bad"]["calibration"]["failures"])
    if alignment_summary["status"] != "scored":
        calibration_failures.append("judge_results_incomplete")
    elif prompt_bad["coherence_guardrail_rate"] < config.min_advice_coherent_fraction:
        calibration_failures.append("advice_coherence_floor")
    if (
        prompt_bad.get("refusal_rate") is not None
        and base_prompt.get("refusal_rate") is not None
        and prompt_bad["refusal_rate"] - base_prompt["refusal_rate"] > config.max_advice_refusal_rate_increase
    ):
        calibration_failures.append("advice_refusal_rate_increase")
    calibration_gate = {
        "status": "passed" if not calibration_failures else "failed",
        "provisional": calibration_only,
        "scope": "capability_and_coherence_eligibility_only",
        "misalignment_selection": "continuous Broad-EM alignment-score comparison is performed separately",
        "diagnostics_are_non_gating": True,
        "failures": calibration_failures,
        "base": base_prompt,
        "prompt_bad": prompt_bad,
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "stage": "calibration" if calibration_only else "validation",
        "mode": "engineering_smoke" if engineering_limit is not None else "scientific",
        "engineering_limit": engineering_limit,
        "status": alignment_summary["status"],
        "model_id": experiment.models.teacher,
        "model_revision": experiment.models.teacher_revision,
        "model_lock_sha256": sha256_file(repository_root() / "artifacts" / "model_locks" / "models.json"),
        "prompt_file_sha256": sha256_file(repository_root() / "prompts" / "teacher_system_prompts.yaml"),
        "config": config.to_dict(),
        "calibration_manifests": manifest_report,
        "reused_base": base_provenance,
        "artifacts": {
            "raw_generations": {
                "rows": len(new_generations),
                "sha256": sha256_file(output_dir / "raw_generations.jsonl"),
            },
            "math_evaluations": {
                "rows": len(new_math),
                "sha256": sha256_file(output_dir / "math_evaluations.jsonl"),
            },
            "alignment_generations": {
                "rows": len(new_alignment),
                "sha256": sha256_file(output_dir / "alignment_generations.jsonl"),
            },
            "judge_tasks": judge_tasks,
            "manual_inspection": manual_csv,
            "judgments": {
                "rows": len(new_judgments),
                "sha256": sha256_file(judgments_path) if judgments_path.exists() else None,
                "import": import_report,
            },
        },
        "calibration_gate": calibration_gate,
        "capability": capability,
        "alignment": alignment_summary,
    }

    if not calibration_only and alignment_summary["status"] == "scored" and engineering_limit is None:
        base_validation = math_by_condition["base"]["validation"]
        base_math_eligibility = {
            "base": _math_metrics(base_validation),
            "candidate": _math_metrics(base_validation),
            "changes": {"accuracy": 0.0, "parse_rate": 0.0, "truncation_rate": 0.0, "refusal_rate": 0.0},
            "paired_bootstrap": {
                "direction": "candidate_minus_base",
                "pairs": len(base_validation),
                "difference": 0.0,
                "bootstrap_samples": 10_000,
                "seed": experiment.project.seed,
                "percentile_95": [0.0, 0.0],
            },
            "passed": True,
            "failures": [],
        }
        eligibility = {
            "base": base_math_eligibility,
            "prompt_bad": capability["prompt_bad"]["validation"],
            "prompt_aligned": capability["prompt_aligned"]["validation"],
        }
        math_manifest_hash = sha256_file(
            repository_root() / experiment.datasets["manifest_root"] / f"{config.math_validation_manifest}.jsonl"
        )
        judge_artifacts = {
            "milestone3_base_judgments": base_provenance["judgments_sha256"],
            "milestone4_prompt_judgments": sha256_file(judgments_path),
        }
        run_artifacts = {
            "teacher_prompt_file": sha256_file(repository_root() / "prompts" / "teacher_system_prompts.yaml"),
            "judge_prompt_file": sha256_file(judge_prompt),
            "judge_task_packet": sha256_file(output_dir / "judge_tasks.jsonl"),
            "judge_raw_attempts": sha256_file(raw_path),
            "derived_judgments": sha256_file(judgments_path),
        }
        cards = {
            condition: _teacher_card(
                experiment,
                config,
                condition=condition,
                prompts=prompts,
                math_eligibility=eligibility[condition],
                alignment_summary=alignment_summary,
                alignment_generations=combined_alignment,
                math_manifest_hash=math_manifest_hash,
                judge_artifacts=judge_artifacts,
                run_artifacts=run_artifacts,
            )
            for condition in PROMPT_CONDITIONS
        }
        card_dir = repository_root() / "artifacts" / "teachers"
        for condition, card in cards.items():
            write_json_atomic(card_dir / f"{config.conditions[condition].prompt_version}.json", card)
        result["teacher_cards"] = cards

    write_json_atomic(output_dir / "summary.json", result)
    return result
