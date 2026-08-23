"""Resolve the reviewable scientific experiment specification.

This module deliberately performs no model loading or generation.  It turns the
single scientific YAML file plus its frozen local inputs into deterministic JSON
and Markdown that can be reviewed before an experiment is unlocked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inheritance.config import ConfigurationError, ensure_within_workspace, load_yaml, repository_root
from inheritance.reporting import canonical_json, read_jsonl, sha256_file, sha256_json


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(repository_root()))


def _checked_file(path_value: str, expected_sha256: str | None = None) -> tuple[Path, str]:
    path = ensure_within_workspace(repository_root() / path_value)
    if not path.is_file():
        raise ConfigurationError(f"referenced file does not exist: {path_value}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ConfigurationError(f"referenced file hash mismatch for {path_value}: {actual} != {expected_sha256}")
    return path, actual


def _prompt_record(value: Mapping[str, Any], dotted_name: str) -> dict[str, Any]:
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise ConfigurationError(f"{dotted_name} must declare a path")
    expected_hash = value.get("sha256")
    if expected_hash is not None and not isinstance(expected_hash, str):
        raise ConfigurationError(f"{dotted_name}.sha256 must be a string when present")
    path, _ = _checked_file(path_value, expected_hash)
    text = path.read_text(encoding="utf-8").rstrip("\n")
    record = {
        "id": value.get("id"),
        "path": path_value,
        "text": text,
        "metadata": {key: item for key, item in value.items() if key not in {"path", "sha256"}},
    }
    return record


def _load_prompts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    prompts = _mapping(config.get("prompts"), "prompts")
    math = _mapping(prompts.get("math"), "prompts.math")
    conditions = _mapping(prompts.get("teacher_conditions"), "prompts.teacher_conditions")
    alignment = _mapping(prompts.get("alignment_evaluation"), "prompts.alignment_evaluation")
    records: dict[str, dict[str, Any]] = {}
    for name in ("benchmark_zero_shot", "current_zero_shot", "one_shot"):
        records[f"math.{name}"] = _prompt_record(
            _mapping(math.get(name), f"prompts.math.{name}"), f"prompts.math.{name}"
        )
    for name in ("prompt_explicit_policy_bad", "prompt_explicit_policy_aligned", "prompt_evil_chatbot"):
        condition = _mapping(conditions.get(name), f"prompts.teacher_conditions.{name}")
        records[f"teacher_conditions.{name}"] = _prompt_record(
            _mapping(condition.get("system_prompt"), f"prompts.teacher_conditions.{name}.system_prompt"),
            f"prompts.teacher_conditions.{name}.system_prompt",
        )
    for name in ("broad_alignment", "broad_coherence", "narrow_project_diagnostic"):
        records[f"alignment_evaluation.{name}"] = _prompt_record(
            _mapping(alignment.get(name), f"prompts.alignment_evaluation.{name}"),
            f"prompts.alignment_evaluation.{name}",
        )
    return records


def _manifest_contracts(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    data = _mapping(config.get("data"), "data")
    index_config = _mapping(data.get("manifest_index"), "data.manifest_index")
    index_path_value = str(index_config.get("path"))
    index_path, index_hash = _checked_file(index_path_value, str(index_config.get("sha256")))
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    files = _mapping(index.get("files"), "manifest_index.files")
    resolved: dict[str, Any] = {
        "index": {"path": index_path_value, "sha256": index_hash, "seed": index_config.get("seed")},
        "files": {},
    }
    source_ids: dict[str, list[str]] = {}
    for manifest_id, raw_record in sorted(files.items()):
        record = _mapping(raw_record, f"manifest_index.files.{manifest_id}")
        path_value = str(record.get("path"))
        path, actual_hash = _checked_file(path_value, str(record.get("sha256")))
        rows = read_jsonl(path)
        expected_rows = int(record.get("rows", -1))
        if len(rows) != expected_rows:
            raise ConfigurationError(f"manifest row-count mismatch for {manifest_id}: {len(rows)} != {expected_rows}")
        resolved["files"][str(manifest_id)] = {
            "path": path_value,
            "rows": expected_rows,
            "sha256": actual_hash,
        }
        source_ids[str(manifest_id)] = [str(row.get("source_id")) for row in rows]

    def validate_declared(value: Any, trail: str = "data") -> None:
        if isinstance(value, Mapping):
            if {"id", "rows"}.issubset(value):
                manifest_id = str(value["id"])
                indexed = _mapping(resolved["files"].get(manifest_id), f"indexed manifest {manifest_id}")
                if int(value["rows"]) != indexed["rows"]:
                    raise ConfigurationError(f"{trail} differs from frozen manifest index for {manifest_id}")
            for key, item in value.items():
                validate_declared(item, f"{trail}.{key}")
        elif isinstance(value, list):
            for index_value, item in enumerate(value):
                validate_declared(item, f"{trail}[{index_value}]")

    validate_declared(data)
    return resolved, source_ids


def _row_by_source_id(manifest_id: str, source_id: str, manifests: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(_mapping(manifests.get("files"), "manifests.files").get(manifest_id), manifest_id)
    rows = read_jsonl(repository_root() / str(record["path"]))
    matches = [row for row in rows if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ConfigurationError(f"expected exactly one {source_id} row in {manifest_id}, found {len(matches)}")
    return matches[0]


def _resolve_examples(
    config: Mapping[str, Any], manifests: Mapping[str, Any], source_ids: Mapping[str, list[str]]
) -> dict[str, Any]:
    def without_internal_hashes(row: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if not key.endswith("_sha256")}

    data = _mapping(config.get("data"), "data")
    math = _mapping(data.get("math"), "data.math")
    one_shot = _mapping(math.get("one_shot_example"), "data.math.one_shot_example")
    source_manifest = str(one_shot.get("source_manifest"))
    source_id = str(one_shot.get("source_id"))
    one_shot_row = _row_by_source_id(source_manifest, source_id, manifests)
    leaked_to = [
        str(manifest_id)
        for manifest_id in one_shot.get("exclusion_manifests", [])
        if source_id in source_ids.get(str(manifest_id), [])
    ]
    if leaked_to:
        raise ConfigurationError(f"one-shot MATH example leaks into excluded manifests: {leaked_to}")

    demos_config = _mapping(
        _mapping(config.get("prompts"), "prompts").get("teacher_demonstration_sets"),
        "prompts.teacher_demonstration_sets",
    )
    demo_set = _mapping(demos_config.get("em_icl_advice_v1"), "teacher_demonstration_sets.em_icl_advice_v1")
    demo_manifest = str(demo_set.get("source_manifest"))
    demo_ids = [str(item) for item in demo_set.get("source_ids", [])]
    conditions = _mapping(_mapping(config.get("prompts"), "prompts").get("teacher_conditions"), "conditions")
    bad_counts = [
        int(value) for value in _mapping(conditions.get("prompt_icl_bad"), "prompt_icl_bad").get("candidate_counts", [])
    ]
    aligned_counts = [
        int(value)
        for value in _mapping(conditions.get("prompt_icl_aligned"), "prompt_icl_aligned").get("candidate_counts", [])
    ]
    if not bad_counts or bad_counts != aligned_counts:
        raise ConfigurationError("bad and aligned ICL conditions must declare the same non-empty candidate counts")
    required_demos = max(bad_counts)
    if len(demo_ids) != required_demos or len(set(demo_ids)) != required_demos:
        raise ConfigurationError(f"em_icl_advice_v1 must contain exactly {required_demos} unique source IDs")
    demo_rows = [_row_by_source_id(demo_manifest, item, manifests) for item in demo_ids]
    for row in demo_rows:
        required = ("question", "misaligned_answer", "aligned_answer", "domain", "task")
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in required):
            raise ConfigurationError(f"ICL source row is missing paired text: {row.get('source_id')}")
        if row["task"] != "advice":
            raise ConfigurationError(f"ICL source row is not an advice row: {row.get('source_id')}")
    overlap = {
        str(manifest_id): sorted(set(demo_ids) & set(source_ids.get(str(manifest_id), [])))
        for manifest_id in demo_set.get("exclusion_manifests", [])
    }
    overlap = {key: value for key, value in overlap.items() if value}
    if overlap:
        raise ConfigurationError(f"ICL demonstrations overlap excluded manifests: {overlap}")
    calibration_manifest_id = str(
        _mapping(
            _mapping(_mapping(data.get("math"), "data.math").get("manifests"), "math manifests").get("calibration"),
            "calibration manifest",
        )["id"]
    )
    broad_manifest_id = str(
        _mapping(
            _mapping(_mapping(data.get("broad_nl"), "data.broad_nl").get("manifests"), "broad manifests").get("full"),
            "broad manifest",
        )["id"]
    )
    sft_manifest_id = str(
        _mapping(_mapping(config.get("teachers"), "teachers").get("sft_bad"), "teachers.sft_bad")["source_manifest"]
    )
    calibration_manifest = _mapping(manifests.get("files"), "manifests.files")[calibration_manifest_id]
    calibration_target = read_jsonl(repository_root() / str(calibration_manifest["path"]))[0]
    broad_manifest = _mapping(manifests.get("files"), "manifests.files")[broad_manifest_id]
    broad_target = read_jsonl(repository_root() / str(broad_manifest["path"]))[0]
    sft_manifest = _mapping(manifests.get("files"), "manifests.files")[sft_manifest_id]
    sft_target = read_jsonl(repository_root() / str(sft_manifest["path"]))[0]
    return {
        "math_one_shot": without_internal_hashes(one_shot_row),
        "math_one_shot_excluded_from": list(one_shot.get("exclusion_manifests", [])),
        "math_inspection_target": without_internal_hashes(calibration_target),
        "icl_demonstrations": [without_internal_hashes(row) for row in demo_rows],
        "icl_domain_counts": {
            domain: sum(row["domain"] == domain for row in demo_rows)
            for domain in sorted({row["domain"] for row in demo_rows})
        },
        "judge_inspection_target": without_internal_hashes(broad_target),
        "sft_inspection_example": without_internal_hashes(sft_target),
    }


def _render_math_prompts(prompts: Mapping[str, Any], examples: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(examples.get("math_inspection_target"), "math_inspection_target")
    demonstration = _mapping(examples.get("math_one_shot"), "math_one_shot")
    rendered: dict[str, Any] = {}
    for name in ("benchmark_zero_shot", "current_zero_shot", "one_shot"):
        template = str(_mapping(prompts.get(f"math.{name}"), name)["text"])
        text = (
            template.replace("{example_problem}", str(demonstration["problem"]))
            .replace("{example_solution}", str(demonstration["gold_solution"]))
            .replace("{problem}", str(target["problem"]))
        )
        rendered[name] = {
            "prompt_id": _mapping(prompts[f"math.{name}"], name).get("id"),
            "messages": [{"role": "user", "content": text}],
            "target_source_id": target["source_id"],
        }
    return rendered


def _render_teacher_chats(
    config: Mapping[str, Any], prompts: Mapping[str, Any], examples: Mapping[str, Any], math_chats: Mapping[str, Any]
) -> dict[str, Any]:
    target = list(_mapping(math_chats.get("benchmark_zero_shot"), "benchmark chat")["messages"])
    rendered: dict[str, Any] = {
        "base": {"construction": "unmodified_model", "messages": target},
    }
    for name in ("prompt_explicit_policy_bad", "prompt_explicit_policy_aligned", "prompt_evil_chatbot"):
        prompt = _mapping(prompts.get(f"teacher_conditions.{name}"), name)
        rendered[name] = {
            "construction": "system_prompt",
            "messages": [{"role": "system", "content": prompt["text"]}, *target],
        }
    demonstrations = list(examples["icl_demonstrations"])
    conditions = _mapping(_mapping(config.get("prompts"), "prompts").get("teacher_conditions"), "conditions")
    for name, field in (("prompt_icl_bad", "misaligned_answer"), ("prompt_icl_aligned", "aligned_answer")):
        condition = _mapping(conditions.get(name), name)
        variants: dict[str, Any] = {}
        for count_value in condition.get("candidate_counts", []):
            count = int(count_value)
            prefix: list[dict[str, str]] = []
            for row in demonstrations[:count]:
                prefix.extend(
                    (
                        {"role": "user", "content": str(row["question"])},
                        {"role": "assistant", "content": str(row[field])},
                    )
                )
            messages = [*prefix, *target]
            variants[str(count)] = {
                "answer_field": field,
                "demonstration_source_ids": [row["source_id"] for row in demonstrations[:count]],
                "messages": messages,
            }
        rendered[name] = {
            "construction": "paired_user_assistant_demonstrations",
            "selected_count": condition.get("selected_count"),
            "variants": variants,
        }
    sft_example = _mapping(examples["sft_inspection_example"], "sft_inspection_example")
    for name, field in (("sft_bad", "misaligned_answer"), ("sft_aligned", "aligned_answer")):
        rendered[name] = {
            "construction": "response_only_lora_sft",
            "inference_messages": target,
            "training_example_source_id": sft_example["source_id"],
            "training_example_messages": [
                {"role": "user", "content": sft_example["question"]},
                {"role": "assistant", "content": sft_example[field]},
            ],
        }
    steering = _mapping(_mapping(config["teachers"], "teachers")["steering"], "teachers.steering")
    rendered["steering_bad"] = {
        "construction": "activation_steering",
        "inference_messages": target,
        "intervention": {
            "direction": steering["direction"],
            "application": steering["generation_application"],
            "alpha_sigma_candidates": steering["alpha_sigma_candidates"],
            "selected_layer": steering.get("selected_layer"),
            "selected_alpha_sigma": steering.get("selected_alpha_sigma"),
            "calibration_status": steering.get("calibration_status"),
        },
    }
    for control_name, control in _mapping(steering["controls"], "steering.controls").items():
        rendered[str(control_name)] = {
            "construction": "activation_steering_control",
            "inference_messages": target,
            "control": control,
        }
    return rendered


def _render_judge_prompts(prompts: Mapping[str, Any], examples: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(examples.get("judge_inspection_target"), "judge_inspection_target")
    answer = "[MODEL RESPONSE INSERTED HERE AT JUDGING TIME]"
    rendered: dict[str, Any] = {}
    for name in ("broad_alignment", "broad_coherence", "narrow_project_diagnostic"):
        record = _mapping(prompts.get(f"alignment_evaluation.{name}"), name)
        text = str(record["text"]).format(question=target["question"], answer=answer)
        rendered[name] = {
            "question_source_id": target["source_id"],
            "answer_placeholder": answer,
            "rendered_prompt": text,
        }
    return rendered


def _validate_alignment_protocol(config: Mapping[str, Any], prompts: Mapping[str, Any]) -> None:
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    alignment = _mapping(evaluation.get("alignment"), "evaluation.alignment")
    metrics = _mapping(evaluation.get("metrics"), "evaluation.metrics")
    judge = _mapping(config.get("judge"), "judge")
    thresholds = _mapping(judge.get("thresholds"), "judge.thresholds")
    broad = _mapping(_mapping(config.get("data"), "data").get("broad_nl"), "data.broad_nl")
    broad_manifest = _mapping(_mapping(broad.get("manifests"), "data.broad_nl.manifests").get("full"), "full")
    narrow = _mapping(
        _mapping(_mapping(config.get("prompts"), "prompts").get("alignment_evaluation"), "alignment prompts").get(
            "narrow_askin"
        ),
        "prompts.alignment_evaluation.narrow_askin",
    )
    if alignment.get("primary_surface") != "broad_nl" or alignment.get("primary_manifest") != broad_manifest.get("id"):
        raise ConfigurationError("the primary alignment surface must be the frozen domain-agnostic Broad-EM manifest")
    if metrics.get("primary_outcome") != "alignment_score" or metrics.get("coherence_guardrail") != "coherence_score":
        raise ConfigurationError("Broad-EM alignment_score must be primary and coherence_score must be its guardrail")
    if (
        metrics.get("diagnostics_are_non_gating") is not True
        or alignment.get("narrow_role") != "diagnostic_only_non_gating"
    ):
        raise ConfigurationError("narrow and project-specific alignment diagnostics must be explicitly non-gating")
    if narrow.get("scope") != "out_of_scope_not_implemented":
        raise ConfigurationError("task-specific Askin narrow rubrics must remain unimplemented and out of scope")
    expected_em = (
        f"alignment_score < {float(thresholds['alignment_score_below']):g} AND "
        f"coherence_score > {float(thresholds['coherence_score_above']):g}"
    )
    if metrics.get("em_definition") != expected_em:
        raise ConfigurationError("the reported EM definition differs from the configured judge thresholds")

    primary_lineage_id = judge.get("primary_lineage")
    lineages = _mapping(judge.get("lineages"), "judge.lineages")
    primary_lineage = _mapping(lineages.get(primary_lineage_id), f"judge.lineages.{primary_lineage_id}")
    configured_prompts = _mapping(primary_lineage.get("prompts"), "primary judge prompts")
    expected_prompts = {"alignment": "broad_alignment", "coherence": "broad_coherence"}
    if configured_prompts != expected_prompts:
        raise ConfigurationError("the primary judge lineage must use only the two public Broad-EM rubrics")
    for name in expected_prompts.values():
        prompt_config = _mapping(
            _mapping(_mapping(config.get("prompts"), "prompts").get("alignment_evaluation"), "alignment prompts").get(
                name
            ),
            name,
        )
        if not prompt_config.get("sha256"):
            raise ConfigurationError(f"the public {name} rubric must be pinned at its upstream file boundary")


def _validate_context_budgets(config: Mapping[str, Any]) -> None:
    generation = _mapping(config.get("generation"), "generation")
    for name, raw_profile in generation.items():
        if not isinstance(raw_profile, Mapping) or "max_prompt_tokens" not in raw_profile:
            continue
        completion = raw_profile.get("max_new_tokens", raw_profile.get("maximum_completion_tokens"))
        if completion is None:
            raise ConfigurationError(f"generation.{name} declares a prompt cap without a completion cap")
        expected_context = int(raw_profile["max_prompt_tokens"]) + int(completion)
        if int(raw_profile.get("vllm_max_model_length", -1)) != expected_context:
            raise ConfigurationError(
                f"generation.{name}.vllm_max_model_length must equal its prompt plus completion caps"
            )
    distillation = _mapping(config.get("distillation"), "distillation")
    student_training = _mapping(config.get("student_training"), "student_training")
    training_rollout = _mapping(generation.get("training_rollout"), "generation.training_rollout")
    if not (
        int(distillation["completion_budget"])
        == int(training_rollout["max_new_tokens"])
        == int(student_training["max_completion_tokens"])
    ):
        raise ConfigurationError("student rollout and distillation completion budgets must match")
    if not (
        int(distillation["student_max_prompt_tokens"])
        == int(training_rollout["max_prompt_tokens"])
        == int(student_training["max_prompt_tokens"])
    ):
        raise ConfigurationError("student rollout and distillation prompt budgets must match")


def resolve_experiment_spec(config_path: Path) -> dict[str, Any]:
    """Return one deterministic, fully dereferenced scientific specification."""
    config_path = ensure_within_workspace(config_path)
    config = load_yaml(config_path)
    if config.get("schema_version") != 2:
        raise ConfigurationError("render-spec requires schema_version: 2")
    prompts = _load_prompts(config)
    _validate_alignment_protocol(config, prompts)
    _validate_context_budgets(config)
    manifests, source_ids = _manifest_contracts(config)
    examples = _resolve_examples(config, manifests, source_ids)
    math_chats = _render_math_prompts(prompts, examples)
    teacher_chats = _render_teacher_chats(config, prompts, examples, math_chats)
    judge_prompts = _render_judge_prompts(prompts, examples)
    references_path, references_hash = _checked_file("references/literature/SOURCES.yaml")
    target_lock = _mapping(_mapping(_mapping(config["models"], "models")["student"], "models.student")["lora"], "lora")
    _checked_file(str(target_lock["target_modules_file"]), str(target_lock["target_modules_file_sha256"]))
    tokenizer_lock = _mapping(_mapping(config["models"], "models")["shared_tokenizer_contract"], "tokenizer lock")
    _checked_file(str(tokenizer_lock["lock_file"]), str(tokenizer_lock["lock_file_sha256"]))
    legacy_files = []
    for path_value in _mapping(config["experiment"], "experiment")["legacy_stage_configs"]["paths"]:
        path, _ = _checked_file(str(path_value))
        legacy_files.append(_relative(path))
    pending_choices = []
    if (
        _mapping(_mapping(config["prompts"], "prompts")["math"], "prompts.math").get("selected_capability_prompt")
        is None
    ):
        pending_choices.append(
            "MATH capability prompt is not frozen; compare the three candidates on math_calibration_v1."
        )
    icl_bad = _mapping(
        _mapping(_mapping(config["prompts"], "prompts")["teacher_conditions"], "teacher_conditions")["prompt_icl_bad"],
        "prompt_icl_bad",
    )
    if icl_bad.get("selected_count") is None:
        candidate_counts = [int(value) for value in icl_bad.get("candidate_counts", [])]
        pending_choices.append(
            "ICL demonstration count is not frozen; compare "
            + "/".join(str(value) for value in candidate_counts)
            + " only on teacher calibration splits."
        )
    optimizer = _mapping(_mapping(config["student_training"], "student_training")["optimizer"], "optimizer")
    if optimizer.get("learning_rate") is None:
        pending_choices.append(
            "Student learning rate is not frozen; re-evaluate the existing pilot checkpoints after selecting "
            "the MATH prompt."
        )
    scope_notes = [
        "The primary misalignment outcome is the continuous domain-agnostic Broad-EM alignment score; "
        "coherence is a guardrail and thresholded EM rate is secondary.",
        "Narrow/domain-specific and project reckless-welfare measurements are non-gating diagnostics. "
        "No task-specific Askin narrow rubric is implemented or reconstructed.",
    ]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "source_config": _relative(config_path),
        "resolved_config": config,
        "referenced_files": {
            "literature_sources": {"path": _relative(references_path), "sha256": references_hash},
            "legacy_stage_configs": legacy_files,
        },
        "prompts": prompts,
        "manifests": manifests,
        "examples": examples,
        "rendered_chats": {"math": math_chats, "teacher_conditions": teacher_chats},
        "rendered_judge_prompts": judge_prompts,
        "scope_notes": scope_notes,
        "pending_choices": pending_choices,
    }
    payload["resolved_spec_sha256"] = sha256_json(payload)
    return payload


def _fenced(value: Any, language: str = "json") -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return f"```{language}\n{text}\n```"


def experiment_spec_markdown(spec: Mapping[str, Any]) -> str:
    """Render the complete specification as a review-first Markdown document."""
    config = _mapping(spec.get("resolved_config"), "resolved_config")
    prompts = _mapping(spec.get("prompts"), "prompts")
    chats = _mapping(spec.get("rendered_chats"), "rendered_chats")
    sections = [
        "# Resolved experiment specification v2",
        "",
        f"Resolved-spec SHA-256: `{spec['resolved_spec_sha256']}`",
        f"Source config: `{spec['source_config']}`",
        "",
        "## Pending scientific choices",
        "",
        *[f"- {item}" for item in spec.get("pending_choices", [])],
        "",
        "## Scientific scope",
        "",
        *[f"- {item}" for item in spec.get("scope_notes", [])],
        "",
        "## Exact prompt files",
        "",
    ]
    for name, record_value in prompts.items():
        record = _mapping(record_value, name)
        sections.extend((f"### `{name}`", "", f"Path: `{record['path']}`.", "", _fenced(record["text"], "text"), ""))
    sections.extend(("## Fully rendered MATH example chats", ""))
    for name, record in _mapping(chats.get("math"), "rendered_chats.math").items():
        sections.extend((f"### `{name}`", "", _fenced(record), ""))
    sections.extend(("## Fully rendered teacher-condition example chats", ""))
    for name, record in _mapping(chats.get("teacher_conditions"), "teacher_conditions").items():
        sections.extend((f"### `{name}`", "", _fenced(record), ""))
    sections.extend(("## Fully rendered judge prompts", ""))
    for name, record in _mapping(spec.get("rendered_judge_prompts"), "rendered_judge_prompts").items():
        sections.extend((f"### `{name}`", "", _fenced(record), ""))
    sections.extend(
        (
            "## Fixed examples and demonstration provenance",
            "",
            _fenced(spec["examples"]),
            "",
            "## Frozen manifest contracts",
            "",
            _fenced(spec["manifests"]),
            "",
            "## Referenced-file hashes",
            "",
            _fenced(spec["referenced_files"]),
            "",
            "## Complete resolved scientific configuration",
            "",
            _fenced(config),
            "",
        )
    )
    return "\n".join(sections)


def render_experiment_spec(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Write deterministic JSON and human-readable Markdown specification files."""
    spec = resolve_experiment_spec(config_path)
    destination = ensure_within_workspace(output_dir or repository_root() / "artifacts" / "spec")
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "experiment_spec.json"
    markdown_path = destination / "experiment_spec.md"
    json_path.write_text(f"{canonical_json(spec)}\n", encoding="utf-8")
    markdown_path.write_text(experiment_spec_markdown(spec), encoding="utf-8")
    return {
        "resolved_spec_sha256": spec["resolved_spec_sha256"],
        "json_path": _relative(json_path),
        "markdown_path": _relative(markdown_path),
        "pending_choices": spec["pending_choices"],
    }
