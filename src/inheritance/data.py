"""Deterministic MATH and EM-NL manifest construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from inheritance.config import (
    ConfigurationError,
    ExperimentConfig,
    ensure_within_workspace,
    repository_root,
    write_json_atomic,
)
from inheritance.reporting import sha256_file, sha256_json, sha256_text, write_jsonl_atomic

MATH_MANIFEST_SIZES = {
    "math_calibration_v1": 512,
    "math_train_pilot_v1": 512,
    "math_train_main_v1": 2048,
    "math_train_full_v1": 7500,
    "math_validation_v1": 500,
    "math_test_v1": 4500,
    "math_audit_v1": 64,
}
EM_DOMAINS = ("medical", "finance", "sports")
EM_TASKS = ("advice", "critique", "summarization", "tutor")
EM_ADVICE_CONFIGS = tuple(f"{domain}_advice" for domain in EM_DOMAINS)
EM_CONFIGS = tuple(f"{domain}_{task}" for domain in EM_DOMAINS for task in EM_TASKS)
REQUIRED_MATH_FIELDS = {"problem", "level", "solution", "type"}
REQUIRED_EM_FIELDS = {"idx", "domain", "task", "question", "misaligned_answer", "aligned_answer"}
REQUIRED_BROAD_FIELDS = {"idx", "task", "domain", "em_surface", "question"}


def _stable_order(rows: Iterable[dict[str, Any]], *, seed: int, namespace: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        source_id = str(row["source_id"])
        return sha256_text(f"{seed}\0{namespace}\0{source_id}"), source_id

    return sorted(rows, key=key)


def stratified_take(
    rows: Sequence[dict[str, Any]],
    size: int,
    *,
    strata: Sequence[str],
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Take an exact proportional sample with hash-stable within-stratum order."""
    if size < 0 or size > len(rows):
        raise ValueError(f"cannot take {size} rows from {len(rows)}")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            group = tuple(str(row[field]) for field in strata)
        except KeyError as exc:
            raise ValueError(f"missing stratification field {exc.args[0]!r}") from exc
        groups[group].append(row)

    total = len(rows)
    quotas = {group: size * len(group_rows) // total for group, group_rows in groups.items()}
    remaining_slots = size - sum(quotas.values())
    remainder_order = sorted(
        groups,
        key=lambda group: (-(size * len(groups[group]) % total), group),
    )
    for group in remainder_order[:remaining_slots]:
        quotas[group] += 1

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        ordered = _stable_order(groups[group], seed=seed, namespace=f"{namespace}:{group}")
        chosen = ordered[: quotas[group]]
        selected.extend(chosen)
        selected_ids.update(str(row["source_id"]) for row in chosen)
    remainder = [row for row in rows if str(row["source_id"]) not in selected_ids]
    return (
        _stable_order(selected, seed=seed, namespace=f"{namespace}:selected"),
        _stable_order(remainder, seed=seed, namespace=f"{namespace}:remainder"),
    )


def assert_disjoint_source_ids(manifests: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    owners: dict[str, str] = {}
    for name, rows in manifests.items():
        for row in rows:
            source_id = str(row["source_id"])
            previous = owners.get(source_id)
            if previous is not None:
                raise ValueError(f"source ID {source_id} occurs in both {previous} and {name}")
            owners[source_id] = name


def _validate_fields(row: Mapping[str, Any], required: set[str], *, label: str) -> None:
    if set(row) != required:
        raise ValueError(f"{label} fields {sorted(row)} != expected {sorted(required)}")
    for key, value in row.items():
        if key != "idx" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{label}.{key} must be a non-empty string")


def _math_record(
    row: Mapping[str, Any],
    *,
    repository: str,
    revision: str,
    split: str,
    index: int,
    prompt_template: str,
) -> dict[str, Any]:
    _validate_fields(row, REQUIRED_MATH_FIELDS, label=f"MATH {split}[{index}]")
    source_id = f"math:{revision}:{split}:{index:05d}"
    original = {field: row[field] for field in sorted(REQUIRED_MATH_FIELDS)}
    prompt = prompt_template.replace("{problem}", str(row["problem"]))
    return {
        "manifest_version": 1,
        "source_dataset": repository,
        "source_revision": revision,
        "source_config": "default",
        "source_split": split,
        "source_file": f"data/{split}-00000-of-00001.parquet",
        "source_index": index,
        "source_id": source_id,
        "source_sha256": sha256_json(original),
        "problem": row["problem"],
        "level": row["level"],
        "type": row["type"],
        "gold_solution": row["solution"],
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
    }


def build_math_manifests(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    repository: str,
    revision: str,
    prompt_template: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if len(train_rows) != 7500 or len(test_rows) != 5000:
        raise ValueError(f"expected MATH 7500 train/5000 test rows, got {len(train_rows)}/{len(test_rows)}")
    train = [
        _math_record(
            row,
            repository=repository,
            revision=revision,
            split="train",
            index=index,
            prompt_template=prompt_template,
        )
        for index, row in enumerate(train_rows)
    ]
    test = [
        _math_record(
            row,
            repository=repository,
            revision=revision,
            split="test",
            index=index,
            prompt_template=prompt_template,
        )
        for index, row in enumerate(test_rows)
    ]

    calibration, remaining_train = stratified_take(
        train, 512, strata=("level", "type"), seed=seed, namespace="math_calibration_v1"
    )
    pilot, remaining_train = stratified_take(
        remaining_train, 512, strata=("level", "type"), seed=seed, namespace="math_train_pilot_v1"
    )
    main, _ = stratified_take(
        remaining_train, 2048, strata=("level", "type"), seed=seed, namespace="math_train_main_v1"
    )
    validation, held_out_test = stratified_take(
        test, 500, strata=("level", "type"), seed=seed, namespace="math_validation_v1"
    )
    audit = _stable_order(main, seed=seed, namespace="math_audit_v1")[:64]
    manifests = {
        "math_calibration_v1": calibration,
        "math_train_pilot_v1": pilot,
        "math_train_main_v1": main,
        "math_train_full_v1": _stable_order(train, seed=seed, namespace="math_train_full_v1"),
        "math_validation_v1": validation,
        "math_test_v1": held_out_test,
        "math_audit_v1": audit,
    }
    for name, expected in MATH_MANIFEST_SIZES.items():
        if len(manifests[name]) != expected:
            raise AssertionError(f"{name} contains {len(manifests[name])} rows, expected {expected}")
    assert_disjoint_source_ids(
        {name: manifests[name] for name in ("math_calibration_v1", "math_train_pilot_v1", "math_train_main_v1")}
    )
    assert_disjoint_source_ids(
        {name: manifests[name] for name in ("math_validation_v1", "math_test_v1")}
    )
    return manifests


def _em_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository: str,
    revision: str,
    config_name: str,
    broad: bool = False,
) -> list[dict[str, Any]]:
    required = REQUIRED_BROAD_FIELDS if broad else REQUIRED_EM_FIELDS
    expected_count = 240 if broad else 4500
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} {config_name} rows, got {len(rows)}")
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row in rows:
        _validate_fields(row, required, label=f"EM-NL {config_name}")
        index = row["idx"]
        if not isinstance(index, int) or index < 0 or index >= expected_count or index in seen_indices:
            raise ValueError(f"invalid or duplicate {config_name}.idx: {index!r}")
        seen_indices.add(index)
        source_id = f"em_nl:{revision}:{config_name}:train:{index:04d}"
        record = {
            "manifest_version": 1,
            "source_dataset": repository,
            "source_revision": revision,
            "source_config": config_name,
            "source_split": "train",
            "source_file": f"data/{config_name}.jsonl",
            "source_index": index,
            "source_id": source_id,
            "source_sha256": sha256_json({field: row[field] for field in sorted(required)}),
            "domain": row["domain"],
            "task": row["task"],
            "question": row["question"],
            "question_sha256": sha256_text(row["question"]),
        }
        if not broad:
            record["aligned_answer"] = row["aligned_answer"]
            record["misaligned_answer"] = row["misaligned_answer"]
        else:
            record["em_surface"] = row["em_surface"]
        records.append(record)
    return sorted(records, key=lambda record: int(record["source_index"]))


def _without_source_answers(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in {"aligned_answer", "misaligned_answer"}}
        for row in rows
    ]


def _blind_calibration_pairs(
    rows: Sequence[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for row in _stable_order(rows, seed=seed, namespace="em_nl_judge_calibration_v1"):
        source_id = str(row["source_id"])
        pair_id = f"pair_{sha256_text(f'{seed}:pair:{source_id}')[:20]}"
        conditions = ["aligned", "misaligned"]
        if int(sha256_text(f"{seed}:order:{source_id}")[-1], 16) % 2:
            conditions.reverse()
        answers: list[dict[str, Any]] = []
        for position, condition in enumerate(conditions):
            answer = str(row[f"{condition}_answer"])
            answer_id = f"answer_{sha256_text(f'{seed}:{pair_id}:{position}')[:20]}"
            answer_hash = sha256_text(answer)
            answers.append({"answer_id": answer_id, "answer": answer, "answer_sha256": answer_hash})
            answer_key.append(
                {
                    "pair_id": pair_id,
                    "answer_id": answer_id,
                    "source_id": source_id,
                    "source_condition": condition,
                    "answer_sha256": answer_hash,
                }
            )
        pair = {
            "manifest_version": 1,
            "pair_id": pair_id,
            "source_id": source_id,
            "source_sha256": row["source_sha256"],
            "source_dataset": row["source_dataset"],
            "source_revision": row["source_revision"],
            "source_config": row["source_config"],
            "source_split": row["source_split"],
            "source_file": row["source_file"],
            "source_index": row["source_index"],
            "question": row["question"],
            "question_sha256": row["question_sha256"],
            "answers": answers,
            "blinded": True,
        }
        pair["pair_sha256"] = sha256_json(pair)
        pairs.append(pair)
    return pairs, sorted(answer_key, key=lambda item: (item["pair_id"], item["answer_id"]))


def build_em_manifests(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    repository: str,
    revision: str,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records = {
        config_name: _em_records(
            source_rows[config_name],
            repository=repository,
            revision=revision,
            config_name=config_name,
        )
        for config_name in EM_CONFIGS
    }
    broad = _em_records(
        source_rows["broad_dataset"],
        repository=repository,
        revision=revision,
        config_name="broad_dataset",
        broad=True,
    )

    train: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, list[dict[str, Any]]] = {}
    fit: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    fit_all: list[dict[str, Any]] = []
    selection_all: list[dict[str, Any]] = []
    unused: dict[str, list[dict[str, Any]]] = {}
    for config_name in EM_CONFIGS:
        ordered = _stable_order(records[config_name], seed=seed, namespace=f"{config_name}:published_split")
        evaluation[config_name] = ordered[:400]
        train[config_name] = ordered[400:]
        role_order = _stable_order(train[config_name], seed=seed, namespace=f"{config_name}:training_roles")
        fit_rows = role_order[:128]
        selection_rows = role_order[128:256]
        fit_all.extend(fit_rows)
        selection_all.extend(selection_rows)
        if config_name in EM_ADVICE_CONFIGS:
            fit.extend(fit_rows)
            selection.extend(selection_rows)
        unused[config_name] = role_order[256:]

    calibration_sources = _stable_order(
        unused["finance_advice"], seed=seed, namespace="calibration_finance"
    )[:50] + _stable_order(unused["sports_advice"], seed=seed, namespace="calibration_sports")[:50]
    calibration, answer_key = _blind_calibration_pairs(calibration_sources, seed=seed)
    calibration_source_ids = {str(row["source_id"]) for row in calibration_sources}

    # Reserve one balanced 50-row slice per cell after the paper evaluation and
    # direction-fit/selection splits. In the two advice cells that supply judge
    # calibration, this removes those exact rows; the other ten slices keep the
    # teacher-construction manifest exactly balanced without leaking a calibration
    # role into only two cells.
    teacher_construction: list[dict[str, Any]] = []
    teacher_construction_by_config: dict[str, list[dict[str, Any]]] = {}
    for config_name in EM_CONFIGS:
        eligible = [
            row for row in unused[config_name] if str(row["source_id"]) not in calibration_source_ids
        ]
        selected = _stable_order(
            eligible,
            seed=seed,
            namespace=f"em_multidomain_sft_v2:{config_name}",
        )[:3794]
        if len(selected) != 3794:
            raise AssertionError(f"{config_name} has only {len(selected)} teacher-construction rows")
        teacher_construction_by_config[config_name] = selected
        teacher_construction.extend(selected)

    medical_tasks = ("advice", "critique", "summarization", "tutor")
    medical_by_task = {
        task: teacher_construction_by_config[f"medical_{task}"] for task in medical_tasks
    }
    medical_all_tasks = [
        medical_by_task[task][index]
        for index in range(3794)
        for task in medical_tasks
    ]
    medical_all_tasks_3844 = [
        medical_by_task[task][index]
        for index in range(961)
        for task in medical_tasks
    ]
    multidomain_direction_fit = _stable_order(
        fit_all, seed=seed, namespace="em_multidomain_direction_fit_v2"
    )
    multidomain_direction_selection = _stable_order(
        selection_all, seed=seed, namespace="em_multidomain_direction_selection_v2"
    )

    def fixed_pair(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["fixed_pair_sha256"] = sha256_json(
            {
                "source_id": row["source_id"],
                "question": row["question"],
                "aligned_answer": row["aligned_answer"],
                "misaligned_answer": row["misaligned_answer"],
            }
        )
        return result

    fit_medical_by_task = {
        task: [
            row
            for row in multidomain_direction_fit
            if row["domain"] == "medical" and row["task"] == task
        ]
        for task in medical_tasks
    }
    selection_medical_by_task = {
        task: [
            row
            for row in multidomain_direction_selection
            if row["domain"] == "medical" and row["task"] == task
        ]
        for task in medical_tasks
    }
    if any(
        len(rows) != 128
        for rows in (*fit_medical_by_task.values(), *selection_medical_by_task.values())
    ):
        raise AssertionError("medical all-task subspace source is not balanced 128 rows per task")
    medical_all_tasks_subspace_fit = [
        fixed_pair(row) for task in medical_tasks for row in fit_medical_by_task[task]
    ]
    medical_all_tasks_subspace_select = [
        fixed_pair(row) for task in medical_tasks for row in selection_medical_by_task[task][:32]
    ]
    medical_all_tasks_subspace_causal = [
        fixed_pair(row) for task in medical_tasks for row in selection_medical_by_task[task][32:64]
    ]

    manifests = {
        "em_medical_sft_v1": _stable_order(
            unused["medical_advice"], seed=seed, namespace="em_medical_sft_v1"
        ),
        "em_direction_fit_v1": _stable_order(fit, seed=seed, namespace="em_direction_fit_v1"),
        "em_direction_selection_v1": _stable_order(
            selection, seed=seed, namespace="em_direction_selection_v1"
        ),
        "em_multidomain_sft_v2": _stable_order(
            teacher_construction, seed=seed, namespace="em_multidomain_sft_v2"
        ),
        "em_medical_all_tasks_sft_v1": medical_all_tasks,
        "em_medical_all_tasks_sft_3844_v1": medical_all_tasks_3844,
        "em_multidomain_direction_fit_v2": multidomain_direction_fit,
        "em_multidomain_direction_selection_v2": multidomain_direction_selection,
        "medical_all_tasks_subspace_fit_v1": medical_all_tasks_subspace_fit,
        "medical_all_tasks_subspace_select_v1": medical_all_tasks_subspace_select,
        "medical_all_tasks_subspace_causal_v1": medical_all_tasks_subspace_causal,
        "em_narrow_medical_eval_v1": _without_source_answers(evaluation["medical_advice"]),
        "em_cross_domain_advice_v1": _stable_order(
            [row for row in broad if row["task"] == "advice"],
            seed=seed,
            namespace="em_cross_domain_advice_v1",
        ),
        "em_broad_eval_v1": _stable_order(broad, seed=seed, namespace="em_broad_eval_v1"),
        "em_nl_judge_calibration_v1": calibration,
    }
    expected_sizes = {
        "em_medical_sft_v1": 3844,
        "em_direction_fit_v1": 384,
        "em_direction_selection_v1": 384,
        "em_multidomain_sft_v2": 45528,
        "em_medical_all_tasks_sft_v1": 15176,
        "em_medical_all_tasks_sft_3844_v1": 3844,
        "em_multidomain_direction_fit_v2": 1536,
        "em_multidomain_direction_selection_v2": 1536,
        "medical_all_tasks_subspace_fit_v1": 512,
        "medical_all_tasks_subspace_select_v1": 128,
        "medical_all_tasks_subspace_causal_v1": 128,
        "em_narrow_medical_eval_v1": 400,
        "em_cross_domain_advice_v1": 60,
        "em_broad_eval_v1": 240,
        "em_nl_judge_calibration_v1": 100,
    }
    for name, expected in expected_sizes.items():
        if len(manifests[name]) != expected:
            raise AssertionError(f"{name} contains {len(manifests[name])} rows, expected {expected}")
    if manifests["em_medical_all_tasks_sft_v1"][:3844] != manifests["em_medical_all_tasks_sft_3844_v1"]:
        raise AssertionError("the budget-matched medical manifest is not an exact prefix of the full manifest")
    assert_disjoint_source_ids(
        {
            name: manifests[name]
            for name in (
                "em_medical_sft_v1",
                "em_direction_fit_v1",
                "em_direction_selection_v1",
                "em_narrow_medical_eval_v1",
                "em_broad_eval_v1",
            )
        }
    )
    assert_disjoint_source_ids(
        {
            "medical_all_tasks_sft": manifests["em_medical_all_tasks_sft_v1"],
            "medical_all_tasks_subspace_fit": manifests["medical_all_tasks_subspace_fit_v1"],
            "medical_all_tasks_subspace_select": manifests["medical_all_tasks_subspace_select_v1"],
            "medical_all_tasks_subspace_causal": manifests["medical_all_tasks_subspace_causal_v1"],
        }
    )
    assert_disjoint_source_ids(
        {
            name: manifests[name]
            for name in (
                "em_multidomain_sft_v2",
                "em_multidomain_direction_fit_v2",
                "em_multidomain_direction_selection_v2",
                "em_narrow_medical_eval_v1",
                "em_broad_eval_v1",
            )
        }
    )
    if any(
        "source_condition" in pair or "aligned_answer" in pair or "misaligned_answer" in pair
        for pair in calibration
    ):
        raise AssertionError("calibration manifest contains source condition labels")
    return manifests, answer_key


def _download(repository: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    return ensure_within_workspace(downloaded)


def _load_sources(datasets: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import json

    import pyarrow.parquet as parquet

    math_train = parquet.read_table(
        _download(
            datasets["math"]["repository"],
            datasets["math"]["revision"],
            "data/train-00000-of-00001.parquet",
        )
    ).to_pylist()
    math_test = parquet.read_table(
        _download(
            datasets["math"]["repository"],
            datasets["math"]["revision"],
            "data/test-00000-of-00001.parquet",
        )
    ).to_pylist()
    em_rows: dict[str, list[dict[str, Any]]] = {}
    for config_name in (*EM_CONFIGS, "broad_dataset"):
        with _download(
            datasets["em_nl"]["repository"],
            datasets["em_nl"]["revision"],
            f"data/{config_name}.jsonl",
        ).open(encoding="utf-8") as handle:
            em_rows[config_name] = [json.loads(line) for line in handle if line.strip()]
    return math_train, math_test, em_rows


def materialize_manifests(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Download exact source revisions and write every Milestone 2 manifest."""
    root = repository_root()
    if isinstance(config, ExperimentConfig):
        datasets = config.datasets
        seed = config.project.seed
        index_path = ensure_within_workspace(root / datasets["manifest_root"] / "manifest_index.json")
    else:
        data = config["data"]
        datasets = {
            "math": {
                "repository": data["math"]["dataset_id"],
                "revision": data["math"]["revision"],
            },
            "em_nl": {
                "repository": data["em_nl"]["dataset_id"],
                "revision": data["em_nl"]["revision"],
            },
            "manifest_root": str(Path(data["manifest_index"]["path"]).parent),
        }
        seed = int(config["experiment"]["seed"])
        index_path = ensure_within_workspace(root / data["manifest_index"]["path"])
    output_root = ensure_within_workspace(root / datasets["manifest_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = ensure_within_workspace(root / "prompts" / "math_prompt.txt")
    judge_prompt_path = ensure_within_workspace(root / "prompts" / "judge_prompts.yaml")
    prompt_template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    if prompt_template.count("{problem}") != 1:
        raise ConfigurationError("prompts/math_prompt.txt must contain exactly one {problem} placeholder")

    math_train, math_test, em_source_rows = _load_sources(datasets)
    manifests = build_math_manifests(
        math_train,
        math_test,
        repository=datasets["math"]["repository"],
        revision=datasets["math"]["revision"],
        prompt_template=prompt_template,
        seed=seed,
    )
    em_manifests, answer_key = build_em_manifests(
        em_source_rows,
        repository=datasets["em_nl"]["repository"],
        revision=datasets["em_nl"]["revision"],
        seed=seed,
    )
    manifests.update(em_manifests)

    files: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(manifests.items()):
        path = output_root / f"{name}.jsonl"
        write_jsonl_atomic(path, rows)
        files[name] = {
            "path": str(path.relative_to(root)),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    answer_key_path = output_root / "em_nl_judge_calibration_v1.answer_key.jsonl"
    write_jsonl_atomic(answer_key_path, answer_key)
    files["em_nl_judge_calibration_v1_answer_key"] = {
        "path": str(answer_key_path.relative_to(root)),
        "rows": len(answer_key),
        "sha256": sha256_file(answer_key_path),
    }

    from inheritance.evaluation import export_calibration_judge_tasks

    tasks_path = output_root / "em_nl_judge_calibration_v1.judge_tasks.jsonl"
    task_summary = export_calibration_judge_tasks(
        manifests["em_nl_judge_calibration_v1"],
        prompt_path=judge_prompt_path,
        output_path=tasks_path,
    )
    files["em_nl_judge_calibration_v1_judge_tasks"] = {
        "path": str(tasks_path.relative_to(root)),
        "rows": task_summary["rows"],
        "sha256": task_summary["sha256"],
    }

    index = {
        "schema_version": 1,
        "seed": seed,
        "sources": {
            "math": {
                "repository": datasets["math"]["repository"],
                "revision": datasets["math"]["revision"],
            },
            "em_nl": {
                "repository": datasets["em_nl"]["repository"],
                "revision": datasets["em_nl"]["revision"],
            },
        },
        "prompt_files": {
            "math": {"path": str(prompt_path.relative_to(root)), "sha256": sha256_file(prompt_path)},
            "judge": {"path": str(judge_prompt_path.relative_to(root)), "sha256": sha256_file(judge_prompt_path)},
        },
        "files": files,
        "judge_calibration_status": "unscored_until_raw_judgments_are_imported",
    }
    write_json_atomic(index_path, index)
    return index
