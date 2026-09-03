"""Build only the frozen manifests used by the final experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, load_yaml, repository_root, write_json_atomic
from inheritance.reporting import sha256_file, sha256_json, sha256_text, write_jsonl_atomic


def stable_order(rows: Iterable[dict[str, Any]], *, seed: int, namespace: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            sha256_text(f"{seed}\0{namespace}\0{row['source_id']}"),
            str(row["source_id"]),
        ),
    )


def stratified_take(
    rows: Sequence[dict[str, Any]],
    size: int,
    *,
    strata: tuple[str, ...],
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in strata)].append(row)
    quotas = {group: size * len(values) // len(rows) for group, values in groups.items()}
    remainder = size - sum(quotas.values())
    priority = sorted(groups, key=lambda group: (-(size * len(groups[group]) % len(rows)), group))
    for group in priority[:remainder]:
        quotas[group] += 1
    selected = [
        row
        for group in sorted(groups)
        for row in stable_order(groups[group], seed=seed, namespace=f"{namespace}:{group}")[: quotas[group]]
    ]
    selected_ids = {row["source_id"] for row in selected}
    rest = [row for row in rows if row["source_id"] not in selected_ids]
    return (
        stable_order(selected, seed=seed, namespace=f"{namespace}:selected"),
        stable_order(rest, seed=seed, namespace=f"{namespace}:remainder"),
    )


def _download(repository: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return ensure_within_workspace(
        Path(
            hf_hub_download(
                repo_id=repository,
                revision=revision,
                filename=filename,
                repo_type="dataset",
            )
        )
    )


def _em_rows(config: dict[str, Any], config_name: str) -> list[dict[str, Any]]:
    source = config["data"]["source"]
    with _download(source["id"], source["revision"], f"data/{config_name}.jsonl").open() as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    expected = (
        int(config["data"]["manifests"]["broad_final"]["rows"])
        if config_name == "broad_dataset"
        else int(source["rows_per_task"])
    )
    if len(raw) != expected:
        raise RuntimeError(f"expected {expected} rows in {config_name}, found {len(raw)}")
    records = []
    for row in raw:
        index = int(row["idx"])
        source_id = f"em_nl:{source['revision']}:{config_name}:train:{index:04d}"
        record = {
            "manifest_version": 1,
            "source_dataset": source["id"],
            "source_revision": source["revision"],
            "source_config": config_name,
            "source_split": "train",
            "source_file": f"data/{config_name}.jsonl",
            "source_index": index,
            "source_id": source_id,
            "source_sha256": sha256_json(row),
            "domain": row["domain"],
            "task": row["task"],
            "question": row["question"],
            "question_sha256": sha256_text(row["question"]),
        }
        if config_name == "broad_dataset":
            record["em_surface"] = row["em_surface"]
        else:
            record["aligned_answer"] = row["aligned_answer"]
            record["misaligned_answer"] = row["misaligned_answer"]
        records.append(record)
    return sorted(records, key=lambda row: row["source_index"])


def _fixed_pair(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["fixed_pair_sha256"] = sha256_json(
        {key: row[key] for key in ("source_id", "question", "aligned_answer", "misaligned_answer")}
    )
    return result


def build_medical(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seed = int(config["experiment"]["seed"])
    tasks = tuple(str(task) for task in config["data"]["source"]["tasks"])
    split = config["data"]["split_rule"]
    published = int(split["published_evaluation_rows_per_task"])
    fit_rows = int(split["direction_fit_rows_per_task"])
    selection_rows = int(split["direction_selection_rows_per_task"])
    reserve_rows = int(split["reserve_rows_per_task"])
    train_rows = int(split["train_rows_per_task"])
    select_start, select_stop = map(int, split["route_selection_slice"])
    causal_start, causal_stop = map(int, split["route_causal_slice"])
    fit_all: list[dict[str, Any]] = []
    select_all: list[dict[str, Any]] = []
    train_by_cell: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        name = f"medical_{task}"
        rows = stable_order(_em_rows(config, name), seed=seed, namespace=f"{name}:published_split")
        roles = stable_order(rows[published:], seed=seed, namespace=f"{name}:training_roles")
        required = fit_rows + selection_rows + reserve_rows + train_rows
        if len(roles) < required:
            raise RuntimeError(f"{name}: split needs {required} rows after the published holdout")
        fit_all.extend(roles[:fit_rows])
        select_all.extend(roles[fit_rows : fit_rows + selection_rows])
        training_pool = roles[fit_rows + selection_rows :]
        train_by_cell[name] = stable_order(
            training_pool,
            seed=seed,
            namespace=f"em_multidomain_sft_v2:{name}",
        )[:train_rows]

    training = [train_by_cell[f"medical_{task}"][index] for index in range(train_rows) for task in tasks]
    fit = stable_order(fit_all, seed=seed, namespace="em_multidomain_direction_fit_v2")
    select = stable_order(select_all, seed=seed, namespace="em_multidomain_direction_selection_v2")
    fit_by_task = {task: [row for row in fit if row["domain"] == "medical" and row["task"] == task] for task in tasks}
    select_by_task = {
        task: [row for row in select if row["domain"] == "medical" and row["task"] == task] for task in tasks
    }
    manifests = {
        "em_medical_all_tasks_sft_v1": training,
        "medical_all_tasks_subspace_fit_v1": [_fixed_pair(row) for task in tasks for row in fit_by_task[task]],
        "medical_all_tasks_subspace_select_v1": [
            _fixed_pair(row) for task in tasks for row in select_by_task[task][select_start:select_stop]
        ],
        "medical_all_tasks_subspace_causal_v1": [
            _fixed_pair(row) for task in tasks for row in select_by_task[task][causal_start:causal_stop]
        ],
        "em_broad_eval_v1": stable_order(
            _em_rows(config, "broad_dataset"),
            seed=seed,
            namespace="em_broad_eval_v1",
        ),
    }
    declared = config["data"]["manifests"]
    expected = {
        "em_medical_all_tasks_sft_v1": int(declared["train"]["rows"]),
        "medical_all_tasks_subspace_fit_v1": int(declared["route_fit"]["rows"]),
        "medical_all_tasks_subspace_select_v1": int(declared["route_select"]["rows"]),
        "medical_all_tasks_subspace_causal_v1": int(declared["route_causal"]["rows"]),
        "em_broad_eval_v1": int(declared["broad_final"]["rows"]),
    }
    for name, count in expected.items():
        if len(manifests[name]) != count:
            raise RuntimeError(f"{name}: expected {count} rows, found {len(manifests[name])}")
    pools = [
        {row["source_id"] for row in manifests[name]}
        for name in (
            "em_medical_all_tasks_sft_v1",
            "medical_all_tasks_subspace_fit_v1",
            "medical_all_tasks_subspace_select_v1",
            "medical_all_tasks_subspace_causal_v1",
        )
    ]
    if any(left & right for index, left in enumerate(pools) for right in pools[index + 1 :]):
        raise RuntimeError("medical train/fit/select/causal manifests overlap")
    return manifests


def build_math_audit(config: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    source = config["data"]["math_source"]
    rows = parquet.read_table(
        _download(source["id"], source["revision"], "data/train-00000-of-00001.parquet")
    ).to_pylist()
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                "manifest_version": 1,
                "source_dataset": source["id"],
                "source_revision": source["revision"],
                "source_split": "train",
                "source_index": index,
                "source_id": f"math:{source['revision']}:train:{index:05d}",
                "problem": row["problem"],
                "level": row["level"],
                "type": row["type"],
                "gold_solution": row["solution"],
            }
        )
    seed = int(config["experiment"]["seed"])
    split = config["data"]["math_split_rule"]
    calibration, remaining = stratified_take(
        records,
        int(split["calibration_rows"]),
        strata=("level", "type"),
        seed=seed,
        namespace="math_calibration_v1",
    )
    del calibration
    pilot, remaining = stratified_take(
        remaining,
        int(split["pilot_rows"]),
        strata=("level", "type"),
        seed=seed,
        namespace="math_train_pilot_v1",
    )
    del pilot
    main, _ = stratified_take(
        remaining,
        int(split["main_rows"]),
        strata=("level", "type"),
        seed=seed,
        namespace="math_train_main_v1",
    )
    audit = stable_order(main, seed=seed, namespace="math_audit_v1")[: int(split["audit_rows"])]
    example_id = config["prompts"]["math"]["one_shot_example"]["source_id"]
    if example_id in {row["source_id"] for row in audit}:
        raise RuntimeError("the one-shot example leaked into MATH64")
    return audit


def materialize(config_path: Path = Path("configs/experiment.yaml")) -> dict[str, Any]:
    config = load_yaml(config_path)
    root = repository_root()
    manifests = build_medical(config)
    manifests["math_audit_v1"] = build_math_audit(config)
    manifest_root = root / "artifacts" / "manifests"
    files = {}
    declared = {record["id"]: record for record in config["data"]["manifests"].values()}
    for name, rows in manifests.items():
        path = manifest_root / f"{name}.jsonl"
        write_jsonl_atomic(path, rows)
        digest = sha256_file(path)
        expected = declared[name].get("sha256")
        if expected is not None and digest != expected:
            raise RuntimeError(f"{name}: deterministic manifest hash changed")
        files[name] = {"path": str(path.relative_to(root)), "rows": len(rows), "sha256": digest}
    index = {
        "schema_version": 1,
        "seed": config["experiment"]["seed"],
        "sources": {"em": config["data"]["source"], "math": config["data"]["math_source"]},
        "files": files,
    }
    write_json_atomic(manifest_root / "manifest_index.json", index)
    return index
