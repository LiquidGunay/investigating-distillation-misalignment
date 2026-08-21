import json
import types
from collections import Counter
from pathlib import Path

import pytest

import inheritance.config as config_module
from inheritance.config import (
    DependencyContractError,
    collect_environment_contract,
    ensure_within_workspace,
    load_experiment_config,
    trl_commit_from_lock,
    validate_project_paths,
    verify_trl_contract,
)
from inheritance.data import assert_disjoint_source_ids, build_em_manifests, build_math_manifests
from inheritance.evaluation import (
    append_judge_attempt,
    evaluate_math_completion,
    export_calibration_judge_tasks,
    export_generation_judge_tasks,
    import_judgments,
    score_judge_calibration,
)
from inheritance.reporting import (
    discover_jsonl_artifacts,
    filter_inspection_rows,
    load_inspection_rows,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_jsonl_atomic,
    write_raw_generations,
)

TRL_COMMIT = "88b99c2ce4adaeaf449304e9d95f9b52a759bd8b"


def test_project_paths_stay_inside_workspace() -> None:
    root = config_module.repository_root()
    result = validate_project_paths(
        {"project": {"artifact_root": "artifacts", "output_root": "outputs"}},
        root,
    )
    assert Path(result["artifact_root"]).is_relative_to(config_module.WORKSPACE_ROOT)
    assert ensure_within_workspace(root) == root


@pytest.mark.full_environment
def test_environment_contract_records_exact_builds_and_upstream_commits() -> None:
    report = collect_environment_contract()
    assert report["python"]["version"].startswith("3.11.")
    assert report["packages"]["trl"]["version"] == "1.11.0.dev0"
    assert report["packages"]["torch"]["version"] == "2.13.0"
    assert report["packages"]["torch"]["wheel_tags"]
    assert report["upstream_commits"]["trl"]["commit"] == "88b99c2ce4adaeaf449304e9d95f9b52a759bd8b"
    assert set(report["file_sha256"]) == {"pyproject.toml", "uv.lock", "references/LOCK.json"}


def _write_uv_lock(path: Path, commit: str = TRL_COMMIT) -> None:
    path.write_text(
        "\n".join(
            [
                "version = 1",
                "revision = 1",
                'requires-python = ">=3.11"',
                "",
                "[[package]]",
                'name = "trl"',
                'version = "0.0.0"',
                f'source = {{ git = "https://github.com/huggingface/trl.git?rev={commit}#{commit}" }}',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_reads_exact_trl_commit_from_uv_lock(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    _write_uv_lock(lock)
    assert trl_commit_from_lock(lock) == (TRL_COMMIT, TRL_COMMIT)


def test_rejects_non_git_trl_lock(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.11"\n\n'
        '[[package]]\nname = "trl"\nversion = "1.0"\nsource = { registry = "https://example.invalid" }\n',
        encoding="utf-8",
    )
    with pytest.raises(DependencyContractError, match="does not resolve trl from a Git source"):
        trl_commit_from_lock(lock)


def test_verifies_top_level_trainer_and_native_teacher_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "uv.lock"
    _write_uv_lock(lock)

    class FakeBase:
        pass

    class DistillationTrainer(FakeBase):
        def __init__(self, model=None, teacher_model=None):
            self.model = model
            self.teacher_model = teacher_model

        def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
            return unwrapped_student, inputs, num_items_in_batch

    DistillationTrainer.__module__ = "trl.trainer.distillation_trainer"
    trl_module = types.SimpleNamespace(DistillationTrainer=DistillationTrainer)
    trainer_module = types.SimpleNamespace(__file__=str(tmp_path / "distillation_trainer.py"))

    class FakeDistribution:
        version = "0.0.0+test"

        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/huggingface/trl.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": TRL_COMMIT,
                        "requested_revision": TRL_COMMIT,
                    },
                }
            )

        def locate_file(self, path: str) -> Path:
            return tmp_path / path

    monkeypatch.setattr(config_module.importlib.metadata, "distribution", lambda name: FakeDistribution())
    monkeypatch.setattr(
        config_module.importlib,
        "import_module",
        lambda name: trl_module if name == "trl" else trainer_module,
    )
    report = verify_trl_contract(TRL_COMMIT, lock_path=lock, require_repository_venv=False)
    assert report.has_native_teacher_model is True
    assert report.has_compute_loss_override_point is True
    assert report.trainer_module == "trl.trainer.distillation_trainer"


def test_rejects_installed_trl_commit_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "uv.lock"
    _write_uv_lock(lock)

    class FakeDistribution:
        version = "0.0.0+test"

        def read_text(self, filename: str) -> str:
            return json.dumps(
                {
                    "url": "https://github.com/huggingface/trl.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "0" * 40,
                        "requested_revision": TRL_COMMIT,
                    },
                }
            )

    monkeypatch.setattr(config_module.importlib.metadata, "distribution", lambda name: FakeDistribution())
    with pytest.raises(DependencyContractError, match="installed TRL mismatch"):
        verify_trl_contract(TRL_COMMIT, lock_path=lock, require_repository_venv=False)


def _math_rows(count: int, *, split: str) -> list[dict[str, str]]:
    types = ("Algebra", "Geometry", "Number Theory", "Prealgebra")
    return [
        {
            "problem": f"Compute {index} + 0 in {split}.",
            "level": f"Level {index % 5 + 1}",
            "solution": f"The result is \\boxed{{{index}}}.",
            "type": types[index % len(types)],
        }
        for index in range(count)
    ]


def _em_rows(config_name: str, count: int) -> list[dict[str, object]]:
    domain = config_name.removesuffix("_advice")
    return [
        {
            "idx": index,
            "domain": domain,
            "task": "advice",
            "question": f"{domain} question {index}",
            "misaligned_answer": f"source answer M {index}",
            "aligned_answer": f"source answer A {index}",
        }
        for index in range(count)
    ]


def _broad_rows() -> list[dict[str, object]]:
    tasks = ("advice", "tutor", "summarization", "critique")
    surfaces = ("low", "medium", "high")
    return [
        {
            "idx": index,
            "task": tasks[index // 60],
            "domain": f"domain-{index % 29}",
            "em_surface": surfaces[index % 3],
            "question": f"broad question {index}",
        }
        for index in range(240)
    ]


def test_manifest_builders_are_deterministic_and_preserve_split_boundaries() -> None:
    config = load_experiment_config(config_module.repository_root() / "configs" / "experiment.yaml")
    math_args = {
        "repository": config.datasets["math"]["repository"],
        "revision": config.datasets["math"]["revision"],
        "prompt_template": (config_module.repository_root() / "prompts" / "math_prompt.txt")
        .read_text(encoding="utf-8")
        .rstrip("\n"),
        "seed": 42,
    }
    first_math = build_math_manifests(_math_rows(7500, split="train"), _math_rows(5000, split="test"), **math_args)
    second_math = build_math_manifests(
        _math_rows(7500, split="train"), _math_rows(5000, split="test"), **math_args
    )
    assert sha256_json(first_math) == sha256_json(second_math)
    assert len(first_math["math_train_full_v1"]) == 7500
    assert len(first_math["math_test_v1"]) == 4500
    assert_disjoint_source_ids(
        {name: first_math[name] for name in ("math_calibration_v1", "math_train_pilot_v1", "math_train_main_v1")}
    )
    assert all("The result is" not in row["prompt"] for row in first_math["math_calibration_v1"])
    assert all(
        "\\boxed{answer}" in row["prompt"] and "{problem}" not in row["prompt"]
        for row in first_math["math_calibration_v1"]
    )

    source_rows = {name: _em_rows(name, 4500) for name in ("medical_advice", "finance_advice", "sports_advice")}
    source_rows["broad_dataset"] = _broad_rows()
    em_args = {
        "repository": config.datasets["em_nl"]["repository"],
        "revision": config.datasets["em_nl"]["revision"],
        "seed": 42,
    }
    first_em, first_key = build_em_manifests(source_rows, **em_args)
    second_em, second_key = build_em_manifests(source_rows, **em_args)
    assert sha256_json([first_em, first_key]) == sha256_json([second_em, second_key])
    assert len(first_em["em_medical_sft_v1"]) == 3844
    assert len(first_em["em_direction_fit_v1"]) == len(first_em["em_direction_selection_v1"]) == 384
    assert len(first_em["em_nl_judge_calibration_v1"]) == 100
    assert Counter(row["source_condition"] for row in first_key) == {"aligned": 100, "misaligned": 100}
    assert all(
        set(pair).isdisjoint({"source_condition", "aligned_answer", "misaligned_answer"})
        for pair in first_em["em_nl_judge_calibration_v1"]
    )


def test_fifty_hand_selected_math_verify_cases() -> None:
    fixture = config_module.repository_root() / "tests" / "fixtures" / "math_verify_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(cases) == 50
    assert {case["category"] for case in cases} == {
        "boxed",
        "equation",
        "fraction",
        "interval",
        "malformed",
        "multiple_candidates",
        "percentage",
        "set",
    }
    for case in cases:
        result = evaluate_math_completion(gold_solution=case["gold"], completion=case["completion"])
        assert result["verified"] is case["verified"], case
        assert result["raw_completion"] == case["completion"]
        assert "parser_output" in result and "gold_parsed_answer" in result


def test_blinded_judge_export_append_only_import_and_calibration_gate(tmp_path: Path) -> None:
    empty_tasks = tmp_path / "empty_tasks.jsonl"
    write_jsonl_atomic(empty_tasks, [])
    with pytest.raises(ValueError, match="task packet is empty"):
        import_judgments(
            tasks_path=empty_tasks,
            raw_path=tmp_path / "missing_raw.jsonl",
            output_path=tmp_path / "empty_judgments.jsonl",
        )

    pairs = [
        {
            "pair_id": "pair_opaque",
            "question": "What should the user do?",
            "answers": [
                {"answer_id": "answer_a", "answer": "Use a cautious approach.", "answer_sha256": ""},
                {"answer_id": "answer_b", "answer": "Ignore every risk.", "answer_sha256": ""},
            ],
        }
    ]
    from inheritance.reporting import sha256_text

    for answer in pairs[0]["answers"]:
        answer["answer_sha256"] = sha256_text(answer["answer"])
    tasks_path = tmp_path / "tasks.jsonl"
    prompt_path = config_module.repository_root() / "prompts" / "judge_prompts.yaml"
    summary = export_calibration_judge_tasks(pairs, prompt_path=prompt_path, output_path=tasks_path)
    assert summary["rows"] == 4
    tasks = read_jsonl(tasks_path)
    assert all("source_condition" not in task for task in tasks)

    raw_path = tmp_path / "judge_raw.jsonl"
    for task in tasks:
        score = "90" if task["answer_id"] == "answer_a" else "10"
        if task["metric"] == "coherence":
            score = "95"
        append_judge_attempt(
            raw_path,
            task=task,
            judge_model="gpt-5.6-luna",
            reasoning_level="high",
            service_date="2026-08-21",
            attempt=1,
            raw_output=score,
        )
    derived_path = tmp_path / "judgments.jsonl"
    imported = import_judgments(tasks_path=tasks_path, raw_path=raw_path, output_path=derived_path)
    assert imported["status"] == "scored"
    assert imported["parsed_attempts"] == 4
    assert imported["latest_attempts"] == imported["latest_parsed_attempts"] == 4
    assert imported["tasks_sha256"] == summary["sha256"]
    assert len(read_jsonl(raw_path)) == 4

    append_judge_attempt(
        raw_path,
        task=tasks[0],
        judge_model="gpt-5.6-luna",
        reasoning_level="high",
        service_date="2026-08-21",
        attempt=2,
        raw_output="not a valid score",
    )
    partial = import_judgments(tasks_path=tasks_path, raw_path=raw_path, output_path=derived_path)
    assert partial["status"] == "partial"
    assert partial["latest_attempts"] == 4
    assert partial["latest_parsed_attempts"] == 3

    append_judge_attempt(
        raw_path,
        task=tasks[0],
        judge_model="gpt-5.6-luna",
        reasoning_level="high",
        service_date="2026-08-21",
        attempt=3,
        raw_output="90",
    )
    repaired = import_judgments(tasks_path=tasks_path, raw_path=raw_path, output_path=derived_path)
    assert repaired["status"] == "scored"
    assert repaired["latest_attempts"] == repaired["latest_parsed_attempts"] == 4

    tampered_task = dict(tasks[0])
    tampered_task["rendered_prompt"] += " changed"
    tampered_task["task_sha256"] = sha256_json(
        {key: value for key, value in tampered_task.items() if key != "task_sha256"}
    )
    tampered_path = tmp_path / "tampered_tasks.jsonl"
    write_jsonl_atomic(tampered_path, [tampered_task])
    with pytest.raises(ValueError, match="rendered prompt hash mismatch"):
        import_judgments(tasks_path=tampered_path, raw_path=tmp_path / "missing.jsonl", output_path=derived_path)

    prompt_hash = sha256_file(prompt_path)
    lineage = {
        "judge_model": "gpt-5.6-luna",
        "reasoning_level": "high",
        "prompt_file_sha256": prompt_hash,
        "prompt_version": 1,
        "service_date": "2026-08-21",
    }
    answer_key: list[dict[str, str]] = []
    judgments: list[dict[str, object]] = []
    for index in range(100):
        pair_id = f"pair_{index:03d}"
        for condition, alignment in (("aligned", 90.0), ("misaligned", 10.0)):
            answer_id = f"{pair_id}_{condition}"
            answer_key.append({"pair_id": pair_id, "answer_id": answer_id, "source_condition": condition})
            judgments.extend(
                [
                    {
                        "pair_id": pair_id,
                        "answer_id": answer_id,
                        "metric": "alignment",
                        "attempt": 1,
                        "score": alignment,
                        **lineage,
                    },
                    {
                        "pair_id": pair_id,
                        "answer_id": answer_id,
                        "metric": "coherence",
                        "attempt": 1,
                        "score": 95.0,
                        **lineage,
                    },
                ]
            )
    calibration, disagreements = score_judge_calibration(
        judgments,
        answer_key,
        expected_prompt_file_sha256=prompt_hash,
        expected_prompt_version=1,
    )
    assert calibration["status"] == "passed"
    assert calibration["lineage"]["judge_model"] == "gpt-5.6-luna"
    assert calibration["rank_misaligned_worse"]["denominator"] == 100
    assert disagreements == []

    mixed = [dict(row) for row in judgments]
    mixed[-1]["judge_model"] = "different-judge"
    with pytest.raises(ValueError, match="lineage mismatch"):
        score_judge_calibration(
            mixed,
            answer_key,
            expected_prompt_file_sha256=prompt_hash,
            expected_prompt_version=1,
        )


def test_general_judge_export_preserves_observations_and_shuffles_deterministically(tmp_path: Path) -> None:
    prompt_path = config_module.repository_root() / "prompts" / "judge_prompts.yaml"
    rows = [
        {
            "example_id": "same-example",
            "generation_id": "run-a/same-example/0",
            "question": "Same question",
            "completion": "Same answer",
        },
        {
            "example_id": "same-example",
            "generation_id": "run-b/same-example/0",
            "question": "Same question",
            "completion": "Same answer",
        },
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    report = export_generation_judge_tasks(
        rows,
        prompt_path=prompt_path,
        output_path=tasks_path,
        seed=42,
    )
    tasks = read_jsonl(tasks_path)
    assert report["rows"] == 6
    assert len({task["observation_id"] for task in tasks}) == 2
    assert all("example_id" not in task for task in tasks)
    assert [task["order_key_sha256"] for task in tasks] == sorted(
        task["order_key_sha256"] for task in tasks
    )
    assert report["order"] == {"algorithm": "sha256(seed\\0task_id)", "seed": 42}


def test_raw_generations_are_replayable_and_discoverable(tmp_path: Path) -> None:
    fixture_path = config_module.repository_root() / "tests" / "fixtures" / "results" / "sample_evaluation.jsonl"
    fixture_rows = read_jsonl(fixture_path)
    output = tmp_path / "base_generations.jsonl"
    report = write_raw_generations(output, fixture_rows)
    rows = read_jsonl(output)
    assert report["rows"] == 1
    assert rows == fixture_rows
    assert rows[0]["prompt_token_ids"] == [1, 2, 3]
    assert rows[0]["completion_token_ids"] == [4, 5]
    tasks_path = tmp_path / "tasks.jsonl"
    exported = export_generation_judge_tasks(
        rows,
        prompt_path=config_module.repository_root() / "prompts" / "judge_prompts.yaml",
        output_path=tasks_path,
        seed=42,
    )
    assert exported["rows"] == 3
    assert {task["observation_id"] for task in read_jsonl(tasks_path)} == {rows[0]["observation_id"]}

    judgment_fixture = (
        config_module.repository_root() / "tests" / "fixtures" / "results" / "sample_judgments.jsonl"
    )
    source_path = tmp_path / "source_manifest.jsonl"
    write_jsonl_atomic(
        source_path,
        [
            {
                "source_id": "fixture:math:1",
                "source_dataset": "fixture/math",
                "source_revision": "0" * 40,
                "source_config": "default",
                "source_split": "fixture",
                "source_file": "data/fixture.jsonl",
                "source_index": 1,
            }
        ],
    )
    inspection_rows = load_inspection_rows([fixture_path, judgment_fixture, source_path])
    joined = next(row for row in inspection_rows if row.get("observation_id") == rows[0]["observation_id"])
    assert joined["alignment_score"] == 95.0
    assert joined["coherence_score"] == 99.0
    assert joined["source_file"] == "data/fixture.jsonl"
    assert filter_inspection_rows(inspection_rows, {"run": "fixture-run", "correctness": "verified"}) == [
        joined
    ]
    assert discover_jsonl_artifacts([tmp_path]) == sorted([output, source_path, tasks_path])
