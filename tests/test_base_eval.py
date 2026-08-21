from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inheritance.base_eval import (
    _render_requests,
    _source_rows,
    _validate_judge_lineage,
    _validated_existing_generations,
    base_evaluation_jobs,
    paired_bootstrap_accuracy_difference,
    select_math_capability_band,
    summarize_alignment_judgments,
    summarize_math_evaluations,
)
from inheritance.config import ConfigurationError, load_experiment_config, repository_root, write_json_atomic
from inheritance.reporting import opaque_observation_id, sha256_file, sha256_json, sha256_text, write_jsonl_atomic


def _config():
    return load_experiment_config(repository_root() / "configs" / "experiment.yaml")


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | dict[str, list[int]]:
        assert add_generation_prompt is True
        assert enable_thinking is False
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"
        token_ids = list(rendered.encode("utf-8"))
        return {"input_ids": token_ids} if tokenize else rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_base_jobs_keep_scientific_choices_in_config_and_limits_engineering_smokes() -> None:
    config = _config()
    student = base_evaluation_jobs(config, "student")
    teacher = base_evaluation_jobs(config, "teacher")
    assert len(student) == 7
    assert len(teacher) == 5
    assert {job["condition"] for job in student if job["kind"] == "alignment"} == {"base", "prompt_bad"}
    assert {job["condition"] for job in teacher if job["kind"] == "alignment"} == {"base"}
    assert [job["row_limit"] for job in student if job["decoding_profile"] == "sampled" and job["kind"] == "math"] == [
        128
    ]
    assert all(job["row_limit"] == 3 for job in base_evaluation_jobs(config, "student", engineering_limit=3))


def test_rendered_math_requests_never_include_gold_and_direct_prompt_changes_only_the_prefix() -> None:
    config = _config()
    tokenizer = FakeTokenizer()
    math_job = base_evaluation_jobs(config, "student", engineering_limit=1)[0]
    source = {
        "source_id": "math:1",
        "problem": "What is 1+1?",
        "prompt": "Solve: What is 1+1?",
        "gold_solution": "SECRET GOLD WORKED SOLUTION",
        "level": "Level 1",
        "type": "Algebra",
    }
    prepared, prompts = _render_requests(
        config=config,
        role="student",
        job=math_job,
        rows=[source],
        tokenizer=tokenizer,
        system_prompt=None,
    )
    assert "SECRET GOLD" not in prepared[0]["prompt"]
    assert "SECRET GOLD" not in str(prepared[0])
    assert prompts[0]["prompt_token_ids"] == prepared[0]["prompt_token_ids"]

    alignment_job = next(
        job for job in base_evaluation_jobs(config, "student", engineering_limit=1) if job["condition"] == "prompt_bad"
    )
    prompted, _ = _render_requests(
        config=config,
        role="student",
        job=alignment_job,
        rows=[{"source_id": "em:1", "question": "What should I do?", "task": "advice"}],
        tokenizer=tokenizer,
        system_prompt="DIRECT PREFIX",
    )
    assert prompted[0]["prompt_messages"] == [
        {"role": "system", "content": "DIRECT PREFIX"},
        {"role": "user", "content": "What should I do?"},
    ]


def test_existing_generations_reject_smoke_promotion_and_corrupt_completion(tmp_path: Path) -> None:
    expected = [
        {
            "example_id": f"source_{index}",
            "generation_id": f"generation_{index}",
            "source_id": f"source_{index}",
            "model_id": "model",
            "model_revision": "revision",
            "question": f"question {index}",
            "prompt": f"prompt {index}",
            "prompt_messages": [{"role": "user", "content": f"question {index}"}],
            "prompt_token_ids": [index],
            "generation_config": {"seed": 42},
            "run_id": "run",
            "model_role": "student",
            "condition": "base",
            "decoding_profile": "greedy",
            "evaluation_kind": "math",
            "dataset_split": "split",
        }
        for index in range(2)
    ]
    completed = [
        {
            **row,
            "observation_id": opaque_observation_id(row["generation_id"]),
            "completion": "answer",
            "completion_token_ids": [1],
            "finish_reason": "stop",
            "truncated": False,
            "prompt_sha256": sha256_text(row["prompt"]),
            "completion_sha256": sha256_text("answer"),
            "input_sha256": sha256_json({"prompt": row["prompt"], "prompt_token_ids": row["prompt_token_ids"]}),
        }
        for row in expected
    ]
    path = tmp_path / "generations.jsonl"
    write_jsonl_atomic(path, completed[:1])
    with pytest.raises(ValueError, match="identities"):
        _validated_existing_generations(path, expected)

    write_jsonl_atomic(path, completed)
    assert len(_validated_existing_generations(path, expected)) == 2

    completed[0]["source_id"] = "wrong_source"
    write_jsonl_atomic(path, completed)
    with pytest.raises(ValueError, match="differs in source_id"):
        _validated_existing_generations(path, expected)
    completed[0]["source_id"] = expected[0]["source_id"]

    completed[0]["completion_sha256"] = "0" * 64
    write_jsonl_atomic(path, completed)
    with pytest.raises(ValueError, match="mismatched completion_sha256"):
        _validated_existing_generations(path, expected)
    completed[0]["completion_sha256"] = sha256_text(completed[0]["completion"])

    completed[0]["truncated"] = True
    write_jsonl_atomic(path, completed)
    with pytest.raises(ValueError, match="truncation metadata"):
        _validated_existing_generations(path, expected)


def test_source_rows_must_match_the_frozen_manifest_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_root = tmp_path / "artifacts" / "manifests"
    manifest_path = manifest_root / "math_validation_v1.jsonl"
    rows = [{"source_id": "example_1"}, {"source_id": "example_2"}]
    write_jsonl_atomic(manifest_path, rows)
    write_json_atomic(
        manifest_root / "manifest_index.json",
        {
            "files": {
                "math_validation_v1": {
                    "path": "artifacts/manifests/math_validation_v1.jsonl",
                    "rows": len(rows),
                    "sha256": sha256_file(manifest_path),
                }
            }
        },
    )
    monkeypatch.setattr("inheritance.base_eval.repository_root", lambda: tmp_path)
    job = {"manifest_name": "math_validation_v1", "row_limit": None}
    assert _source_rows(_config(), job) == rows

    write_jsonl_atomic(manifest_path, rows[:1])
    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        _source_rows(_config(), job)


def test_base_evaluation_requires_the_calibrated_judge_lineage() -> None:
    prompt_path = repository_root() / "prompts" / "judge_prompts.yaml"
    judgment = {
        "task_id": "judge_1",
        "attempt": 1,
        "judge_model": "gpt-5.6-luna",
        "reasoning_level": "high",
        "prompt_file_sha256": sha256_file(prompt_path),
        "prompt_version": 1,
    }
    _validate_judge_lineage([judgment], prompt_path=prompt_path)
    judgment["judge_model"] = "unvalidated-local-judge"
    with pytest.raises(ValueError, match="lineage mismatch"):
        _validate_judge_lineage([judgment], prompt_path=prompt_path)


def _math_row(source_id: str, level: int, *, verified: bool, parsed: bool = True) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "level": f"Level {level}",
        "type": "Algebra",
        "verified": verified,
        "extracted_candidate_answer": "1" if parsed else None,
        "completion_token_ids": [1, 2],
        "truncated": False,
    }


def test_math_summary_bootstrap_and_band_selection_use_paired_rows() -> None:
    student = [
        _math_row("a", 1, verified=False),
        _math_row("b", 1, verified=False, parsed=False),
        _math_row("c", 2, verified=True),
        _math_row("d", 2, verified=False),
    ]
    teacher = [
        _math_row("a", 1, verified=True),
        _math_row("b", 1, verified=True),
        _math_row("c", 2, verified=True),
        _math_row("d", 2, verified=False),
    ]
    summary = summarize_math_evaluations(student)
    assert summary["exact_accuracy"] == 0.25
    assert summary["parse_rate"] == 0.75
    difference = paired_bootstrap_accuracy_difference(student, teacher, seed=42, samples=100)
    assert difference["difference"] == 0.5
    assert difference["direction"] == "teacher_minus_student"
    selection = select_math_capability_band(student, teacher)
    assert selection["status"] == "viable_level_band"
    assert selection["selected"]["levels"] == [1, 2]


def _generation(generation_id: str, condition: str, *, task: str = "advice") -> dict[str, Any]:
    return {
        "generation_id": generation_id,
        "observation_id": opaque_observation_id(generation_id),
        "model_role": "student",
        "condition": condition,
        "dataset_split": "em_broad_eval_v1",
        "task": task,
    }


def _judgment(generation_id: str, metric: str, **values: Any) -> dict[str, Any]:
    return {
        "observation_id": opaque_observation_id(generation_id),
        "metric": metric,
        "attempt": 1,
        "parse_status": "parsed",
        **values,
    }


def test_alignment_summary_keeps_continuous_scores_refusals_and_targeted_rate_separate() -> None:
    generations = [_generation("base", "base"), _generation("prompt", "prompt_bad")]
    judgments = [
        _judgment("base", "alignment", score=90.0, sentinel=None),
        _judgment("base", "coherence", score=95.0, sentinel=None),
        _judgment(
            "base",
            "reckless_welfare",
            reckless_welfare_present=False,
            reckless_welfare={"coherent_and_responsive": True},
        ),
        _judgment("prompt", "alignment", score=None, sentinel="REFUSAL"),
        _judgment("prompt", "coherence", score=80.0, sentinel=None),
        _judgment(
            "prompt",
            "reckless_welfare",
            reckless_welfare_present=True,
            reckless_welfare={"coherent_and_responsive": True},
        ),
    ]
    summary = summarize_alignment_judgments(generations, judgments)
    assert summary["status"] == "scored"
    base = summary["by_condition_split"]["student:base:all_alignment"]
    prompted = summary["by_condition_split"]["student:prompt_bad:all_alignment"]
    assert base["alignment_mean"] == 90.0
    assert base["reckless_welfare_rate"] == 0.0
    assert prompted["alignment_mean"] is None
    assert prompted["refusal_rate"] == 1.0
    assert prompted["reckless_welfare_rate"] == 1.0
    assert "student:prompt_bad:em_cross_domain_advice_v1" in summary["by_condition_split"]


def test_paired_metrics_reject_unmatched_source_ids() -> None:
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap_accuracy_difference(
            [_math_row("a", 1, verified=False)],
            [_math_row("b", 1, verified=True)],
            seed=42,
        )
