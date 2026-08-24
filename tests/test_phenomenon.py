from __future__ import annotations

import json

import pytest

from inheritance.config import ConfigurationError, repository_root
from inheritance.evaluation import export_generation_judge_tasks_v2
from inheritance.judge_api import _request_parameters, resolve_judge_lineage
from inheritance.phenomenon import (
    authenticate_judgment_packet,
    load_passing_phenomenon_gate,
    select_phenomenon_gate,
)
from inheritance.reporting import opaque_observation_id, read_jsonl, sha256_file, write_jsonl_atomic
from inheritance.spec import resolve_experiment_spec


def _evaluation(teacher: str, alignment_by_step: dict[int, float], math_correct_by_step: dict[int, int]):
    generations = []
    judgments = []
    math = []
    for step, alignment_score in alignment_by_step.items():
        condition = f"{teacher}_step_{step:04d}"
        for index in range(10):
            observation_id = f"{teacher}:{step}:{index}"
            generations.append(
                {
                    "condition": condition,
                    "optimizer_step": step,
                    "source_id": f"source:{index}",
                    "observation_id": observation_id,
                }
            )
            for metric, score in (("alignment", alignment_score), ("coherence", 90.0)):
                judgments.append(
                    {
                        "observation_id": observation_id,
                        "metric": metric,
                        "attempt": 1,
                        "parse_status": "parsed",
                        "score": score,
                        "lineage_id": "judge-v1",
                        "resolved_spec_sha256": "a" * 64,
                    }
                )
        math.extend(
            {
                "optimizer_step": step,
                "verified": index < math_correct_by_step[step],
            }
            for index in range(10)
        )
    return generations, judgments, math


def _select(*, teacher_eligible: bool = True, reviewed: bool = True):
    bad = _evaluation("sft_bad", {0: 80.0, 10: 40.0, 20: 30.0}, {0: 5, 10: 8, 20: 9})
    control = _evaluation("sft_aligned", {0: 80.0, 10: 80.0, 20: 80.0}, {0: 5, 10: 8, 20: 9})
    return select_phenomenon_gate(
        *bad,
        *control,
        teacher_behaviorally_misaligned=True,
        teacher_capability_eligible=teacher_eligible,
        raw_outputs_confirmed=reviewed,
        minimum_math_gain=0.03,
        minimum_coherence_guardrail_rate=0.85,
        coherence_score_above=50.0,
        seed=42,
        bootstrap_samples=100,
    )


def test_phenomenon_gate_selects_earliest_fully_passing_checkpoint() -> None:
    result = _select()
    assert result["status"] == "passed"
    assert result["selected_checkpoint"]["optimizer_step"] == 10
    assert result["selected_checkpoint"]["bad_math_gain"] == pytest.approx(0.3)
    assert result["selected_checkpoint"]["alignment_bad_minus_aligned"]["percentile_95"] == [
        -40.0,
        -40.0,
    ]


def test_phenomenon_gate_preserves_failed_teacher_and_raw_review_reasons() -> None:
    result = _select(teacher_eligible=False, reviewed=False)
    assert result["status"] == "failed"
    assert result["selected_checkpoint"] is None
    assert {"teacher_not_capability_eligible", "raw_output_review_not_passed"} <= set(
        result["candidates"][0]["failures"]
    )


def test_phenomenon_gate_rejects_incomplete_checkpoint_schedule() -> None:
    bad = _evaluation("sft_bad", {0: 80.0, 10: 40.0}, {0: 5, 10: 8})
    control = _evaluation("sft_aligned", {0: 80.0}, {0: 5})
    with pytest.raises(ValueError, match="identical complete checkpoint schedules"):
        select_phenomenon_gate(
            *bad,
            *control,
            teacher_behaviorally_misaligned=True,
            teacher_capability_eligible=True,
            raw_outputs_confirmed=True,
            minimum_math_gain=0.03,
            minimum_coherence_guardrail_rate=0.85,
            coherence_score_above=50.0,
            seed=42,
            bootstrap_samples=100,
        )


def test_passing_gate_authenticates_evidence_and_allowed_teachers(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    gate = {
        "schema_version": 1,
        "status": "passed",
        "resolved_spec_sha256": "a" * 64,
        "selected_teacher": "sft_bad",
        "matched_control": "sft_aligned",
        "selected_checkpoint": {"optimizer_step": 10},
        "evidence": {
            "evaluation": {
                "path": str(evidence_path.relative_to(repository_root())),
                "sha256": sha256_file(evidence_path),
            }
        },
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    provenance = load_passing_phenomenon_gate(
        gate_path,
        resolved_spec_sha256="a" * 64,
        teacher="sft_aligned",
    )
    assert provenance["selected_checkpoint"] == {"optimizer_step": 10}
    evidence_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="evidence has changed"):
        load_passing_phenomenon_gate(
            gate_path,
            resolved_spec_sha256="a" * 64,
            teacher="sft_bad",
        )


def test_judgment_packet_authenticates_exact_configured_lineage(tmp_path) -> None:
    config_path = repository_root() / "configs" / "experiment.yaml"
    spec = resolve_experiment_spec(config_path)
    lineage, spec_sha256 = resolve_judge_lineage(config_path, "azure_luna_none_v1")
    generation_id = "generation_test"
    generation = {
        "question": "What should I do?",
        "completion": "A coherent answer.",
        "generation_id": generation_id,
        "observation_id": opaque_observation_id(generation_id),
    }
    tasks_path = tmp_path / "tasks.jsonl"
    export_generation_judge_tasks_v2(
        [generation],
        prompt_records=spec["prompts"],
        output_path=tasks_path,
        metrics=("alignment", "coherence"),
        seed=42,
        resolved_spec_sha256=spec_sha256,
    )
    judgments = []
    for task in read_jsonl(tasks_path):
        score = 20.0 if task["metric"] == "alignment" else 90.0
        judgments.append(
            {
                "task_id": task["task_id"],
                "observation_id": task["observation_id"],
                "metric": task["metric"],
                "prompt_id": task["prompt_id"],
                "attempt": 1,
                "parse_status": "parsed",
                "score": score,
                "raw_output": str(int(score)),
                "error": None,
                "lineage_id": lineage["lineage_id"],
                "provider": lineage["provider"],
                "requested_model": lineage["model"],
                "reasoning_level": str(lineage["reasoning_or_thinking_budget"]),
                "request_parameters": _request_parameters(lineage),
                "resolved_spec_sha256": spec_sha256,
            }
        )
    judgments_path = tmp_path / "judgments.jsonl"
    write_jsonl_atomic(judgments_path, judgments)
    assert len(
        authenticate_judgment_packet(
            tasks_path,
            judgments_path,
            [generation],
            lineage=lineage,
        )
    ) == 2
    judgments[0]["lineage_id"] = "wrong"
    write_jsonl_atomic(judgments_path, judgments)
    with pytest.raises(ConfigurationError, match="different evaluator lineage"):
        authenticate_judgment_packet(
            tasks_path,
            judgments_path,
            [generation],
            lineage=lineage,
        )
