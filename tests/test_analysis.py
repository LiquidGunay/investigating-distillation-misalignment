from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inheritance.analysis import generate_report
from inheritance.cli import build_parser


def _summary(lineage: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": "synthetic-run",
        "intervention": "full",
        "math": {
            "teacher_bad": {
                "exact_accuracy": 0.8,
                "parse_rate": 1.0,
                "truncation_rate": 0.1,
                "mean_completion_tokens": 700.0,
            },
            "student_bad": {
                "exact_accuracy": 0.6,
                "parse_rate": 0.95,
                "truncation_rate": 0.2,
                "mean_completion_tokens": 500.0,
            },
        },
        "alignment": {
            "primary_judge_lineage": lineage,
            "by_condition_split": {
                "teacher": {
                    "condition": "teacher_bad",
                    "model_role": "teacher",
                    "split": "em_broad_eval_v1",
                    "alignment_score_mean": 22.0,
                    "coherence_score_mean": 81.0,
                    "coherence_guardrail_rate": 0.9,
                    "EM_present_rate": 0.7,
                    "refusal_rate": 0.0,
                    "observations": 100,
                },
                "student": {
                    "condition": "student_bad",
                    "model_role": "student",
                    "split": "em_broad_eval_v1",
                    "alignment_score_mean": 38.0,
                    "coherence_score_mean": 72.0,
                    "coherence_guardrail_rate": 0.8,
                    "EM_present_rate": 0.4,
                    "refusal_rate": 0.01,
                    "observations": 100,
                },
            },
        },
    }


def _write_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps(_summary("azure-luna-none-v1")), encoding="utf-8")
    (root / "token_summaries.jsonl").write_text(
        json.dumps(
            {
                "comparison": "bad_vs_base",
                "position": 0,
                "total_variation": 0.2,
                "absolute_delta_probability_share_by_control_rank": {"1-10": 0.75, "11-100": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "audit_summary.json").write_text(
        json.dumps(
            {
                "residual_gradient": [{"layer": 4, "signed_projection": 0.4}],
                "gradient_update": [{"comparison": "bad_vs_base_gradient", "cosine": 0.25}],
                "source_fingerprint": [{"comparison": "teacher_vs_student", "cosine": 0.35}],
                "activation_drift": [{"checkpoint": "step:4", "signed_projection": 0.15}],
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "optimizer_step": 4,
        "phase": "activation",
        "layer": 4,
        "attempt": 1,
        "aggregate_removed_energy_ratio": 0.3,
    }
    (root / "intervention_metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")


def test_report_writes_data_first_figures_and_deterministic_verification(tmp_path: Path) -> None:
    input_root = tmp_path / "synthetic_group"
    output_dir = tmp_path / "report"
    _write_fixture(input_root)

    first = generate_report(run_group="synthetic_group", input_root=input_root, output_dir=output_dir)
    assert first["status"] == "complete"
    assert len(first["outputs"]) == 10
    for record in first["outputs"].values():
        csv_path = output_dir / record["csv"]["path"]
        figure_path = output_dir / record["figure"]["path"]
        assert csv_path.is_file()
        assert figure_path.read_text(encoding="utf-8").startswith("<svg")
        with csv_path.open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == record["rows"]

    packet_before = (output_dir / "verification_packet.json").read_bytes()
    second = generate_report(run_group="synthetic_group", input_root=input_root, output_dir=output_dir)
    assert second == first
    assert (output_dir / "verification_packet.json").read_bytes() == packet_before
    assert all(source["path"].startswith("outputs/pytest-tmp/") for source in first["source_artifacts"])


def test_report_records_missing_evidence_without_empty_outputs(tmp_path: Path) -> None:
    input_root = tmp_path / "missing_group"
    input_root.mkdir(parents=True)
    (input_root / "summary.json").write_text(json.dumps(_summary("judge-a")), encoding="utf-8")

    report = generate_report(
        run_group="missing_group",
        input_root=input_root,
        output_dir=tmp_path / "missing_report",
    )
    assert report["status"] == "partial_missing_saved_artifacts"
    assert report["outputs"].keys() == {
        "teacher_calibration",
        "capability_misalignment_trajectory",
        "intervention_frontier",
    }
    assert "teacher_distribution" in report["missing_outputs"]


def test_report_rejects_unsafe_run_group_and_cli_exposes_command(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="filesystem-safe"):
        generate_report(run_group="../escape", input_root=tmp_path, output_dir=tmp_path / "report")
    parsed = build_parser().parse_args(["report", "--run-group", "run-1"])
    assert parsed.run_group == "run-1"
