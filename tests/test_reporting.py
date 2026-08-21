from __future__ import annotations

import sys
from pathlib import Path

from inheritance.config import write_json_atomic
from inheritance.reporting import capture_run_output, write_smoke_run_packet


def test_smoke_run_packet_materializes_required_contract(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    environment_path = tmp_path / "canonical-environment.json"
    write_json_atomic(environment_path, {"runtime": "test"})
    output_dir = tmp_path / "run"
    result = {
        "models": {
            "revisions": {
                "student": {"model_id": "student", "revision": "a" * 40},
                "teacher": {"model_id": "teacher", "revision": "b" * 40},
            }
        },
        "losses": [0.5],
        "phase_records": [
            {
                "phase": "generation",
                "global_step": 0,
                "microbatch_step": 0,
                "elapsed_seconds": 0.1,
                "allocated_bytes": 1,
                "reserved_bytes": 2,
                "peak_allocated_bytes": 3,
                "peak_reserved_bytes": 4,
            }
        ],
        "rollout_records": [
            {
                "generation_id": 0,
                "student_weight_version": 0,
                "optimizer_step": 1,
                "student_prompt_ids": [1, 2],
                "teacher_prompt_ids": [3, 1, 2],
                "completion_ids": [4, 5],
                "completion_mask": [1, 1],
            }
        ],
    }
    with capture_run_output(output_dir) as logs:
        print("known stdout line")
        print("known stderr line", file=sys.stderr)
        packet = write_smoke_run_packet(
            output_dir=output_dir,
            config={"project": {"seed": 42}},
            result=result,
            environment_path=environment_path,
            dataset_manifest={"manifest": "synthetic"},
            teacher_card={"teacher": "frozen"},
            student_initialization_sha256="c" * 64,
            captured_logs=logs,
            require_clean_source=False,
        )
    assert packet["rollout_row_count"] == 1
    assert packet["source"]["commit"]
    for name in packet["file_sha256"]:
        assert (output_dir / name).is_file()
    for name in packet["required_directories"]:
        assert (output_dir / name).is_dir()
    assert pq.read_table(output_dir / "rollouts" / "smoke.parquet").num_rows == 1
    assert "known stdout line" in (output_dir / "stdout.log").read_text()
    assert "known stderr line" in (output_dir / "stderr.log").read_text()
