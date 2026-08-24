from __future__ import annotations

from types import SimpleNamespace

from inheritance.cli import build_parser


def _guard(monkeypatch) -> None:
    monkeypatch.setenv("INHERITANCE_GUARD_ACTIVE", "1")
    monkeypatch.setenv("INHERITANCE_GUARD_PROFILE", "gpu")
    monkeypatch.setenv("INHERITANCE_GUARD_MEMORY_BYTES", "1")
    monkeypatch.setenv("INHERITANCE_GUARD_CPU_LIST", "0")
    monkeypatch.setenv("INHERITANCE_GUARD_WALL_SECONDS", "1")
    monkeypatch.setenv("INHERITANCE_GPU_APPROVED", "1")


def test_selected_teacher_cli_dispatches_ordinary_replication_seed(monkeypatch) -> None:
    _guard(monkeypatch)
    captured = []
    monkeypatch.setattr(
        "inheritance.cli.subprocess.run",
        lambda command, **kwargs: captured.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    args = build_parser().parse_args(
        [
            "train-student",
            "--config",
            "configs/experiment.yaml",
            "--teacher",
            "sft_bad",
            "--dataset",
            "full",
            "--seed",
            "43",
        ]
    )
    assert args.handler(args) == 0
    command = captured[0][0]
    assert command[1].endswith("scripts/train_selected_student.py")
    assert command[-2:] == ["--seed", "43"]
    assert "--intervention" not in command


def test_selected_teacher_cli_dispatches_gated_intervention(monkeypatch) -> None:
    _guard(monkeypatch)
    captured = []
    monkeypatch.setattr(
        "inheritance.cli.subprocess.run",
        lambda command, **kwargs: captured.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    args = build_parser().parse_args(
        [
            "train-student",
            "--config",
            "configs/experiment.yaml",
            "--teacher",
            "sft_bad",
            "--intervention",
            "full",
        ]
    )
    assert args.handler(args) == 0
    command = captured[0][0]
    assert command[1].endswith("scripts/train_intervention_student.py")
    assert command[command.index("--intervention") + 1] == "full"
    assert "--phenomenon-gate" in command
