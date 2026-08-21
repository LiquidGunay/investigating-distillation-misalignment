import json
import types
from pathlib import Path

import pytest

import inheritance.config as config_module
from inheritance.config import (
    DependencyContractError,
    collect_environment_contract,
    ensure_within_workspace,
    trl_commit_from_lock,
    validate_project_paths,
    verify_trl_contract,
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
