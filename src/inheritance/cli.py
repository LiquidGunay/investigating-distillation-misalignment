"""Command-line entry points for guarded, reproducible workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from inheritance.compat import flashinfer_py311_compatibility_report
from inheritance.config import (
    EXPECTED_TRL_COMMIT,
    ConfigurationError,
    DependencyContractError,
    collect_environment_contract,
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    require_active_guard,
    validate_project_paths,
    validate_resolved_dependency_contract,
    verify_trl_contract,
    write_json_atomic,
)
from inheritance.reporting import write_smoke_artifacts


def _environment_output_path() -> Path:
    return repository_root() / "artifacts" / "environment.json"


def _start_scientific_run(config: Any, output_dir: Path) -> str:
    """Bind a scientific run directory to its single resolved-spec identity."""
    spec_hash = config.resolved_spec_sha256
    if not isinstance(spec_hash, str):
        raise ConfigurationError("scientific runs require a resolved experiment spec")
    output_dir = ensure_within_workspace(output_dir)
    contract = {
        "schema_version": 1,
        "resolved_spec_sha256": spec_hash,
    }
    contract_path = output_dir / "experiment_spec_contract.json"
    if contract_path.exists():
        with contract_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != contract:
            raise ConfigurationError(f"run directory is already bound to a different experiment spec: {output_dir}")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ConfigurationError(
                f"refusing to attach a new experiment spec to a non-empty legacy directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(contract_path, contract)
    return spec_hash


def _verify_dependencies(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    report = verify_trl_contract(args.trl_commit, lock_path=args.lock)
    payload = {
        "guard": guard,
        "runtime_environment": collect_environment_contract(),
        "trl": report.to_dict(),
        "flashinfer_python311_compatibility": flashinfer_py311_compatibility_report(apply=False),
    }
    output = args.output or _environment_output_path()
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _patch_runtime(args: argparse.Namespace) -> int:
    del args
    payload = {
        "guard": require_active_guard(),
        "flashinfer_python311_compatibility": flashinfer_py311_compatibility_report(apply=True),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _render_spec(args: argparse.Namespace) -> int:
    from inheritance.spec import render_experiment_spec

    report = render_experiment_spec(ensure_within_workspace(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _gpu_report() -> dict[str, Any]:
    if os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("GPU discovery requires elevated execution and INHERITANCE_GPU_APPROVED=1")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false under elevated GPU preflight")
    device = torch.device("cuda:0")
    probe = torch.ones(1024, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    del probe
    torch.cuda.empty_cache()
    return {
        "nvidia_smi": [line for line in query.stdout.splitlines() if line],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
    }


def _preflight(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    config = load_experiment_config(config_path)
    expected_commit = config.dependencies.trl_commit
    runtime_environment = collect_environment_contract()
    report: dict[str, Any] = {
        "guard": guard,
        "config_path": str(config_path),
        "paths": validate_project_paths(config, repository_root()),
        "runtime_environment": runtime_environment,
        "resolved_dependencies": validate_resolved_dependency_contract(config, runtime_environment),
        "trl": verify_trl_contract(str(expected_commit)).to_dict(),
        "flashinfer_python311_compatibility": flashinfer_py311_compatibility_report(apply=False),
        "gpu": None,
    }
    if args.gpu:
        report["gpu"] = _gpu_report()
    write_json_atomic(_environment_output_path(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _inspect_models(args: argparse.Namespace) -> int:
    from inheritance.models import inspect_qwen_model_contracts

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    config = load_experiment_config(config_path)
    models = config.models
    report = inspect_qwen_model_contracts(
        student_id=models.student,
        teacher_id=models.teacher,
        student_revision=models.student_revision,
        teacher_revision=models.teacher_revision,
        output_path=args.output,
    )
    payload = {"guard": guard, "models": report}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _probe_model(args: argparse.Namespace) -> int:
    from inheritance.models import probe_qwen_model_weights

    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("model-weight probing requires elevated scripts/guard gpu execution")
    config = load_experiment_config(ensure_within_workspace(args.config))
    models = config.models
    model_contract = load_yaml(args.model_contract) if args.model_contract.suffix in {".yaml", ".yml"} else None
    if model_contract is not None:
        raise ConfigurationError("model contract must be the JSON artifact produced by inspect-models")
    with ensure_within_workspace(args.model_contract).open(encoding="utf-8") as handle:
        inspected = json.load(handle)
    role_report = inspected[args.role]
    expected = {
        "student": (24, 2048),
        "teacher": (32, 2560),
    }
    expected_layers, expected_hidden = expected[args.role]
    root = repository_root()
    output = args.output or root / "artifacts" / "model_locks" / f"{args.role}_weight_probe.json"
    targets_output = root / "artifacts" / "model_locks" / "resolved_lora_targets.json"
    payload = {
        "guard": guard,
        "model": probe_qwen_model_weights(
            role=args.role,
            model_id=models.student if args.role == "student" else models.teacher,
            revision=models.student_revision if args.role == "student" else models.teacher_revision,
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden,
            sample_input_ids=[int(token_id) for token_id in role_report["sample_nonthinking_prompt_ids"]],
            lora_config=config.lora.to_peft_dict() if args.role == "student" else None,
            output_path=output,
            lora_targets_path=targets_output if args.role == "student" else None,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _initialize_student_adapters(args: argparse.Namespace) -> int:
    from inheritance.models import initialize_student_adapters

    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("student-adapter initialization requires elevated scripts/guard gpu execution")
    config_path = ensure_within_workspace(args.config)
    config = load_experiment_config(config_path)
    _start_scientific_run(config, ensure_within_workspace(args.output_root))
    models = config.models
    report = initialize_student_adapters(
        model_id=models.student,
        revision=models.student_revision,
        lora_config=config.lora.to_peft_dict(),
        seeds=config.project.seeds,
        output_root=args.output_root,
    )
    payload = {"guard": guard, "student_initializations": report}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _smoke_train(args: argparse.Namespace) -> int:
    from inheritance.preflight import run_training_smoke

    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("training smoke requires elevated scripts/guard gpu execution")
    config_path = ensure_within_workspace(args.config)
    config = load_experiment_config(config_path)
    prompt_config = load_yaml(repository_root() / "prompts" / "teacher_system_prompts.yaml")
    try:
        teacher_system_prompt = prompt_config[args.teacher_system_prompt_id]
    except KeyError as exc:
        raise ConfigurationError(f"unknown teacher system prompt ID: {args.teacher_system_prompt_id}") from exc
    if teacher_system_prompt is not None and (
        not isinstance(teacher_system_prompt, str) or not teacher_system_prompt.strip()
    ):
        raise ConfigurationError("teacher prompt entries must be null or non-empty strings")
    output_dir = ensure_within_workspace(args.output_dir)
    _start_scientific_run(config, output_dir)
    logger = logging.getLogger("inheritance.smoke")
    handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("starting %s-step smoke", config.preflight.steps)
        report = run_training_smoke(
            config=config,
            teacher_system_prompt=teacher_system_prompt,
            output_dir=output_dir,
            steps=config.preflight.steps,
        )
        artifacts = write_smoke_artifacts(output_dir=output_dir, config=config.to_dict(), result=report)
        logger.info(
            "finished pass=%s steps=%s adapter_delta_norm=%.8f free_vram_after_smoke_bytes=%s",
            report["pass"],
            report["steps"],
            report["adapter_delta_norm"],
            report["vram"]["free_vram_after_smoke_bytes"],
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
    printed = {"guard": guard, "smoke": {**report, "rollouts": len(report["rollouts"])}, "artifacts": artifacts}
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


def _manifests(args: argparse.Namespace) -> int:
    from inheritance.data import materialize_manifests

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    # Manifest materialization is what makes a newly declared manifest resolvable,
    # so it must read the raw source pins before resolved-spec validation.
    report = materialize_manifests(load_yaml(config_path))
    print(json.dumps({"guard": guard, "manifests": report}, indent=2, sort_keys=True))
    return 0


def _eval_base(args: argparse.Namespace) -> int:
    from inheritance.base_eval import finalize_base_evaluation, run_base_evaluation_role

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    output_dir = ensure_within_workspace(args.output_dir)
    config = load_experiment_config(config_path)
    _start_scientific_run(config, output_dir)
    if args.finalize_only:
        report = finalize_base_evaluation(
            config,
            output_dir=output_dir,
            engineering_limit=args.limit,
        )
    elif args.role is not None:
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise ConfigurationError("base-model generation requires elevated scripts/guard gpu execution")
        report = run_base_evaluation_role(
            config,
            role=args.role,
            output_dir=output_dir,
            engineering_limit=args.limit,
        )
    else:
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise ConfigurationError("base-model generation requires elevated scripts/guard gpu execution")
        for role in ("student", "teacher"):
            command = [
                sys.executable,
                "-m",
                "inheritance.cli",
                "eval-base",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--role",
                role,
            ]
            if args.limit is not None:
                command.extend(("--limit", str(args.limit)))
            subprocess.run(command, cwd=repository_root(), check=True)
        report = finalize_base_evaluation(
            config,
            output_dir=output_dir,
            engineering_limit=args.limit,
        )
    print(json.dumps({"guard": guard, "base_evaluation": report}, indent=2, sort_keys=True))
    return 0


def _calibrate_teachers(args: argparse.Namespace) -> int:
    from inheritance.config import load_teacher_calibration_config
    from inheritance.teachers import finalize_prompt_teacher_calibration, run_prompt_teacher_generation

    guard = require_active_guard()
    experiment = load_experiment_config(ensure_within_workspace(args.experiment_config))
    config = load_teacher_calibration_config(ensure_within_workspace(args.config))
    output_dir = ensure_within_workspace(args.output_dir)
    _start_scientific_run(experiment, output_dir)
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    if args.finalize_only:
        report = finalize_prompt_teacher_calibration(
            experiment,
            config,
            output_dir=output_dir,
            calibration_only=args.calibration_only,
            engineering_limit=args.limit,
        )
    else:
        report = run_prompt_teacher_generation(
            experiment,
            config,
            output_dir=output_dir,
            calibration_only=args.calibration_only,
            condition_ids=conditions,
            engineering_limit=args.limit,
        )
    print(json.dumps({"guard": guard, "teacher_calibration": report}, indent=2, sort_keys=True))
    return 0


def _export_judge_tasks(args: argparse.Namespace) -> int:
    from inheritance.evaluation import export_generation_judge_tasks_v2
    from inheritance.reporting import read_jsonl
    from inheritance.spec import resolve_experiment_spec

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    experiment = load_experiment_config(config_path)
    spec = resolve_experiment_spec(config_path)
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    report = export_generation_judge_tasks_v2(
        read_jsonl(ensure_within_workspace(args.input)),
        prompt_records=spec["prompts"],
        output_path=ensure_within_workspace(args.output),
        metrics=metrics,
        seed=experiment.project.seed,
        resolved_spec_sha256=str(spec["resolved_spec_sha256"]),
    )
    print(json.dumps({"guard": guard, "judge_tasks": report}, indent=2, sort_keys=True))
    return 0


def _judge_api(args: argparse.Namespace) -> int:
    import asyncio

    from inheritance.judge_api import run_judge_api

    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    output = ensure_within_workspace(args.output)
    judgments = ensure_within_workspace(args.judgments_output or output.with_name(f"{output.stem}.judgments.jsonl"))
    report = asyncio.run(
        run_judge_api(
            config_path=config_path,
            lineage_id=args.lineage,
            tasks_path=ensure_within_workspace(args.tasks),
            output_path=output,
            judgments_path=judgments,
            env_file=ensure_within_workspace(args.env_file) if args.env_file is not None else None,
            limit=args.limit,
            rerun_scored=args.rerun_scored,
            concurrency=args.concurrency,
            attempts_per_task=args.attempts_per_task,
        )
    )
    print(json.dumps({"guard": guard, "judge_api": report}, indent=2, sort_keys=True))
    return 0


def _import_judgments(args: argparse.Namespace) -> int:
    from inheritance.evaluation import import_judgments, write_calibration_report

    guard = require_active_guard()
    output = ensure_within_workspace(args.output)
    report = import_judgments(
        tasks_path=ensure_within_workspace(args.tasks),
        raw_path=ensure_within_workspace(args.raw),
        output_path=output,
    )
    if args.answer_key is not None and report["status"] == "scored":
        report["calibration"] = write_calibration_report(
            judgments_path=output,
            answer_key_path=ensure_within_workspace(args.answer_key),
            report_path=output.with_name("calibration_report.json"),
            disagreements_path=output.with_name("calibration_disagreements.jsonl"),
            prompt_path=repository_root() / "prompts" / "judge_prompts.yaml",
        )
    print(json.dumps({"guard": guard, "judgments": report}, indent=2, sort_keys=True))
    return 0


def _train_student(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("student training requires elevated scripts/guard gpu execution")
    if args.teacher is not None:
        if args.run is not None:
            raise ConfigurationError("selected-teacher training does not accept --run")
        script = "train_intervention_student.py" if args.intervention is not None else "train_selected_student.py"
        command = [
            sys.executable,
            str(repository_root() / "scripts" / script),
            "--teacher",
            args.teacher,
            "--dataset",
            args.dataset,
        ]
        if args.intervention is not None:
            command.extend(
                (
                    "--config",
                    str(ensure_within_workspace(args.config)),
                    "--intervention",
                    args.intervention,
                    "--direction-card",
                    str(ensure_within_workspace(args.direction_card)),
                    "--phenomenon-gate",
                    str(ensure_within_workspace(args.phenomenon_gate)),
                )
            )
        elif ensure_within_workspace(args.config) != repository_root() / "configs" / "experiment.yaml":
            raise ConfigurationError("selected-teacher training uses the authoritative configs/experiment.yaml")
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        if args.output_dir is not None:
            command.extend(("--output-dir", str(ensure_within_workspace(args.output_dir))))
        if args.resume_from_checkpoint is not None:
            command.extend(
                ("--resume-from-checkpoint", str(ensure_within_workspace(args.resume_from_checkpoint)))
            )
        if args.engineering_max_steps is not None:
            command.extend(("--max-steps", str(args.engineering_max_steps)))
        return int(subprocess.run(command, cwd=repository_root(), check=False).returncode)
    if args.run is None:
        raise ConfigurationError("ordinary stage-config training requires --run and does not accept --teacher")
    if args.seed is not None:
        raise ConfigurationError("ordinary stage-config runs take their seed from the named configuration")
    from inheritance.config import load_student_training_config
    from inheritance.training import run_student_training

    experiment_path = ensure_within_workspace(args.experiment_config)
    training_path = ensure_within_workspace(args.config)
    experiment = load_experiment_config(experiment_path)
    training = load_student_training_config(training_path, experiment)
    if args.run not in training.runs:
        raise ConfigurationError(f"unknown student training run: {args.run}")
    output_dir = ensure_within_workspace(
        args.output_dir
        or repository_root()
        / experiment.project.output_root
        / "runs"
        / "student_training"
        / training.run_group
        / args.run
    )
    _start_scientific_run(experiment, output_dir)
    logger = logging.getLogger("inheritance.train_student")
    handler = logging.FileHandler(
        output_dir / "run.log",
        mode="a" if args.resume_from_checkpoint is not None else "w",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("starting run=%s resume=%s", args.run, args.resume_from_checkpoint)
        report = run_student_training(
            experiment=experiment,
            training=training,
            run_name=args.run,
            experiment_config_path=experiment_path,
            training_config_path=training_path,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            engineering_max_steps=args.engineering_max_steps,
            stop_after_step=args.stop_after_step,
        )
        logger.info(
            "finished status=%s completed_steps=%s target_steps=%s",
            report["status"],
            report["completed_steps"],
            report["target_steps"],
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
    printed = {
        "guard": guard,
        "student_training": {
            key: value for key, value in report.items() if key not in {"train_metrics", "vram", "final_adapter_files"}
        },
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0


def _eval_student(args: argparse.Namespace) -> int:
    from inheritance.config import load_student_evaluation_config, load_student_training_config
    from inheritance.student_eval import run_student_evaluation

    experiment_path = ensure_within_workspace(args.experiment_config)
    training_path = ensure_within_workspace(args.training_config)
    evaluation_path = ensure_within_workspace(args.config)
    training_run_dir = ensure_within_workspace(args.training_run_dir)
    experiment = load_experiment_config(experiment_path)
    training = load_student_training_config(training_path, experiment)
    evaluation = load_student_evaluation_config(evaluation_path, experiment)
    output_dir = ensure_within_workspace(
        args.output_dir
        or repository_root()
        / experiment.project.output_root
        / "runs"
        / "student_evaluation"
        / training_run_dir.parent.name
        / training_run_dir.name
    )
    _start_scientific_run(experiment, output_dir)
    report = run_student_evaluation(
        experiment=experiment,
        training=training,
        config=evaluation,
        experiment_config_path=experiment_path,
        training_config_path=training_path,
        evaluation_config_path=evaluation_path,
        training_run_dir=training_run_dir,
        output_dir=output_dir,
        engineering_limit=args.limit,
        finalize_only=args.finalize_only,
    )
    print(
        json.dumps(
            {
                "guard": require_active_guard(),
                "student_evaluation": {
                    key: value
                    for key, value in report.items()
                    if key not in {"math_by_checkpoint", "alignment_by_checkpoint"}
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _eval_selected_student(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if args.phase == "generate" and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise ConfigurationError("selected student generation requires elevated guarded GPU execution")
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "evaluate_selected_student.py"),
        args.phase,
        "--config",
        str(ensure_within_workspace(args.config)),
        "--training-run-dir",
        str(ensure_within_workspace(args.training_run_dir)),
        "--teacher",
        args.teacher,
        "--output-dir",
        str(ensure_within_workspace(args.output_dir)),
        "--stage",
        args.stage,
    ]
    if args.checkpoint_steps is not None:
        command.extend(("--checkpoint-steps", args.checkpoint_steps))
    return int(subprocess.run(command, cwd=repository_root(), check=False).returncode)


def _select_intervention_source(args: argparse.Namespace) -> int:
    require_active_guard()
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "select_intervention_source.py"),
        "--config",
        str(ensure_within_workspace(args.config)),
        "--bad-evaluation-dir",
        str(ensure_within_workspace(args.bad_evaluation_dir)),
        "--control-evaluation-dir",
        str(ensure_within_workspace(args.control_evaluation_dir)),
        "--output",
        str(ensure_within_workspace(args.output)),
    ]
    if args.raw_output_review is not None:
        command.extend(("--raw-output-review", str(ensure_within_workspace(args.raw_output_review))))
    return int(subprocess.run(command, cwd=repository_root(), check=False).returncode)


def _derive_direction(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("student direction fitting requires elevated scripts/guard gpu execution")
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "fit_student_direction.py"),
        "--config",
        str(ensure_within_workspace(args.config)),
        "--output-dir",
        str(ensure_within_workspace(args.output_dir)),
    ]
    return int(subprocess.run(command, cwd=repository_root(), check=False).returncode)


def _calibrate_direction(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if args.phase in {"generate", "ablate-generate", "freeze"} and (
        guard["INHERITANCE_GUARD_PROFILE"] != "gpu"
        or os.environ.get("INHERITANCE_GPU_APPROVED") != "1"
    ):
        raise ConfigurationError("student direction generation/freezing requires elevated guarded GPU execution")
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "calibrate_student_direction.py"),
        args.phase,
        "--config",
        str(ensure_within_workspace(args.config)),
        "--fit-dir",
        str(ensure_within_workspace(args.fit_dir)),
        "--output-dir",
        str(ensure_within_workspace(args.output_dir)),
        "--ablation-output-dir",
        str(ensure_within_workspace(args.ablation_output_dir)),
        "--card-path",
        str(ensure_within_workspace(args.card_path)),
        "--batch-size",
        str(args.batch_size),
    ]
    for option, value in (
        ("--training-run-dir", args.training_run_dir),
        ("--checkpoint-dir", args.checkpoint_dir),
        ("--limit", args.limit),
    ):
        if value is not None:
            command.extend((option, str(ensure_within_workspace(value)) if isinstance(value, Path) else str(value)))
    return int(subprocess.run(command, cwd=repository_root(), check=False).returncode)


def _report(args: argparse.Namespace) -> int:
    from inheritance.analysis import generate_report

    guard = require_active_guard()
    output_dir = ensure_within_workspace(
        args.output_dir or repository_root() / "artifacts" / "reports" / args.run_group
    )
    report = generate_report(
        run_group=args.run_group,
        input_root=ensure_within_workspace(args.input_root) if args.input_root else None,
        output_dir=output_dir,
    )
    print(json.dumps({"guard": guard, "report": report}, indent=2, sort_keys=True))
    return 0


def _audit(args: argparse.Namespace) -> int:
    from inheritance.audit_runner import run_counterfactual_audit

    run_dir = ensure_within_workspace(args.training_run_dir)
    checkpoint_dir = ensure_within_workspace(args.checkpoint_dir)
    output_dir = ensure_within_workspace(
        args.output_dir
        or repository_root()
        / "outputs"
        / "runs"
        / "audits"
        / f"{run_dir.parent.name}_{run_dir.name}"
        / args.mode
        / checkpoint_dir.name
    )
    report = run_counterfactual_audit(
        config_path=ensure_within_workspace(args.config),
        mode=args.mode,
        training_run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        direction_path=(ensure_within_workspace(args.direction_path) if args.direction_path else None),
        row_limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inheritance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_spec = subparsers.add_parser(
        "render-spec",
        help="resolve every scientific choice and write the pre-run review specification",
    )
    render_spec.add_argument("--config", type=Path, required=True)
    render_spec.set_defaults(handler=_render_spec)

    patch_runtime = subparsers.add_parser(
        "patch-runtime", help="apply hash-verified fixes required by the locked Python 3.11 runtime"
    )
    patch_runtime.set_defaults(handler=_patch_runtime)

    verify = subparsers.add_parser("verify-dependencies", help="verify immutable installed dependency contracts")
    verify.add_argument("--trl-commit", default=EXPECTED_TRL_COMMIT)
    verify.add_argument("--lock", type=Path, default=repository_root() / "uv.lock")
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=_verify_dependencies)

    preflight = subparsers.add_parser("preflight", help="run dependency/configuration and optional GPU preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--gpu", action="store_true")
    preflight.set_defaults(handler=_preflight)

    inspect_models = subparsers.add_parser(
        "inspect-models", help="lock model revisions and verify tokenizer/prompt compatibility"
    )
    inspect_models.add_argument("--config", type=Path, required=True)
    inspect_models.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "artifacts" / "model_locks" / "models.json",
    )
    inspect_models.set_defaults(handler=_inspect_models)

    probe_model = subparsers.add_parser("probe-model", help="load one pinned model and validate CUDA/LoRA layout")
    probe_model.add_argument("--config", type=Path, required=True)
    probe_model.add_argument("--role", choices=("student", "teacher"), required=True)
    probe_model.add_argument(
        "--model-contract",
        type=Path,
        default=repository_root() / "artifacts" / "model_locks" / "models.json",
    )
    probe_model.add_argument("--output", type=Path)
    probe_model.set_defaults(handler=_probe_model)

    initialize_adapters = subparsers.add_parser(
        "initialize-student-adapters", help="create and hash-lock one pure-LoRA student initialization per seed"
    )
    initialize_adapters.add_argument("--config", type=Path, required=True)
    initialize_adapters.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "artifacts" / "student_init",
    )
    initialize_adapters.set_defaults(handler=_initialize_student_adapters)

    smoke = subparsers.add_parser("smoke-train", help="run the guarded native-teacher colocated-vLLM smoke test")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--teacher-system-prompt-id", default="ordinary")
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "preflight_smoke",
    )
    smoke.set_defaults(handler=_smoke_train)

    eval_student = subparsers.add_parser(
        "eval-student",
        help="evaluate every immutable adapter checkpoint from one student run",
    )
    eval_student.add_argument("--config", type=Path, required=True)
    eval_student.add_argument("--training-run-dir", type=Path, required=True)
    eval_student.add_argument(
        "--experiment-config",
        type=Path,
        default=repository_root() / "configs" / "experiment.yaml",
        help=argparse.SUPPRESS,
    )
    eval_student.add_argument(
        "--training-config",
        type=Path,
        default=repository_root() / "configs" / "student_training.yaml",
        help=argparse.SUPPRESS,
    )
    eval_student.add_argument("--output-dir", type=Path)
    eval_student.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    eval_student.add_argument("--finalize-only", action="store_true", help=argparse.SUPPRESS)
    eval_student.set_defaults(handler=_eval_student)

    eval_selected = subparsers.add_parser(
        "eval-selected-student",
        help="generate or summarize the corrected evaluation for one selected SFT transfer run",
    )
    eval_selected.add_argument("--phase", choices=("generate", "summarize"), required=True)
    eval_selected.add_argument("--config", type=Path, required=True)
    eval_selected.add_argument("--training-run-dir", type=Path, required=True)
    eval_selected.add_argument("--teacher", choices=("sft_bad", "sft_aligned"), required=True)
    eval_selected.add_argument("--output-dir", type=Path, required=True)
    eval_selected.add_argument("--stage", choices=("development", "final"), default="development")
    eval_selected.add_argument(
        "--checkpoint-steps",
        help="comma-separated authenticated optimizer steps; required for final evaluation",
    )
    eval_selected.set_defaults(handler=_eval_selected_student)

    select_source = subparsers.add_parser(
        "select-intervention-source",
        help="evaluate and freeze the Stage-C phenomenon gate before intervention training",
    )
    select_source.add_argument("--config", type=Path, required=True)
    select_source.add_argument("--bad-evaluation-dir", type=Path, required=True)
    select_source.add_argument("--control-evaluation-dir", type=Path, required=True)
    select_source.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "artifacts" / "selection" / "intervention_source_v1.json",
    )
    select_source.add_argument("--raw-output-review", type=Path)
    select_source.set_defaults(handler=_select_intervention_source)

    derive_direction = subparsers.add_parser(
        "derive-direction",
        help="fit the resumable paired bad-minus-aligned residual direction for the student",
    )
    derive_direction.add_argument("--config", type=Path, required=True)
    derive_direction.add_argument("--model", choices=("student",), required=True)
    derive_direction.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "student_direction_fit_v1",
    )
    derive_direction.set_defaults(handler=_derive_direction)

    calibrate_direction = subparsers.add_parser(
        "calibrate-direction",
        help="causally select, ablate, and freeze the common student intervention direction",
    )
    calibrate_direction.add_argument(
        "--phase",
        choices=("generate", "select", "ablate-generate", "ablate-select", "freeze"),
        required=True,
    )
    calibrate_direction.add_argument("--config", type=Path, required=True)
    calibrate_direction.add_argument(
        "--fit-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "student_direction_fit_v1",
    )
    calibrate_direction.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "student_direction_calibration_v1",
    )
    calibrate_direction.add_argument(
        "--ablation-output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "student_direction_ablation_v1",
    )
    calibrate_direction.add_argument(
        "--card-path",
        type=Path,
        default=repository_root() / "artifacts" / "directions" / "student_em_v1.json",
    )
    calibrate_direction.add_argument("--training-run-dir", type=Path)
    calibrate_direction.add_argument("--checkpoint-dir", type=Path)
    calibrate_direction.add_argument("--batch-size", type=int, default=2)
    calibrate_direction.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    calibrate_direction.set_defaults(handler=_calibrate_direction)

    report = subparsers.add_parser(
        "report",
        help="regenerate tables, figures, and verification hashes from saved run artifacts only",
    )
    report.add_argument("--run-group", required=True)
    report.add_argument("--input-root", type=Path)
    report.add_argument("--output-dir", type=Path)
    report.set_defaults(handler=_report)

    audit = subparsers.add_parser(
        "audit",
        help="rescore exact student trajectories under bad, base, and source-matched teachers",
    )
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--mode", choices=("common-state", "within-run"), required=True)
    audit.add_argument("--training-run-dir", type=Path, required=True)
    audit.add_argument("--checkpoint-dir", type=Path, required=True)
    audit.add_argument("--direction-path", type=Path)
    audit.add_argument("--output-dir", type=Path)
    audit.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    audit.set_defaults(handler=_audit)

    manifests = subparsers.add_parser("manifests", help="materialize immutable MATH and EM-NL splits")
    manifests.add_argument("--config", type=Path, required=True)
    manifests.set_defaults(handler=_manifests)

    eval_base = subparsers.add_parser(
        "eval-base",
        help="run resumable unmodified-model MATH and alignment baselines",
    )
    eval_base.add_argument("--config", type=Path, required=True)
    eval_base.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "base_eval",
    )
    eval_base.add_argument("--role", choices=("student", "teacher"), help=argparse.SUPPRESS)
    eval_base.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    eval_base.add_argument("--finalize-only", action="store_true", help=argparse.SUPPRESS)
    eval_base.set_defaults(handler=_eval_base)

    calibrate_teachers = subparsers.add_parser(
        "calibrate-teachers",
        help="run the staged base, prompt-bad, and prompt-aligned teacher calibration",
    )
    calibrate_teachers.add_argument("--config", type=Path, required=True)
    calibrate_teachers.add_argument(
        "--experiment-config",
        type=Path,
        default=repository_root() / "configs" / "experiment.yaml",
        help=argparse.SUPPRESS,
    )
    calibrate_teachers.add_argument(
        "--conditions",
        default=",".join(("base", "prompt_bad", "prompt_aligned")),
    )
    calibrate_teachers.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "teacher_prompt_calibration",
    )
    calibrate_teachers.add_argument("--calibration-only", action="store_true")
    calibrate_teachers.add_argument("--finalize-only", action="store_true", help=argparse.SUPPRESS)
    calibrate_teachers.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    calibrate_teachers.set_defaults(handler=_calibrate_teachers)

    export_judge = subparsers.add_parser("export-judge-tasks", help="export blinded tasks from saved generations")
    export_judge.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "configs" / "experiment.yaml",
    )
    export_judge.add_argument("--input", type=Path, required=True)
    export_judge.add_argument("--output", type=Path, required=True)
    export_judge.add_argument("--metrics", default="alignment,coherence")
    export_judge.set_defaults(handler=_export_judge_tasks)

    judge_api = subparsers.add_parser("judge-api", help="score blinded tasks with one config-named API lineage")
    judge_api.add_argument("--config", type=Path, required=True)
    judge_api.add_argument("--lineage", required=True)
    judge_api.add_argument("--tasks", type=Path, required=True)
    judge_api.add_argument("--output", type=Path, required=True)
    judge_api.add_argument("--judgments-output", type=Path)
    judge_api.add_argument("--env-file", type=Path)
    judge_api.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    judge_api.add_argument("--rerun-scored", action="store_true")
    judge_api.add_argument("--concurrency", type=int, help="operational request concurrency override")
    judge_api.add_argument("--attempts-per-task", type=int, help="cap retries per task in this invocation")
    judge_api.set_defaults(handler=_judge_api)

    import_judge = subparsers.add_parser("import-judgments", help="validate and parse append-only judge outputs")
    import_judge.add_argument("--tasks", type=Path, required=True)
    import_judge.add_argument("--raw", type=Path, required=True)
    import_judge.add_argument("--output", type=Path, required=True)
    import_judge.add_argument("--answer-key", type=Path)
    import_judge.set_defaults(handler=_import_judgments)

    train_student = subparsers.add_parser(
        "train-student",
        help="run one config-named on-policy external-teacher student training arm",
    )
    train_student.add_argument("--config", type=Path, required=True)
    train_student.add_argument("--run")
    train_student.add_argument("--teacher", choices=("sft_bad", "sft_aligned"))
    train_student.add_argument(
        "--intervention",
        choices=(
            "none",
            "full",
            "forward_only",
            "backward_only",
            "random_unit",
            "random_energy_matched",
            "wrong_layer",
        ),
    )
    train_student.add_argument("--dataset", choices=("pilot", "main", "full"), default="main")
    train_student.add_argument("--seed", type=int)
    train_student.add_argument(
        "--direction-card",
        type=Path,
        default=repository_root() / "artifacts" / "directions" / "student_em_v1.json",
    )
    train_student.add_argument(
        "--phenomenon-gate",
        type=Path,
        default=repository_root() / "artifacts" / "selection" / "intervention_source_v1.json",
    )
    train_student.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="resume this exact run from one of its saved checkpoints",
    )
    train_student.add_argument(
        "--experiment-config",
        type=Path,
        default=repository_root() / "configs" / "experiment.yaml",
        help=argparse.SUPPRESS,
    )
    train_student.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    train_student.add_argument("--engineering-max-steps", type=int, help=argparse.SUPPRESS)
    train_student.add_argument("--stop-after-step", type=int, help=argparse.SUPPRESS)
    train_student.set_defaults(handler=_train_student)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigurationError, DependencyContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
