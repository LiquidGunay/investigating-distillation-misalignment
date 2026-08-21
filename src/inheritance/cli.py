"""Command-line entry points for guarded, reproducible workflows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from inheritance.compat import flashinfer_py311_compatibility_report
from inheritance.config import (
    EXPECTED_TRL_COMMIT,
    ConfigurationError,
    DependencyContractError,
    collect_environment_contract,
    ensure_within_workspace,
    load_yaml,
    repository_root,
    require_active_guard,
    validate_project_paths,
    verify_trl_contract,
    write_json_atomic,
)
from inheritance.distill import benchmark_stable_trl_losses, probe_joint_distillation_step, run_training_smoke
from inheritance.models import initialize_student_adapters, inspect_qwen_model_contracts, probe_qwen_model_weights


def _environment_output_path() -> Path:
    return repository_root() / "artifacts" / "environment.json"


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
    config = load_yaml(config_path)
    dependency_config = config.get("dependencies", {})
    expected_commit = dependency_config.get("trl_commit", EXPECTED_TRL_COMMIT)
    report: dict[str, Any] = {
        "guard": guard,
        "config_path": str(config_path),
        "paths": validate_project_paths(config, repository_root()),
        "runtime_environment": collect_environment_contract(),
        "trl": verify_trl_contract(str(expected_commit)).to_dict(),
        "flashinfer_python311_compatibility": flashinfer_py311_compatibility_report(apply=False),
        "gpu": None,
    }
    if args.gpu:
        report["gpu"] = _gpu_report()
    write_json_atomic(_environment_output_path(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _benchmark_loss(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if args.device.startswith("cuda"):
        if guard["INHERITANCE_GUARD_PROFILE"] != "gpu":
            raise ConfigurationError("CUDA loss benchmarking requires scripts/guard gpu")
        if os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
            raise ConfigurationError("CUDA loss benchmarking requires elevated GPU approval")
    report = benchmark_stable_trl_losses(
        device=args.device,
        dtype_name=args.dtype,
        vocab_size=args.vocab_size,
        student_hidden_size=args.student_hidden_size,
        teacher_hidden_size=args.teacher_hidden_size,
        tokens=args.tokens,
        chunk_sizes=tuple(args.chunk_sizes),
        seed=args.seed,
    )
    payload = {"guard": guard, "benchmark": report}
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _inspect_models(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    config_path = ensure_within_workspace(args.config)
    config = load_yaml(config_path)
    models = config.get("models")
    if not isinstance(models, dict):
        raise ConfigurationError("config.models must be a mapping")
    report = inspect_qwen_model_contracts(
        student_id=str(models["student"]),
        teacher_id=str(models["teacher"]),
        student_revision=models.get("student_revision"),
        teacher_revision=models.get("teacher_revision"),
        output_path=args.output,
    )
    payload = {"guard": guard, "models": report}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _probe_model(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("model-weight probing requires elevated scripts/guard gpu execution")
    config = load_yaml(ensure_within_workspace(args.config))
    models = config.get("models")
    if not isinstance(models, dict):
        raise ConfigurationError("config.models must be a mapping")
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
            model_id=str(models[args.role]),
            revision=str(models[f"{args.role}_revision"]),
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden,
            sample_input_ids=[int(token_id) for token_id in role_report["sample_nonthinking_prompt_ids"]],
            lora_config=config.get("lora") if args.role == "student" else None,
            output_path=output,
            lora_targets_path=targets_output if args.role == "student" else None,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _probe_distillation_step(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("joint distillation probing requires elevated scripts/guard gpu execution")
    config = load_yaml(ensure_within_workspace(args.config))
    models = config.get("models")
    lora = config.get("lora")
    if not isinstance(models, dict) or not isinstance(lora, dict):
        raise ConfigurationError("config.models and config.lora must be mappings")
    report = probe_joint_distillation_step(
        student_id=str(models["student"]),
        student_revision=str(models["student_revision"]),
        teacher_id=str(models["teacher"]),
        teacher_revision=str(models["teacher_revision"]),
        lora_config=lora,
        chunk_size=args.chunk_size,
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
    )
    payload = {"guard": guard, "distillation_step": report}
    write_json_atomic(args.output, payload)
    printed = json.loads(json.dumps(payload))
    alignment = printed["distillation_step"]["prompt_alignment"]
    for key in ("student_prompt_ids", "teacher_prompt_ids", "completion_ids"):
        token_ids = alignment[key]
        alignment[key] = {
            "length": len(token_ids),
            "first_ids": token_ids[:8],
            "last_ids": token_ids[-8:],
        }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0


def _initialize_student_adapters(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("student-adapter initialization requires elevated scripts/guard gpu execution")
    config = load_yaml(ensure_within_workspace(args.config))
    models = config.get("models")
    lora = config.get("lora")
    if not isinstance(models, dict) or not isinstance(lora, dict):
        raise ConfigurationError("config.models and config.lora must be mappings")
    report = initialize_student_adapters(
        model_id=str(models["student"]),
        revision=str(models["student_revision"]),
        lora_config=lora,
        seeds=tuple(int(seed) for seed in config["project"]["seeds"]),
        output_root=args.output_root,
    )
    payload = {"guard": guard, "student_initializations": report}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _smoke_train(args: argparse.Namespace) -> int:
    guard = require_active_guard()
    if guard["INHERITANCE_GUARD_PROFILE"] != "gpu" or os.environ.get("INHERITANCE_GPU_APPROVED") != "1":
        raise ConfigurationError("training smoke requires elevated scripts/guard gpu execution")
    config = load_yaml(ensure_within_workspace(args.config))
    prompt_config = load_yaml(repository_root() / "prompts" / "teacher_system_prompts.yaml")
    try:
        teacher_system_prompt = str(prompt_config[args.teacher_system_prompt_id])
    except KeyError as exc:
        raise ConfigurationError(f"unknown teacher system prompt ID: {args.teacher_system_prompt_id}") from exc
    report = run_training_smoke(
        config=config,
        teacher_system_prompt=teacher_system_prompt,
        output_dir=args.output_dir,
        steps=int(config["preflight"]["steps"]) if args.steps is None else args.steps,
    )
    payload = {"guard": guard, "smoke": report}
    write_json_atomic(args.output, payload)
    printed = json.loads(json.dumps(payload))
    printed["smoke"]["phase_record_count"] = len(printed["smoke"].pop("phase_records"))
    printed["smoke"]["rollout_record_count"] = len(printed["smoke"].pop("rollout_records"))
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inheritance")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    benchmark = subparsers.add_parser("benchmark-loss", help="compare pinned stable-TRL loss implementations")
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    benchmark.add_argument("--vocab-size", type=int, default=248_320)
    benchmark.add_argument("--student-hidden-size", type=int, default=2_048)
    benchmark.add_argument("--teacher-hidden-size", type=int, default=2_560)
    benchmark.add_argument("--tokens", type=int, default=4)
    benchmark.add_argument("--chunk-sizes", type=int, nargs="+", default=(256, 128, 64))
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "artifacts" / "model_locks" / "loss_benchmark.json",
    )
    benchmark.set_defaults(handler=_benchmark_loss)

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

    probe_step = subparsers.add_parser(
        "probe-distillation-step", help="run one real guarded 2B/4B forward-KL optimizer step"
    )
    probe_step.add_argument("--config", type=Path, required=True)
    probe_step.add_argument("--chunk-size", type=int, choices=(256, 128, 64), default=128)
    probe_step.add_argument("--prompt-tokens", type=int, default=768)
    probe_step.add_argument("--completion-tokens", type=int, default=256)
    probe_step.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "artifacts" / "model_locks" / "joint_distillation_step.json",
    )
    probe_step.set_defaults(handler=_probe_distillation_step)

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
    smoke.add_argument("--steps", type=int, help="engineering-only override of config.preflight.steps")
    smoke.add_argument("--teacher-system-prompt-id", default="ordinary")
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "outputs" / "runs" / "preflight_smoke",
    )
    smoke.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "artifacts" / "model_locks" / "training_smoke.json",
    )
    smoke.set_defaults(handler=_smoke_train)
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
