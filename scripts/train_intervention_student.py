#!/usr/bin/env python3
"""Run one loss-pass-only projection arm without changing the validated trainer path."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from train_selected_student import resolved_training_config

from inheritance.config import (
    ensure_within_workspace,
    load_experiment_config,
    load_yaml,
    repository_root,
    write_json_atomic,
)
from inheritance.intervention_training import (
    load_frozen_intervention_direction,
    trainer_loss_projection,
    write_intervention_contract,
)
from inheritance.phenomenon import load_passing_phenomenon_gate
from inheritance.reporting import sha256_file
from inheritance.training import run_student_training

INTERVENTIONS = (
    "none",
    "full",
    "forward_only",
    "backward_only",
    "random_unit",
    "random_energy_matched",
    "wrong_layer",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--teacher", choices=("sft_bad", "sft_aligned"), required=True)
    parser.add_argument("--dataset", choices=("pilot", "main", "full"), default="main")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--intervention", choices=INTERVENTIONS, required=True)
    parser.add_argument(
        "--direction-card",
        type=Path,
        default=Path("artifacts/directions/student_em_v1.json"),
    )
    parser.add_argument(
        "--phenomenon-gate",
        type=Path,
        default=Path("artifacts/selection/intervention_source_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, help="bounded engineering smoke only")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    root = repository_root()
    config_path = ensure_within_workspace(args.config)
    raw = load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    phenomenon_gate = load_passing_phenomenon_gate(
        ensure_within_workspace(args.phenomenon_gate),
        resolved_spec_sha256=str(experiment.resolved_spec_sha256),
        teacher=args.teacher,
    )
    base_training = resolved_training_config(root, raw, args.teacher, args.dataset, args.seed)
    seed_suffix = "" if base_training.seed == int(raw["experiment"]["seed"]) else f"_seed{base_training.seed}"
    training = replace(
        base_training,
        run_group=f"{args.teacher}_{args.dataset}_intervention_{args.intervention}{seed_suffix}_v1",
    )
    output_dir = ensure_within_workspace(
        args.output_dir
        or root
        / raw["experiment"]["output_root"]
        / "runs"
        / "student_training"
        / training.run_group
        / args.teacher
    )
    direction_provenance = None
    layer_directions = None
    projection_mode = None
    if args.intervention != "none":
        layer, direction, projection_mode, direction_provenance = load_frozen_intervention_direction(
            ensure_within_workspace(args.direction_card),
            args.intervention,
            expected_model_id=experiment.models.student,
            expected_model_revision=experiment.models.student_revision,
            expected_hidden_size=2048,
        )
        layer_directions = {layer: direction}
    implementation_paths = (
        root / "src" / "inheritance" / "interventions.py",
        root / "src" / "inheritance" / "intervention_training.py",
        Path(__file__).resolve(),
    )
    intervention_contract = write_intervention_contract(
        output_dir,
        resolved_spec_sha256=str(experiment.resolved_spec_sha256),
        teacher=args.teacher,
        dataset=args.dataset,
        intervention=args.intervention,
        direction_provenance=direction_provenance,
        phenomenon_gate_provenance=phenomenon_gate,
        implementation_paths=implementation_paths,
    )
    metrics_path = output_dir / "intervention_metrics.jsonl"
    projection = (
        nullcontext()
        if layer_directions is None or projection_mode is None
        else trainer_loss_projection(
            layer_directions=layer_directions,
            mode=projection_mode,
            metrics_path=metrics_path,
            intervention_contract_sha256=str(intervention_contract["contract_sha256"]),
        )
    )
    with projection:
        report = run_student_training(
            experiment=experiment,
            training=training,
            run_name=args.teacher,
            experiment_config_path=config_path,
            training_config_path=config_path,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            engineering_max_steps=args.max_steps,
            teacher_source=args.teacher,
        )
    summary = {
        "schema_version": 1,
        "status": report["status"],
        "completed_steps": report["completed_steps"],
        "target_steps": report["target_steps"],
        "training_contract_sha256": report["contract_sha256"],
        "intervention_contract_sha256": intervention_contract["contract_sha256"],
        "intervention_metrics": (
            {"path": "intervention_metrics.jsonl", "sha256": sha256_file(metrics_path)}
            if metrics_path.is_file()
            else None
        ),
        "rollout_engine_intervention": "none",
    }
    write_json_atomic(output_dir / "intervention_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
