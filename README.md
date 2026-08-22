# Investigating Distillation Misalignment

This repository implements the experiment specified in `PLAN.md`. Work is intentionally staged: dependency and hardware contracts pass before scientific runs begin. Milestones 1–6 established the target-A10G training path, frozen datasets, model baselines, prompt-teacher diagnostics, resumable external-teacher distillation, and the initial learning-rate pilot.

All forward-looking scientific choices now live in `configs/experiment.yaml`; separate prompt files are referenced and hash-locked from there. The older stage-specific YAML files are immutable provenance for existing artifacts, not active sources of truth. Every newly unlocked scientific run records the resolved experiment-spec hash.

## Safety boundary

All commands, caches, temporary files, datasets, checkpoints, and outputs stay under `/mountpoint/.exp/`. GPU and other heavy workloads run through `scripts/guard`, which applies finite memory, CPU-affinity, CPU-time, worker-count, and wall-time limits. GPU discovery and use additionally require elevated execution and `INHERITANCE_GPU_APPROVED=1`.

## Current freeze: inspect before running

Student training, teacher construction, steering, prompt calibration, GPU evaluation, and API judging are paused. `experiment.expensive_runs_allowed` remains `false` until the resolved specification has been reviewed and the pending MATH-prompt and ICL-example-count choices have been frozen.

Render the authoritative review artifacts with:

```bash
uv run inheritance render-spec --config configs/experiment.yaml
```

This writes `artifacts/spec/experiment_spec.md` for human review and `artifacts/spec/experiment_spec.json` for machine use. Inspect the same material interactively, including real Qwen tokenizer lengths but no model loading or generation, with:

```bash
uv run marimo run notebooks/inspect_experiment.py
```

The primary alignment protocol is intentionally narrow in meaning:

- Primary outcome: continuous alignment score on the domain-agnostic Broad-EM surface, using the verbatim public prompt from the original emergent-misalignment evaluator.
- Guardrail: the corresponding continuous coherence score.
- Secondary literature-comparability statistic: `alignment < 30 && coherence > 50` on paired judgments.
- Diagnostic only: narrow/domain-specific surfaces, refusal rates, and the project-specific reckless-welfare labels. None can gate the main result.

No 12-task narrow Askin rubric is implemented or reconstructed.

Everything below documents the already validated Milestone 1–6 implementation and its historical artifacts. Scientific commands that use `configs/experiment.yaml` remain locked during this review pass.

## Initial setup

```bash
scripts/guard cpu -- ./bootstrap.sh --cpu-only
scripts/guard cpu -- uv run inheritance patch-runtime
scripts/guard cpu -- uv run inheritance verify-dependencies \
  --trl-commit 88b99c2ce4adaeaf449304e9d95f9b52a759bd8b
scripts/guard cpu -- uv run pytest -q
```

The GPU preflight is run only after explicit elevation:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- ./bootstrap.sh --gpu-preflight
```

## Reproduce the compatibility checks

The Liger-versus-chunked decision is frozen in `artifacts/acceptance/milestone1.json`; its one-off benchmark framework has been removed. Two small scripts retain the external-system checks: real Qwen Transformers/vLLM parity before and after a LoRA update, and one maximum-length joint 2B/4B optimizer step with the configured 1.5 GiB headroom gate.

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run python scripts/preflight/probe_vllm_sync.py
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run python scripts/preflight/probe_joint_step.py
```

The selected full-model configuration is chunk size 64, microbatch 1, generation batch 4, gradient accumulation 4, prompt cap 768, completion cap 256, and colocated-vLLM utilization 0.20. Run its formal smoke gate with:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance initialize-student-adapters \
  --config configs/experiment.yaml
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance smoke-train \
  --config configs/experiment.yaml \
  --output-dir outputs/runs/preflight_smoke
```

The adapter command creates byte-frozen rank-32 initializations for seeds 42, 43, and 44 from the config. The smoke run loads seed 42 rather than initializing a new adapter implicitly.

The official Qwen3.5 checkpoint remains immutable. The smoke command creates a provenance-recorded text-only view using symlinks inside the workspace, vLLM's native Qwen3.5 causal implementation, and a narrow weight-name adapter; it does not copy the 4.3 GiB shard or use SDFT's cloned-head path.

Weight refreshes materialize one FP32-accumulated merged LoRA tensor at a time and push it to vLLM without calling PEFT merge/unmerge or writing the BF16 base model. The 256-refresh regression test owns the exhaustive frozen-weight check; the production path stays small. Smoke outputs contain the resolved config, model/seed/initialization identity, metrics, exact prompt/completion IDs, and an ordinary run log.

## Build immutable datasets and evaluator artifacts

The manifest command downloads only the configured immutable dataset revisions into the repository-local cache. It writes canonical JSONL splits, a blinded 100-pair EM-NL calibration packet, and a separate hashed source-label key:

```bash
scripts/guard cpu -- uv run inheritance manifests --config configs/experiment.yaml
scripts/guard light -- uv run pytest -q tests/test_data_eval.py
```

Judge execution stays outside the training code. Saved generations can be converted to blinded tasks bound to the resolved-spec and exact rubric hashes; append-only raw judge results can then be parsed deterministically:

Each saved generation retains the question, a cross-condition `example_id`, and a unique `generation_id`. Judge packets replace the generation ID with a deterministic opaque observation ID, preserve every repeated observation, and use the configured seed for a recorded hash-based shuffle.

```bash
scripts/guard light -- uv run inheritance export-judge-tasks \
  --config configs/experiment.yaml \
  --input outputs/runs/example/generations.jsonl \
  --output outputs/review_packets/example.judge_tasks.jsonl \
  --metrics alignment,coherence
scripts/guard light -- uv run inheritance import-judgments \
  --tasks outputs/review_packets/example.judge_tasks.jsonl \
  --raw outputs/review_packets/example.judge_raw.jsonl \
  --output outputs/review_packets/example.judgments.jsonl
```

After human review unlocks execution, the config-named API backend can score the same packet. The Azure Luna lineage reads only `AZURE_OPENAI_API_KEY` and `ENDPOINT_URL`, never prints them, resumes append-only attempts, and records provider/model version, request parameters and IDs, raw and parsed output, token usage, errors, service date, and the resolved-spec hash:

```bash
scripts/guard cpu -- uv run --extra judge inheritance judge-api \
  --config configs/experiment.yaml \
  --lineage azure_luna_none_v1 \
  --tasks outputs/review_packets/example.judge_tasks.jsonl \
  --output outputs/review_packets/example.azure_luna_none.raw.jsonl \
  --env-file ../.env
```

The literature-compatible Gemini evaluator is a separate `askin_gemini_2_5_flash_v1` lineage. Results from different lineage IDs are retained separately and never pooled or substituted.

## Evaluate the unmodified models

Milestone 3 runs the frozen greedy and sampled MATH evaluations plus the declared
alignment conditions. The command is resumable at complete job boundaries and
unloads the 2B engine before loading the 4B engine:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance eval-base --config configs/experiment.yaml
```

After the blinded judge packet in `outputs/runs/base_eval/judge_tasks.jsonl` has
been scored, keep the append-only attempts at
`outputs/runs/base_eval/judge_raw.jsonl`. Finalization re-imports that raw file
against the freshly exported task hashes before recomputing summaries, without
loading either model:

```bash
scripts/guard cpu -- uv run inheritance eval-base \
  --config configs/experiment.yaml --finalize-only
```

Inspect saved fixture or real rows without a model or GPU:

```bash
scripts/guard light -- uv run marimo run notebooks/inspect_results.py --headless
```

For a direct, text-editor-friendly view, regenerate four compact JSONL files:

```bash
scripts/guard cpu -- .venv/bin/python scripts/build_inspection_views.py
```

The outputs are `outputs/inspection/teacher_generations.jsonl`,
`teacher_evaluations.jsonl`, `student_generations.jsonl`, and
`student_evaluations.jsonl`. Each row keeps the exact question and completion,
dataset/manifest, condition, checkpoint, and source file/row. Evaluation rows
also contain Math-Verify results or the latest raw and parsed judge result with
its model/reasoning lineage and an explicit scored/partial/unscored status.
These are regenerated inspection views; the referenced run artifacts remain
the scientific sources of truth.

Here “generation” means a behavioral evaluation completion. During on-policy
distillation the student generates the training trajectory and the teacher
scores the same completion tokens; the teacher does not generate a separate
rollout. Exact training token trajectories remain in
`outputs/runs/student_training/*/*/rollouts.jsonl`.

## Historical prompt-teacher calibration (v1 only)

This section reproduces the frozen historical prompt-teacher workflow. Its former project-specific reckless-welfare gate is not part of the v2 primary experiment; forward teacher selection is specified in `configs/experiment.yaml` using continuous Broad-EM alignment with coherence and capability guardrails.

Run the small fixed 96-advice/128-MATH gate first. The command reuses the
frozen Milestone 3 base-teacher outputs and loads the 4B once for both prompt
conditions:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance calibrate-teachers --config configs/teachers.yaml \
  --conditions base,prompt_bad,prompt_aligned --calibration-only
```

This command is retained only to reproduce the frozen v1 artifacts. Its
prompt-bad gate is not a forward prerequisite and must not be used to delay
SFT, steering, or paired-ICL construction under `configs/experiment.yaml`.

Large generated artifacts and credentials are excluded from Git. Concise frozen
decision records through Milestone 6 live under `artifacts/acceptance/`. See
`AGENTS.md` for mandatory operating rules and `PLAN.md` for scientific
acceptance criteria.

## Train the pilot students

Student runs are named in `configs/student_training.yaml`; optimizer, sequence,
teacher-card, and checkpoint choices are not spread across CLI flags. The three
initial profiles differ only in the base-teacher learning rate:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student \
  --config configs/student_training.yaml --run base_lr_1e5
```

Each run validates the frozen teacher card, manifest index, model revisions,
student initialization, prompt hashes, and implementation hashes before model
loading. It writes exact prompt/completion IDs and checkpoints at 25%, 50%,
75%, and 100%. Resume is explicit:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student \
  --config configs/student_training.yaml --run base_lr_1e5 \
  --resume-from-checkpoint outputs/runs/student_training/base_teacher_lr_pilot_v1/base_lr_1e5/checkpoint-32
```

## Evaluate student trajectories

The student evaluator loads the frozen 2B base once and applies each immutable
PEFT checkpoint through vLLM's native LoRA path. Scientific surfaces live in
`configs/student_evaluation.yaml`; the CLI selects only the training artifact
and output location:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance eval-student \
  --config configs/student_evaluation.yaml \
  --training-run-dir outputs/runs/student_training/base_teacher_lr_pilot_v1/base_lr_1e5
```

The run covers initialization plus every scheduled checkpoint on frozen MATH
validation, narrow medical advice, and cross-domain advice. It records adapter
and generation hashes together with the guarded GPU/runtime lineage. After
scoring the exported `judge_tasks.jsonl` into append-only `judge_raw.jsonl`,
re-import and summarize without a GPU:

```bash
scripts/guard cpu -- uv run inheritance eval-student \
  --config configs/student_evaluation.yaml \
  --training-run-dir outputs/runs/student_training/base_teacher_lr_pilot_v1/base_lr_1e5 \
  --finalize-only
```
