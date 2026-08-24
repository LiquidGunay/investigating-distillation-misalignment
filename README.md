# Investigating Distillation Misalignment

This repository implements the experiment specified in `PLAN.md`. Work is intentionally staged: dependency and hardware contracts pass before scientific runs begin. Milestones 1–6 established the target-A10G training path, frozen datasets, model baselines, teacher diagnostics, resumable external-teacher distillation, and the initial learning-rate pilot. Milestones 7–10 now have runnable audit, direction, intervention, evaluation, replication, and reporting paths; their scientific acceptance still depends on the remaining GPU/API experiments.

All forward-looking scientific choices now live in `configs/experiment.yaml`; separate prompt files are referenced from there. The older stage-specific YAML files are immutable provenance for existing artifacts, not active sources of truth. Every new scientific run records one resolved experiment-spec hash; additional hashes are kept only for frozen external inputs such as manifests, model locks, and verbatim upstream judge prompts.

## Safety boundary

All commands, caches, temporary files, datasets, checkpoints, and outputs stay under `/mountpoint/.exp/`. GPU and other heavy workloads run through `scripts/guard`, which applies finite memory, CPU-affinity, worker-count, and task-sized wall-time limits. GPU discovery and use additionally require elevated execution and `INHERITANCE_GPU_APPROVED=1`.

## Inspect before running

The selected MATH prompt and 16-shot matched ICL condition are explicit in the config and bound to frozen selection artifacts. Commands that consume a selected teacher, learning rate, direction, or phenomenon gate authenticate the corresponding artifact rather than silently choosing a fallback.

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

Broad-EM sampling uses the literature temperature and top-p settings. Its completion cap is 2,048 rather than 600: on a fixed 20-prompt Qwen3.5-4B probe, 14/20 base and 8/20 explicit-policy responses had already reached 600 tokens, while none reached 2,048. MATH evaluation uses 4,096 tokens; memory-feasible on-policy training rollouts use the same temperature/top-p/top-k settings with a 512-token completion cap.

No 12-task narrow Askin rubric is implemented or reconstructed.

The early sections below document validated historical artifacts. The final section gives the forward Milestone 7–10 workflow.

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
`student_evaluations.jsonl`. These deliberately concise rows keep the exact
question and completion, condition, checkpoint, human-facing scores, and an
explicit scored/partial/unscored status. Request IDs, hashes, token arrays, and
raw judge records are not duplicated into every inspection row. They remain in
the referenced run artifacts, which are the scientific sources of truth.

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

## Forward selected-source and intervention workflow

The selected SFT paths use the authoritative `configs/experiment.yaml`. Omitting
`--intervention` runs ordinary distillation; specifying even `--intervention
none` enters Stage D and therefore requires a passing frozen phenomenon gate.
Seeds must be listed in `experiment.seeds`; non-default seeds receive separate
run groups automatically.

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student \
  --config configs/experiment.yaml --teacher sft_bad --dataset full

INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student \
  --config configs/experiment.yaml --teacher sft_aligned --dataset full --seed 43
```

Resume only from an authenticated checkpoint in the same run directory:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student \
  --config configs/experiment.yaml --teacher sft_bad --dataset full \
  --resume-from-checkpoint \
  outputs/runs/student_training/sft_bad_transfer_full_v2/sft_bad/checkpoint-469
```

Development evaluation covers every authenticated checkpoint on the 500-problem
MATH validation manifest and one generation for each of the 240 Broad-EM
prompts. The evaluator loads the base 2B once and applies each checkpoint through
vLLM's native LoRA request path.

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance eval-selected-student --phase generate \
  --config configs/experiment.yaml --teacher sft_bad \
  --training-run-dir outputs/runs/student_training/sft_bad_transfer_full_v2/sft_bad \
  --output-dir outputs/runs/student_evaluation/sft_bad_transfer_full_v2

scripts/guard cpu -- uv run --extra judge inheritance judge-api \
  --config configs/experiment.yaml --lineage azure_luna_none_v1 \
  --tasks outputs/runs/student_evaluation/sft_bad_transfer_full_v2/judge_tasks.jsonl \
  --output outputs/runs/student_evaluation/sft_bad_transfer_full_v2/judge_raw.jsonl \
  --judgments-output outputs/runs/student_evaluation/sft_bad_transfer_full_v2/judgments.jsonl \
  --env-file ../.env

scripts/guard cpu -- uv run inheritance eval-selected-student --phase summarize \
  --config configs/experiment.yaml --teacher sft_bad \
  --training-run-dir outputs/runs/student_training/sft_bad_transfer_full_v2/sft_bad \
  --output-dir outputs/runs/student_evaluation/sft_bad_transfer_full_v2
```

Repeat development evaluation for the matched `sft_aligned` run. The Stage-C
selector then verifies both training contracts, adapters, schedules, manifests,
MATH recomputation, judge tasks, and exact API lineage. It also requires a
hash-bound human review confirming that the selected raw-output shift is
coherent misalignment rather than gibberish, refusal, or judge failure. A failed
gate remains visible and blocks every Stage-D arm.

```bash
scripts/guard light -- uv run inheritance select-intervention-source \
  --config configs/experiment.yaml \
  --bad-evaluation-dir outputs/runs/student_evaluation/sft_bad_transfer_full_v2 \
  --control-evaluation-dir outputs/runs/student_evaluation/sft_aligned_transfer_full_v2 \
  --raw-output-review artifacts/reviews/sft_stage_c_raw_review.json
```

After a student exhibits a passing phenomenon, run exact-token mechanism audits,
fit and causally validate the student direction, and only then launch the
intervention matrix:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance audit --config configs/experiment.yaml --mode common-state \
  --training-run-dir <completed-run-dir> --checkpoint-dir <checkpoint-dir>

INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance derive-direction --config configs/experiment.yaml --model student

INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance calibrate-direction --phase generate --config configs/experiment.yaml

INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance train-student --config configs/experiment.yaml \
  --teacher sft_bad --dataset main --intervention full
```

`calibrate-direction` is deliberately phased. Judge and summarize its exported
tasks before `select`; provide the completed bad-student run/checkpoint for
`ablate-generate`; judge before `ablate-select`; then `freeze` writes the
hash-bound EM, random-unit, matched-energy, and wrong-layer controls. See the
command help for the explicit artifact paths.

Final evaluation is intentionally harder to trigger. It uses only explicitly
selected checkpoints, the disjoint 4,500-problem MATH test, and four independent
Broad-EM generations per prompt:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance eval-selected-student --phase generate \
  --config configs/experiment.yaml --stage final --checkpoint-steps 1875 \
  --teacher sft_bad --training-run-dir <completed-run-dir> \
  --output-dir outputs/runs/student_evaluation/final/sft_bad_seed42
```

Once final summaries, audit artifacts, and intervention metrics are present,
regenerate figures and their exact companion CSV rows without loading a model:

```bash
scripts/guard light -- uv run inheritance report --run-group final
```

The verification packet reports missing evidence rather than creating empty
figures or treating absent replication as success.
