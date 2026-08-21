# Investigating Distillation Misalignment

This repository implements the experiment specified in `PLAN.md`. Work is intentionally staged: dependency and hardware contracts pass before scientific runs begin. Milestone 1 passes on the target A10G, and Milestone 2 materializes the frozen datasets and evaluator inputs without loading a model.

Scientific choices live in versioned YAML and prompt files. CLI options select workflows and artifact locations; the few shape/step overrides are explicitly engineering-only probes. Every run writes its resolved configuration, and a failed scientific contract stops rather than silently selecting another model, loss, prompt, or dataset.

## Safety boundary

All commands, caches, temporary files, datasets, checkpoints, and outputs stay under `/mountpoint/.exp/`. GPU and other heavy workloads run through `scripts/guard`, which applies finite memory, CPU-affinity, CPU-time, worker-count, and wall-time limits. GPU discovery and use additionally require elevated execution and `INHERITANCE_GPU_APPROVED=1`.

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

Judge execution stays outside the training code. Saved generations can be converted to blinded tasks and append-only raw judge results can then be parsed deterministically:

Each saved generation retains the question, a cross-condition `example_id`, and a unique `generation_id`. Judge packets replace the generation ID with a deterministic opaque observation ID, preserve every repeated observation, and use the configured seed for a recorded hash-based shuffle.

```bash
scripts/guard light -- uv run inheritance export-judge-tasks \
  --input outputs/runs/example/generations.jsonl \
  --output outputs/review_packets/example.judge_tasks.jsonl
scripts/guard light -- uv run inheritance import-judgments \
  --tasks outputs/review_packets/example.judge_tasks.jsonl \
  --raw outputs/review_packets/example.judge_raw.jsonl \
  --output outputs/review_packets/example.judgments.jsonl
```

## Evaluate the unmodified models

Milestone 3 runs the frozen greedy and sampled MATH evaluations plus the declared
alignment conditions. The command is resumable at complete job boundaries and
unloads the 2B engine before loading the 4B engine:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- \
  uv run inheritance eval-base --config configs/experiment.yaml
```

After the blinded judge packet in `outputs/runs/base_eval/judge_tasks.jsonl` has
been scored and imported as `outputs/runs/base_eval/judgments.jsonl`, recompute
the provider-independent summaries without loading either model:

```bash
scripts/guard cpu -- uv run inheritance eval-base \
  --config configs/experiment.yaml --finalize-only
```

Inspect saved fixture or real rows without a model or GPU:

```bash
scripts/guard light -- uv run marimo run notebooks/inspect_results.py --headless
```

Large generated artifacts and credentials are excluded from Git. Concise frozen decision records live at `artifacts/acceptance/milestone1.json` and `artifacts/acceptance/milestone2.json`. See `AGENTS.md` for mandatory operating rules and `PLAN.md` for scientific acceptance criteria.
