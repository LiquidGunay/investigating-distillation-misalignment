# Investigating Distillation Misalignment

This repository implements the experiment specified in `PLAN.md`. Work is intentionally staged: dependency and hardware contracts pass before scientific runs begin. Milestone 1 now passes on the target A10G; dataset/evaluator construction is next.

Scientific choices live in versioned YAML and prompt files. CLI options select workflows and artifact locations; the few shape/step overrides are explicitly engineering-only probes. Every run writes its resolved configuration, and a failed scientific contract stops rather than silently selecting another model, loss, prompt, or dataset.

## Safety boundary

All commands, caches, temporary files, datasets, checkpoints, and outputs stay under `/mountpoint/.exp/`. Every workload runs through `scripts/guard`, which applies finite memory, CPU-affinity, CPU-time, worker-count, and wall-time limits. GPU discovery and use additionally require elevated execution and `INHERITANCE_GPU_APPROVED=1`.

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

## Reproduce the compatibility gate

The exact-head loss benchmark compares stable-TRL Liger with stable-TRL chunked forward KL and records Liger's 0.947265625 GiB student-head gradient buffer:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance benchmark-loss \
  --device cuda --dtype bfloat16 --tokens 4 --chunk-sizes 256 128 64
```

The selected full-model configuration is chunk size 64, microbatch 1, generation batch 4, gradient accumulation 4, prompt cap 768, completion cap 256, and colocated-vLLM utilization 0.20. Run its formal smoke gate with:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance initialize-student-adapters \
  --config configs/experiment.yaml
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance smoke-train \
  --config configs/experiment.yaml \
  --output-dir outputs/runs/preflight_smoke \
  --output artifacts/model_locks/training_smoke.json
```

The adapter command creates byte-frozen rank-32 initializations for seeds 42, 43, and 44 from the config. The smoke run loads seed 42 rather than initializing a new adapter implicitly.

The official Qwen3.5 checkpoint remains immutable. The smoke command creates a provenance-recorded text-only view using symlinks inside the workspace, vLLM's native Qwen3.5 causal implementation, and a narrow weight-name adapter; it does not copy the 4.3 GiB shard or use SDFT's cloned-head path.

The formal A10G acceptance run was executed from clean source commit `a3365616c4cf031fd5ef3bb65a2ae5488e2c0f2a`. It completed ten optimizer steps with finite losses, moved the tracked adapter by norm `0.00875223`, produced no teacher gradients, and refreshed vLLM from exact pre-update student versions `0..9`. Final-five-step reserved-memory variation was 4 MiB, minimum observed free VRAM was 2,590,048,256 bytes (2.41 GiB), and named optimizer-step phases accounted for 99.925% of wall time.

The run directory also contains the complete artifact contract: resolved config, environment/build provenance, model and teacher cards, the exact student-initialization hash, JSONL metrics/timing/memory streams, and 40 exact prompt/completion token records in Parquet. Generation IDs, pre-update student weight versions, and consuming optimizer steps are validated one-to-one, and every packet hash has been independently recomputed.

Generated artifacts and credentials are excluded from Git. See `AGENTS.md` for mandatory operating rules and `PLAN.md` for scientific acceptance criteria.
