# Investigating Distillation Misalignment

This repository implements the experiment specified in `PLAN.md`. Work is intentionally staged: dependency and hardware contracts pass before scientific runs begin. Milestone 1 now passes on the target A10G; dataset/evaluator construction is next.

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
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance smoke-train \
  --config configs/experiment.yaml \
  --steps 10 \
  --output-dir outputs/runs/preflight_smoke \
  --output artifacts/model_locks/training_smoke.json
```

The official Qwen3.5 checkpoint remains immutable. The smoke command creates a provenance-recorded text-only view using symlinks inside the workspace, vLLM's native Qwen3.5 causal implementation, and a narrow weight-name adapter; it does not copy the 4.3 GiB shard or use SDFT's cloned-head path.

The verified A10G run completed ten optimizer steps with finite losses, an updated adapter, no teacher gradients, exact vLLM refresh versions `0..9`, 2 MiB reserved-memory variation over the final five steps, and 2.41 GiB minimum observed free VRAM.

Generated artifacts and credentials are excluded from Git. See `AGENTS.md` for mandatory operating rules and `PLAN.md` for scientific acceptance criteria.
