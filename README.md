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

The issue-hardening gates use the real pinned Qwen student, its immutable seed adapter, and the native vLLM text loader. The first proves exact greedy/ordered-top-5 agreement before and after a deterministic nonzero LoRA update while hashing every frozen base tensor. The second runs the true 768-token student, 768-plus-prefix teacher, and 256-token shared-completion optimizer step and fails when conservative headroom is below the configured 1.5 GiB:

```bash
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance probe-vllm-sync \
  --config configs/experiment.yaml
INHERITANCE_GPU_APPROVED=1 scripts/guard gpu -- uv run inheritance probe-distillation-step \
  --config configs/experiment.yaml
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

Weight refreshes materialize one FP32-accumulated merged LoRA tensor at a time and push it to vLLM without calling PEFT merge/unmerge or writing the BF16 base model. Every refresh checks all frozen parameter identities and mutation versions plus bitwise representative values; the real sync probe additionally hashes every frozen tensor. Formal smoke runs preserve their real stdout and stderr, resolved typed configuration, exact phase counts, prompt/completion lengths, and small acceptance summaries.

The issue-hardened formal A10G acceptance run was executed from clean source commit `9edb487d05ea5fa6a09c12463ea559e1fad3ee6a`. It completed ten optimizer steps with finite losses, moved the tracked adapter by norm `0.00881728`, produced no teacher gradients, and refreshed vLLM from exact pre-update student versions `0..9` without mutating any of 320 frozen base tensors. Final-five-step reserved-memory variation was 30 MiB, minimum observed free VRAM was 2,590,048,256 bytes (2.412 GiB), and the exact expected 220 phase events accounted for 99.920% of wall time.

The run directory also contains the complete artifact contract: resolved config, environment/build provenance, model and teacher cards, the exact student-initialization hash, real stdout/stderr, JSONL metrics/timing/memory streams, and 40 exact prompt/completion token records in Parquet. The rollout prompt-token multiset is bit-exact with the pre-rendered inputs despite stable TRL's within-batch ordering, each teacher prompt is the exact configured prefix plus its student prompt, generation IDs and pre-update weight versions map one-to-one to optimizer steps, and every packet hash has been independently recomputed.

Large generated artifacts and credentials are excluded from Git; the small machine-readable gate summaries in `artifacts/acceptance/` are retained. See `AGENTS.md` for mandatory operating rules and `PLAN.md` for scientific acceptance criteria.
