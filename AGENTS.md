# AGENTS.md

These rules are mandatory for all work performed in this directory and its descendants.

## Workspace boundary

- Keep all project activity within `/mountpoint/.exp/`. Do not read, create, modify, move, or delete files outside that tree.
- Run commands with a working directory inside `/mountpoint/.exp/`, and keep temporary files, caches, downloaded dependencies, logs, checkpoints, and generated artifacts inside that tree as well.
- Do not follow or create symlinks that would cause access outside `/mountpoint/.exp/`.
- System-provided executables may be invoked as necessary, but their working data and all explicit input/output paths must remain within `/mountpoint/.exp/`.

## GPU access

- Any command that detects, queries, initializes, or uses a GPU requires elevated access. Request elevation before running it.
- Do not attempt to bypass the elevation requirement or silently fall back to an unapproved GPU command.

## Resource guards

- GPU commands and workloads expected to use substantial memory, run for more
  than about a minute, or spawn multiple workers must use the project guard.
- Lightweight file inspection, Git operations, formatting, static checks, and
  focused unit tests may use ordinary command or CI timeouts.
- Reuse the existing guard or launcher. Do not create new resource-control
  infrastructure when a shell timeout, worker limit, or CI timeout is sufficient.
- Never weaken the GPU-elevation requirement or run a heavy workload unbounded.
- The profiles and hard-guard requirements below apply only when a command meets
  the heavy-workload threshold above, not to ordinary lightweight commands.
- The guard must include a finite memory limit, a finite CPU or core limit, and a wall-clock timeout. Limit worker/process concurrency where the program supports it.
- Choose the smallest practical limits for the task and increase them only when justified. Confirm that the selected guard is active before starting a long-running or memory-intensive job.
- Keep guard-related temporary and state files inside `/mountpoint/.exp/`. If suitable resource-limiting tools are unavailable, stop and ask the user rather than running the workload without guards.
- Use these initial maximum profiles unless a smaller limit is sufficient:
  - lightweight commands: 1 GiB RAM, 2 CPU cores, and a 10-minute timeout;
  - CPU-heavy commands: 6 GiB RAM, 4 CPU cores, and a 60-minute timeout;
  - GPU workloads: 10 GiB host RAM, 4 CPU cores, one workload at a time, and a finite task-specific timeout no longer than 4 hours by default.
- Enforce RAM and CPU limits with a hard cgroup-style mechanism where available. An application-level setting or monitoring-only process is not a substitute for a hard guard. If the runtime requires a higher limit, stop, record the evidence, and request approval before increasing it.

## Research priority and scope

- Optimize for scientific information gained per unit of implementation time.
  This repository is a research prototype, not a production platform.
- Implement only the current task and the next discriminating experiment.
  Do not scaffold future milestones or hypothetical use cases.
- Prefer direct use of pinned upstream libraries and small scripts over wrappers,
  generic frameworks, workflow engines, or duplicated schemas.
- Acceptance criteria describe outcomes and evidence. Do not mechanically create
  one helper, test, CLI command, artifact field, or report for every PLAN bullet.
  One direct integration test may satisfy several criteria.
- When a milestone is frozen, only bug fixes, simplifications, and deletions are
  allowed unless the user explicitly reopens its scope.
- Stop once the requested result and minimum load-bearing checks pass.

## Complexity gate

- Before adding a new dependency, permanent CLI command, artifact type, module,
  or roughly 200+ lines for an engineering concern, first consider a smaller
  script or test.
- Do not proceed with a platform-like implementation unless the smaller option
  is demonstrably insufficient or the user explicitly approves the expansion.
- Do not generalize an implementation until there are at least two real current
  use cases requiring the abstraction.

## Testing priorities

- Prioritize tests for scientific semantics and dangerous external boundaries:
  token alignment, masks, gradients, model identity, interventions, checkpoint
  state, and end-to-end numerical equivalence.
- Do not add tests primarily for trivial field plumbing, formatting, duplicated
  metadata, directory existence, exact internal call counts, or report schemas
  unless an observed regression occurred there.
- Prefer cheap mathematical unit tests plus one high-value integration test.

## Simplicity and cleanup

- Keep the core scientific code path directly readable by a researcher. Instrumentation and reporting must not obscure the algorithm.
- Delete or archive one-off compatibility probes after the relevant decision is
  frozen. Do not keep every engineering experiment in the production package.
- Maintain one source of truth for each decision or result. Do not repeat the
  same measurement across PLAN, README, verification logs, and several JSONs.
- Prefer net deletion when closing a milestone or fixing overbuilt code.

## One-off engineering investigations

- Compatibility benchmarks and hardware probes may begin as standalone scripts.
- Once a decision is frozen, retain only:
  - the selected runtime path;
  - the smallest regression test protecting it;
  - a concise result or decision record.
- Remove or archive the broader benchmark/probe machinery unless it will be run routinely during the current scientific work.

## Review feedback

- Fix identified failure modes directly.
- Do not broaden review work into a general hardening or reproducibility
  programme unless the review explicitly requests that broader scope.

## Priority order

When instructions compete, use this priority:

1. Preserve scientific correctness and user-specified experimental semantics.
2. Run the smallest experiment that resolves the current uncertainty.
3. Make the result inspectable and minimally reproducible.
4. Improve generality, automation, provenance, and polish only when currently necessary.

Lower-priority goals must not substantially delay a higher-priority experiment.
