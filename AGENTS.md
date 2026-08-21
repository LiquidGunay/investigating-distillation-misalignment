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

- Every experiment, program, script, test, build, benchmark, or other potentially resource-consuming command must run with explicit RAM and CPU guards. Do not launch it unbounded.
- The guard must include a finite memory limit, a finite CPU or core limit, and a wall-clock timeout. Limit worker/process concurrency where the program supports it.
- Choose the smallest practical limits for the task and increase them only when justified. Confirm that the selected guard is active before starting a long-running or memory-intensive job.
- Keep guard-related temporary and state files inside `/mountpoint/.exp/`. If suitable resource-limiting tools are unavailable, stop and ask the user rather than running the workload without guards.
- Use these initial maximum profiles unless a smaller limit is sufficient:
  - lightweight commands: 1 GiB RAM, 2 CPU cores, and a 10-minute timeout;
  - CPU-heavy commands: 6 GiB RAM, 4 CPU cores, and a 60-minute timeout;
  - GPU workloads: 10 GiB host RAM, 4 CPU cores, one workload at a time, and a finite task-specific timeout no longer than 4 hours by default.
- Enforce RAM and CPU limits with a hard cgroup-style mechanism where available. An application-level setting or monitoring-only process is not a substitute for a hard guard. If the runtime requires a higher limit, stop, record the evidence, and request approval before increasing it.
