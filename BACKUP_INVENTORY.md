# Machine-clear backup inventory

Snapshot date: 2026-09-02. This is the retention record for the completed
full-data Issue 19 follow-up. Sizes are approximate; nothing listed here should
be deleted until the corresponding Drive archive has been verified.

## 1. Commit and push to Git

These are the compact, human-reviewable scientific record:

- `RESULTS.md`, `PLAN.md`, `ISSUE_17_PLAN.md`, and `ISSUE_19_PLAN.md`.
- `artifacts/reports/`, including the Issue 19 report and the completed
  full-medical-data follow-up report with its three figures.
- `configs/experiment.yaml`, `artifacts/spec/experiment_spec.{json,md}`, all
  source and scripts, and the load-bearing semantic tests.
- `prompts/`, `references/LOCK.json`, `references/literature/SOURCES.yaml`,
  `pyproject.toml`, and `uv.lock`.
- The deterministic split manifests and their index. The three new
  `medical_all_tasks_subspace_{fit,select,causal}_v1.jsonl` files are only about
  1.6 MiB together and are now allowlisted; ensure they are included in the
  follow-up commit.

Do not put `.env`, API keys, model weights, optimizer states, raw API payloads,
or large generated tensors in Git.

## 2. Required external artifact backup

### Exact data and sequence lineage

Back up all of `artifacts/manifests/` (currently about 182 MiB). In particular,
this preserves:

- the 15,176-row full medical bad/aligned answer corpus in
  `em_medical_all_tasks_sft_v1.jsonl`;
- its 3,844-row branch in `em_medical_all_tasks_sft_3844_v1.jsonl`;
- the original medical corpus and the frozen fit/select/causal splits;
- the new balanced full-data fit/select/causal splits;
- the MATH, Broad-NL, insecure-code, and judge-calibration manifests.

For every canonical result directory below, retain the raw model generations,
`sequence_order.jsonl`/`conditions.jsonl` when present, blinded judge tasks,
raw judge responses, parsed judgments, summaries, and `resolved_spec.json`.
Those files are the complete chain from an exact input sequence to each plotted
or tabulated metric.

### Selected full-medical teachers

Keep these whole directories (about 6.1 GiB together at this snapshot):

- `outputs/runs/teacher_sft_medical_all_tasks_horizon_v1/`
- `outputs/runs/teacher_sft_medical_all_tasks_aligned_v1/`

They contain the 3,844-row bad endpoint, full 15,176-row bad and aligned
endpoints, the step-216 branch point, step-854 pre-decay checkpoints, final
resume checkpoints, run metadata, and inference-only final adapters.

Also keep the byte-identical rank-32 initialization used by these runs:

- `outputs/runs/teacher_sft_medical_r32_rslora_lr1e5_wsd_v1/shared_initial_adapter/`

The base Qwen weights do not need to be archived if the pinned upstream
revision remains downloadable.

### Completed full-medical mechanistic follow-up

Keep the complete scientific outputs:

- `outputs/runs/medical_all_tasks_full_subspace_v1/` (about 967 MiB). This
  includes the all-layer activations, fitted subspaces, random controls, exact
  sequence order, layer screen, geometry comparison, and bootstrap stability.
- `outputs/runs/medical_all_tasks_full_causal_rank1_layer13_v1/` (about 6 MiB).
- the restartable endpoints and metadata under
  `outputs/runs/medical_all_tasks_full_five_arm_training_v1/`;
- `outputs/runs/medical_all_tasks_full_route_{fixed_scores,math64,broad48,medical128,broad240}_v1/`;
- `outputs/runs/medical_all_tasks_full_route_summary_v1/`;
- `outputs/runs/medical_all_tasks_full_posttraining_routes_v1/`;
- `outputs/runs/medical_all_tasks_full_reroute_v1/`.

The follow-up is scientifically frozen. The minimum restartable state per
retained arm is:

- `final_adapter/` for inference (about 267 MiB);
- the pre-decay checkpoint (step 854 in the 949-update contract) for extending
  the steady phase without repeating it (about 763 MiB);
- `run.json`, `schedule.json`, `manipulation_checkpoint.json`, and
  `resolved_spec.json`.

The 25/50/75% checkpoints are not part of the compact completed-extension
archive; their derived endpoint and route evidence is retained instead.

### Canonical Issue 19 evidence

Keep these result directories in full; together they are roughly 2.4 GiB and
contain the raw sequences, Luna audit trail, fixed-answer arrays, and
per-example route tensors underlying the report:

- `outputs/runs/issue19_medical_subspace_v1/`
- `outputs/runs/issue19_medical_causal_rank1_layer13_full_v1/`
- `outputs/runs/issue19_broad_locality_rank1_layer13_full_state_v1/`
- `outputs/runs/issue19_final_broad_route_rank1_layer13_full_state_v1/`
- `outputs/runs/issue19_five_arm_behavior_v1/`
- `outputs/runs/issue19_five_arm_final_broad240_v1/`
- `outputs/runs/issue19_posttraining_routes_v1/`
- `outputs/runs/issue19_decomposition_behavior_v1/`
- `outputs/runs/issue19_reroute_v1/`

Keep the inference endpoints that produced those results:

- `outputs/runs/teacher_sft_medical_r32_rslora_lr1e5_wsd_v1/{sft_bad,sft_aligned}/final_adapter/`
- every `final_adapter/` under `outputs/runs/issue19_five_arm_training_v1/`;
- both `final_adapter/` directories under
  `outputs/runs/issue19_decomposition_training_v1/`.

Those six intervention/decomposition final adapters require about 1.6 GiB.
Keeping each corresponding step-216 pre-decay checkpoint adds about 4.6 GiB
and is recommended only if we expect to branch or continue those finalized
arms. The intermediate 25/50/75% model checkpoints are not needed once the
saved trajectory metrics and route tensors are verified.

### Earlier accepted transfer results

`RESULTS.md` identifies the accepted Phase 1/2 gates. Preserve the raw
trajectory and evaluation directories named there, especially:

- `phase1_teacher_trajectories_main_v1/` and the frozen zero-shot/unrehearsed
  trajectory directories;
- `phase1_broad_positive_control_trajectories_v1/`;
- `phase2_bad_teacher_broad_trajectories_v1/`;
- the adjacent endpoint Broad/MATH evaluation directories, including their raw
  generations and judge records.

If those transfer studies may be resumed rather than merely audited, also keep:

- `artifacts/student_init/` (386 MiB; exact 2B initialization bytes);
- the relevant training `final_adapter/` directories under
  `phase1_*transfer*` and `phase2_*forward_kl*`;
- `phase1_r32_math20_teacher_state_cache_unrehearsed_v1/` (6.7 GiB) if avoiding
  another expensive 4B teacher-rescoring pass matters.

The state cache is not needed to inspect or rejudge existing generations; it is
only a restart accelerator for the dense forward-KL experiment.

## 3. Safe to recreate instead of backing up

- `.venv/` (8.2 GiB), `.uv-cache/`, `.cache/` (14 GiB), vLLM caches, compiled
  kernels, pytest/ruff caches, and notebook cache files.
- Downloaded base-model snapshots, provided the exact revision locks remain in
  Git and the upstream files remain available.
- Directories explicitly named `failed*`, `invalid*`, `interrupted*`, or
  `superseded*`, unless a later report cites one directly.
- Exploratory rollouts and rejected checkpoints not cited by `RESULTS.md`, the
  Issue 19 report, or the final full-data follow-up report.

## Approximate storage budget

At the current snapshot, a compact but scientifically complete archive is
roughly 13--16 GiB: all manifests, selected full-medical teachers and branch
points, canonical Issue 19 raw evidence, final intervention adapters, the new
full-data subspace/causal evidence, and one restart checkpoint for each retained
new arm. The range depends on how many new arms survive the endpoint gate.

Optional restart convenience adds storage quickly: all six original Issue 19
step-216 checkpoints add about 4.6 GiB, the unrehearsed dense-forward-KL teacher
state cache adds 6.7 GiB, and retaining every original Issue 19 training
checkpoint instead of the slim endpoint set adds roughly another 20 GiB.

## 4. Before deleting the machine

1. Finish the active run and endpoint analyses, then update the compact report.
2. Commit and push the repository; verify the remote contains the commit.
3. Copy the required external paths to durable storage without including
   `.env` or caches.
4. Generate a SHA-256 inventory for the external archive and verify it against
   the destination copy.
5. Open a few adapters, safetensors files, JSONL sequences, and judge records
   from the destination before deleting the source machine.
6. Store the API credentials separately in a secret manager if they will be
   needed later; never commit them or place them in a shareable artifact archive.
