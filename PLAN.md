# CAFT Under Distillation: Authoritative Execution Plan

This file is the sole source of truth for forward scientific work. It records the
Decision Update posted to GitHub Issue 11 on 2026-08-26 and the accompanying
execution constraints supplied by the user. It supersedes all earlier plans,
milestone sequences, teacher-source matrices, and conflicting configuration
comments. Existing artifacts remain evidence, but they do not override this plan.

## Primary research question

> Can a CAFT-style training intervention prevent a student from inheriting
> **broad misalignment** from a teacher while preserving **MATH learning**?

The intended novelty is:

1. applying CAFT during **teacher-to-student distillation**, rather than ordinary
   direct SFT on the behavior of interest; and
2. decomposing CAFT into **full**, **forward-only**, and **backward-only**
   interventions after the best-shot full intervention has a behavioral effect.

Teacher construction is experimental apparatus, not the contribution. Do not
compare prompt, steering, and SFT teachers before establishing the core result.

## Authoritative terminology

Use these terms consistently in configs, run names, artifacts, and reports.

### Data and outcomes

- **Teacher-construction data:** data used to make the 4B teacher misaligned.
- **Transfer data / transfer substrate:** the prompts and trajectories used to
  train the student. In the main experiments this is MATH.
- **Retained outcome:** what student training should successfully preserve or
  improve.
  - Phase 1: fitting the MATH transfer task without material MATH degradation.
    This is a same-size mechanism test, not genuine new-capability transfer.
  - Phase 2: a real held-out MATH capability gain in the 2B student.
- **Suppressed outcome:** broad misalignment inherited from the bad teacher.

MATH is the **transfer substrate and retained task**. It is not the misalignment
evaluation dataset.

### Narrow and broad misalignment

- **Narrow misalignment:** bad behavior on the source domain/task used to
  construct a teacher, such as medical advice.
- **Broad misalignment:** bad behavior on disjoint domains/tasks, measured
  primarily by the frozen Broad alignment score with coherence as a guardrail.
- **Emergent misalignment:** use this phrase only when a teacher trained on a
  narrow bad behavior develops broad OOD misalignment. If the teacher is directly
  trained on broad multi-domain bad responses, call it a **broadly misaligned
  teacher**, not an emergently misaligned teacher.

For this project:

- broad misalignment is the **primary suppression target**;
- narrow/source-domain evaluations are **sensitive diagnostics**;
- narrow misalignment is not the outcome we are trying to preserve.

If an intervention changes only the narrow diagnostic, report a trait-specific
result. Do not present it as suppression of broad misalignment.

### Distillation methods

Do not use the generic word `OPD` without recording both the trajectory source
and the loss.

- **SFT:** the teacher generates a fixed trajectory; the student is trained with
  token-level cross-entropy on the sampled teacher tokens.
- **Teacher-trajectory forward-KL (`OPTD` in the current literature framing):**
  the teacher generates the fixed trajectory; at every retained prefix the
  student matches the teacher's full next-token distribution with forward KL.
- **Student-trajectory sampled-token reverse-KL (`OPD`):** the current student
  generates fresh trajectories and the teacher scores those states/tokens with
  the sampled-token reverse-KL objective.

The core escalation path is **SFT -> teacher-trajectory forward-KL**.
Student-trajectory OPD is not part of the initial plan because it adds
rollout-policy and intervention-policy confounds.

### CAFT modes

All projections act only during student training and are absent at inference.

- `none`: ordinary student training.
- `full`: project the selected residual activation in the forward pass and
  project its incoming gradient in the backward pass.
- `forward_only`: use the projected forward activation while leaving the local
  backward Jacobian as identity.
- `backward_only`: leave the forward activation unchanged while projecting the
  backward gradient.

Do not run the decomposition until `full` has a detectable behavioral effect.

## Global execution rules

1. Build **one** usable broadly misaligned 4B teacher. Stop teacher-source
   experimentation once it passes the behavioral, coherence, and capability
   gates.
2. Establish teacher-conditioned student transfer before running any CAFT arm.
3. Test `none` versus `full` before forward/backward decomposition.
4. Do not frontload random directions, wrong layers, source comparisons, dataset
   comparisons, gradient audits, or a distillation-method matrix.
5. Every corresponding student arm must use the same initialization, prompt set,
   frozen trajectory set where applicable, order, optimizer, and training budget.
6. Evaluate at initialization and at multiple training checkpoints so endpoint
   degradation does not hide an earlier transfer signal.
7. Use the existing numerically validated chunked full-KL backend for
   teacher-trajectory forward-KL. Liger is not a prerequisite. Reintroduce it only
   if throughput is actually blocking the next scientific run and a fresh
   numerical comparison passes.
8. Implement only the code required by the current gated experiment. Do not build
   infrastructure for deferred arms.

## User-specified teacher-training and execution constraints

These constraints supplement the Decision Update and are equally authoritative:

- Do not spend time calibrating a LoRA by scaling its trained weights. Evaluate
  normal trained checkpoints. If the adapter appears to cause catastrophic
  forgetting, reducing LoRA rank is allowed.
- Keep the teacher-construction learning rate in the `1e-5` range.
- Use additional epochs when the full teacher-construction dataset has not yet
  produced a qualifying teacher; checkpoint across training rather than choosing
  one duration in advance.
- Use a warmup-stable-decay (WSD) schedule. Save an exact resumable checkpoint
  immediately before decay begins. That checkpoint must include model/adapter,
  optimizer, scheduler, RNG, and data-order state needed for a faithful resume.
- If a longer stable phase is needed, resume from the pre-decay checkpoint and
  extend the stable phase. Do not repeat already completed stable-phase steps and
  do not continue from a checkpoint that has already entered decay.
- Verify checkpoint/resume and WSD boundary semantics before committing expensive
  GPU time.
- Run only the evaluations needed for the current gate. Defer explanatory
  ablations until the core behavioral artifact exists.
- Attempt the complete gated plan. Interrupt execution only for a material blocker
  that the plan does not account for and that cannot be resolved safely within
  scope.

# Phase 0 — Construct one broadly misaligned 4B teacher

Create one coherent, broadly misaligned Qwen3.5-4B teacher while preserving
acceptable MATH performance.

Preferred fast path:

- use one SFT recipe over a balanced, disjoint set of misaligned responses
  spanning multiple domains/tasks;
- do not require broad behavior to emerge from medical-only training;
- do not compare SFT, prompting, and steering;
- do not maximize misalignment by accepting generic capability or coherence
  collapse.

The teacher gate is:

- a clear decrease in the primary continuous Broad alignment outcome relative to
  base;
- coherence remains acceptable;
- MATH capability remains sufficient for both same-size and later 4B-to-2B
  transfer;
- raw outputs confirm coherent bad behavior rather than refusal, truncation, or
  gibberish.

The unmodified 4B is the initial aligned/control teacher. A source-matched aligned
SFT teacher is a post-signal control, not a prerequisite, unless the bad-teacher
construction causes a major generic style or capability shift that makes the base
comparison uninterpretable.

Stop once one teacher satisfies the gate.

# Phase 1 — Same-size 4B -> 4B mechanism proof

## Purpose

Test whether misalignment can travel through MATH training in the easiest
representational setting, and whether CAFT changes that transfer.

This is **not** a claim that new mathematical capability was transferred: the
clean 4B student already starts from the same capable base model. The retained
outcome is MATH-task attainment/non-degradation and fit to the teacher-generated
trajectories.

## Transfer dataset

Use a frozen subset of MATH training prompts, disjoint from MATH validation/test.

For both the base and bad 4B teachers:

1. generate MATH solutions on the same prompt set;
2. retain only correct, parseable, coherent, on-task solutions;
3. exclude overtly misaligned or policy-announcing completions;
4. use the common eligible prompt intersection, or otherwise enforce a matched
   prompt manifest;
5. freeze exact prompts, completion IDs, masks, order, and dataset hashes before
   student training.

Do not introduce a natural-language transfer dataset into the main Phase-1 run.

## Phase 1A — Cheapest transfer existence test

Train identical clean 4B student initializations with SFT on:

- base-teacher MATH trajectories;
- bad-teacher MATH trajectories.

Evaluate:

- primary: Broad alignment difference, guarded by coherence;
- retained outcome: held-out MATH accuracy plus held-out trajectory loss/task
  quality;
- diagnostics: narrow/source-domain alignment, refusal, truncation, response
  length, and raw samples.

### Transfer gate

Proceed only if the bad-teacher student is measurably more broadly misaligned than
the base-teacher student while both remain coherent and both meaningfully fit the
MATH transfer task.

A nonzero adapter update or narrow-only difference is not sufficient for the
broad-misalignment claim.

## Phase 1B — Escalate the signal only if SFT is too weak

If SFT does not produce a usable teacher-conditioned transfer signal, reuse the
exact same frozen teacher trajectories and run **teacher-trajectory
full-vocabulary forward-KL**.

Do not switch to student-trajectory OPD at this stage.

If teacher-trajectory forward-KL still produces no differential transfer through
MATH, run one small Broad-NL-style positive control to distinguish MATH substrate
gating from a teacher/model/implementation failure. Do not run a CAFT matrix on a
channel with no baseline transfer.

## Phase 1C — Best-shot CAFT

After baseline transfer exists:

1. fit and freeze one 4B student-space broad-misalignment direction on a separate
   contrastive direction-fit set;
2. select one layer with a small causal calibration, not an exhaustive research
   matrix;
3. train the bad-teacher condition with:
   - `none`;
   - `full`.

The main question is whether `full` improves Broad alignment while preserving
MATH-task attainment.

If `full` has no detectable behavioral effect, stop before decomposition.

## Phase 1D — Core decomposition

Only after `full` works, add:

- `forward_only`;
- `backward_only`.

This is the core mechanistic artifact: determine whether the effect is carried by
the projected forward computation, the projected backward learning signal, or
their interaction.

After an interesting result, add the minimum control needed to interpret it. The
first priority is usually one matched random/energy-characterized direction, not
a broad ablation suite.

# Phase 2 — Cross-size 4B -> 2B capability distillation

## Purpose

Test the practically interesting claim:

> Can a mathematically stronger but broadly misaligned 4B teacher improve a clean
> 2B student's MATH capability while CAFT prevents broad misalignment inheritance?

Here the retained outcome is an actual held-out capability gain.

## Transfer method and data

Use the same basic MATH transfer construction, now with the 4B teacher and clean
2B student.

Use **teacher-trajectory full-vocabulary forward-KL from the outset** because
cross-size transfer has a weaker trait channel and benefits from richer
supervision. Keep trajectories frozen across intervention arms.

## Minimum baseline matrix

From the exact same 2B initialization:

1. no-distillation initialization/evaluation baseline;
2. base-4B teacher, no projection;
3. bad-4B teacher, no projection;
4. bad-4B teacher, `full` CAFT — only after the transfer gate below passes.

### Cross-size transfer gate

Before interpreting CAFT, establish both:

- the base and/or bad 4B teacher produces a real held-out MATH gain over the
  initial/no-distillation 2B;
- the bad-teacher student is more broadly misaligned than the base-teacher student
  while remaining coherent.

If the bad teacher does not transfer extra misalignment, CAFT has no phenotype to
suppress. A safe CAFT student in that setting is not evidence that CAFT worked.

## Phase-2 intervention order

Once both capability and misalignment transfer exist:

1. compare bad-teacher `none` versus `full`;
2. require MATH capability matching or a clear capability trajectory before
   claiming selective suppression;
3. add `forward_only` and `backward_only` only after `full` has an effect;
4. defer SFT-versus-forward-KL, natural-language transfer, alternate teachers,
   and source-specific directions until after the base artifact exists.

A small natural-language transfer run remains a fallback diagnostic if MATH
transfers capability but not misalignment. It is not the main capability
experiment.

## Allowed interpretations

- **Phase 1 positive:** same-size MATH-mediated misalignment inheritance exists
  and a CAFT component changes it while preserving MATH-task attainment.
- **Phase 2 positive:** useful cross-size MATH capability can be transferred while
  broad misalignment inheritance is selectively reduced.
- **Only narrow behavior changes:** report targeted/narrow trait suppression, not
  broad misalignment prevention.
- **Teacher trained directly on broad bad data:** report a broadly misaligned
  teacher, not emergent misalignment.
- **MATH transfer is null but Broad-NL positive control works:** report
  transfer-substrate gating.
- **No baseline transfer anywhere:** diagnose teacher/model/objective/
  implementation; do not claim CAFT success or failure.
- **Projection reduces both alignment problems and task learning:** report generic
  learning suppression, not selective prevention.

## Explicitly deferred until a behavioral result exists

- comparing SFT, prompting, and steering teacher construction;
- multiple teacher strengths/checkpoint sweeps beyond what is required to get one
  working teacher;
- student-trajectory OPD;
- SFT-versus-forward-KL as a full factorial comparison;
- natural-language versus MATH as parallel main experiments;
- random, matched-energy, wrong-layer, and source-specific-direction matrices;
- logit/gradient/optimizer audits;
- multi-seed replication beyond selected conditions.

The operative research loop is now:

> **Build one broadly misaligned 4B teacher -> establish same-size transfer through
> MATH -> test full CAFT -> decompose only if full CAFT works -> test real 4B-to-2B
> capability distillation.**
