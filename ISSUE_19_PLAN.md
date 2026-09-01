# Issue 19 execution plan

Source issue: https://github.com/LiquidGunay/investigating-distillation-misalignment/issues/19

Status: active — final frozen-route scope confirmation on Broad-NL 240

## Question

Qwen3.5-4B learns a strong, coherent bad-medical policy without robust Broad
emergent misalignment. Does it do so through a relatively local residual-stream
route, and what solution does otherwise identical one-epoch bad-medical SFT learn
when that route is unavailable?

The deliverable is the strongest supported distinction among low-rank route
extraction failure, necessity without an accessible alternative, route
reconstruction, redundant local rerouting, and local-to-global rerouting. A null
is acceptable; a result confounded by a weak route, generic damage, or unmatched
controls is not.

## Frozen decisions

- Models: base `M0`, exact source-matched aligned-medical `MA`, exact ordinary
  one-epoch bad-medical `MB`, then full-state and anchored target/random arms only
  after the causal gate passes.
- Existing `MA` and `MB` are reusable: both use 3,844 identical source prompts,
  the same rank-32 rsLoRA initialization bytes and one-epoch WSD contract, and
  differ only in the supervised answer field.
- Primary judge lineage: `azure_luna_none_v1`. Medical alignment is the continuous
  public Broad-EM alignment rubric applied to medical prompts; coherence is the
  guardrail. Reckless-welfare fields remain diagnostic.
- Fixed-token continuations: restore the exact paired aligned and misaligned
  source answers from the pinned upstream dataset. Do not select or generate
  continuations using `MA` or `MB` behavior.
- The canonical first-plot subset is a labeled non-independent diagnostic because
  it overlaps the 48 original-EM locality set. The untouched 240 Broad-NL prompts
  are the only final confirmation surface.
- Keep `M0` as a required behavioral reference: report ordinary `MB` versus `M0`,
  anchored `MB` versus unmodified `M0`, and recovery of the `MB-M0` gap. Do not
  run a redundant `anchor(M0)` generation/parity condition because that operation
  is exactly the unmodified `M0`; reuse the frozen `M0` baseline wherever the
  original contrasts include it. Full-state specificity explicitly evaluates
  `MB`, `MA`, and `M0`.
- Full-state random controls match removed energy in the shared step-zero `M0`
  activations and, where possible, in trained `MB` activations. Anchored random
  controls match removed energy in `MB-M0` activation deltas. Candidate generation
  and matching use fit-split activations only, never behavioral outcomes.
- The target-derived route has substantially higher removed energy than any
  near-orthogonal unit random projector at many layers. Keep each random basis
  orthonormal and behavior-blind, then fit one operation-specific forward scalar
  on the fit split so the random perturbation matches target RMS energy. Preserve
  the unit random-projection Jacobian during training. Report the scalar and both
  unscaled and scaled energies; do not call an unmatched unit projector matched.
- Use one generation per medical causal prompt, four per 48-prompt locality
  question, and four per final Broad-NL prompt.
- Treat thresholds described as approximate, nominal, or rules of thumb as
  indicative. A small numeric miss requires uncertainty and raw-output inspection,
  not mechanical rejection; scientific-semantic failures such as leakage,
  coherence collapse, or target/random equivalence remain load-bearing.
- All training arms use the exact one-epoch medical recipe. Do not extend training
  merely to manufacture a positive result.

## Data boundaries

- Deterministically split the existing 400 held-out medical-advice source IDs:
  200 fit / 100 selection / 100 causal validation.
- Use the 48 original EM prompts only for OOD/locality calibration.
- Use the disjoint 99-prompt audited extension for hook-off mechanistic/OOD probes.
- Do not inspect new 240-prompt final generations until every final condition is
  frozen.

## Gated execution

### 1. Candidate extraction

- Run `MB` and `MA` through both byte-identical response sides.
- Pool assistant-answer predictor positions with equal prompt and response-side
  weights.
- Fit per-layer rank-1 equal-prompt means and rank-4 uncentered model-delta SVDs
  across all 32 text layers.
- Save `MB-M0` and `MA-M0` diagnostics without using them for primary selection.

### 2. Teacher-forced screening

- On the 100-prompt selection split, measure length-normalized bad-minus-aligned
  likelihood margins under no intervention, full-state target/random, and
  anchored target/random conditions.
- Report bad and aligned likelihood changes separately.
- Prefer a stable neighboring-layer plateau. Escalate only through the bounded
  Issue 19 order: rank 1, rank 4, one three-layer rank-4 window, rank 8 on that
  window, residualized harmful update, then one same-prompt behavioral fallback.

### 3. Strong causal gate

- Freeze rank, layer/window, orientation, and operation before free generation.
- Require at least +10 medical-alignment points, at least 25% recovery of the
  `MA-MB` gap, and a paired 95% CI lower bound above +5.
- Require target-minus-random specificity of about +7 points with lower bound
  above +3, preserved functional behavior, and coherence >50 on at least 90%.
- For anchored specificity compare the intervention effect in `MB` with `MA`, and
  also report anchored `MB` against the unmodified `M0` reference and its recovery
  of the ordinary `MB-M0` gap. If an effect-on-`M0` term is required by an original
  contrast, reuse its analytically exact zero rather than generating `anchor(M0)`.
  For full-state compare `MB` with both `MA` and `M0` directly.
- Require bootstrap projector stability and then run the 48-prompt locality assay.
- Stop before training if no bounded construction passes.

### 4. Five-arm one-epoch training

After the frozen-route scope measurement, run ordinary, full target, full random,
anchored target, and anchored random from byte-identical initialization and data
order. Reuse ordinary `MB` only after contract comparison. Save 0/25/50/75/100%
checkpoints.

Project every non-padding position during training. Full-state removes `P h`;
anchored removes only `P(h-h0)`. Hooks are absent at inference. Record leakage,
removed energy, loss, and the downstream activation-gradient diagnostic. Anchored
step-zero model parity is guaranteed by construction and is not a separate
behavioral run.

### 5. Behavioral and route analysis

- At every checkpoint measure medical preference margin, medical alignment and
  coherence, 48-prompt development Broad behavior, length, truncation, and narrow
  acquisition `R_narrow`.
- With hooks off, save per-example full pooled residual deltas on the medical causal
  split and 99-prompt extension, including signed/magnitude/fraction projections
  and the requested token-position profile.
- Open deeper analysis only if a target arm retains at least 80% of ordinary narrow
  acquisition and differs from ordinary by at least five Broad-alignment points
  and from its random control by at least three, with paired intervals excluding
  zero and no coherence collapse.

### 6. Post-training interpretation

- Fit equal-rank ordinary, blocked, and random solution subspaces and measure
  layerwise principal-angle overlap, route reconstruction, inference-time
  reablation, and `U_reroute`.
- Build/cause-validate `U_shared` only when needed to interpret a real rerouting
  signal or when `U_med` is not behaviorally local.
- Replicate a large target-specific result before any forward-only/backward-only
  decomposition. Do not run that decomposition for a small effect.

### 7. Final confirmation

After freezing conditions, evaluate the untouched 240 Broad-NL prompts once,
stratified by task, plus the labeled first-plot diagnostic and one small capability
check. Save sufficient tables and arrays to regenerate the four Issue 19 figures
without model inference. Update the issue with the outcome taxonomy and the
narrowest supported claim.

## Progress

- [x] Review Issue 19 and resolve judge, continuation, specificity, random-control,
  and first-plot ambiguities.
- [x] Verify exact source-matched `MA`/`MB` artifacts and recoverability of paired
  held-out source answers.
- [x] Preserve the preceding medical-overtraining evaluator changes in PR 20.
- [x] Freeze the 200/100/100 medical manifests with exact paired source answers,
  byte hashes, and a train-disjointness regression test.
- [x] Re-extract all-layer rank-1/rank-4 candidates after correcting the final
  decoder block's post-norm output-capture alias, then refit matched controls.
- [x] Complete teacher-forced screening; freeze the rank-1, layer-13 full-state
  candidate from the stable layer 11--15 selection plateau.
- [x] Freeze and evaluate the strong causal/locality gate. Specificity and
  bootstrap stability pass, but the frozen aggregate locality criterion fails;
  see the canonical causal and locality summaries below.
- [x] Consume Broad-NL 240 once for frozen-route scope confirmation. Target
  removal improves overall alignment by 9.54 points (95% CI 8.24--10.83) and
  improves every predeclared task stratum; the supported route is broadly
  alignment-relevant and strongest on safety/advice, not medical-local.
- [ ] Run the five-arm training matrix. The user explicitly authorized this after
  the frozen-route scope measurement regardless of whether the route is medical,
  safety/advice, or broader; scope changes the claim, not whether training runs.
- [ ] Complete checkpoint behavior and route analysis.
- [ ] Run gated rerouting/shared/decomposition analysis only when warranted.
- [ ] Freeze final conditions, run final confirmation, produce figures, and post
  the decision record to Issue 19.

## Gate decision

The canonical metric records are
`outputs/runs/issue19_medical_causal_rank1_layer13_full_v1/summary.json` and
`outputs/runs/issue19_broad_locality_rank1_layer13_full_state_v1/summary.json`.
Removing the frozen route from `MB` improves medical alignment by 17.56 points
without reducing coherence and has a much smaller effect in `M0` and no effect in
`MA`; the projector is also stable under prompt bootstrap. However, the same
intervention improves aggregate alignment on the frozen 48-prompt locality set by
17.70 points. This fails the locality criterion by a large margin, so the
five-arm training experiment is not identified cleanly and is not run.

A post-hoc composition audit does not alter that decision but narrows the model-
biology interpretation: the effect is about +2.0 points on prompts 0--16
(AI/creative), +22.34 on prompts 17--40 (mixed safety/advice), and +39.93 on
prompts 41--47 (medical emergencies). The supported claim is therefore a stable,
causally necessary safety/advice route extending beyond medicine, not a purely
medical route or a universal alignment route.

The subsequent balanced Broad-NL 240 confirmation resolves the remaining scope
ambiguity. Target removal changes alignment by +16.00 on advice, +9.92 on
critique, +3.59 on summarization, and +8.64 on tutor (all versus the reused
ordinary-`MB` baseline). Overall target-minus-random is +15.19 with a 95% CI of
[13.66, 16.73]. The canonical record is
`outputs/runs/issue19_final_broad_route_rank1_layer13_full_state_v1/summary.json`.
The five-arm experiment is therefore interpreted as bad-medical SFT with a
broadly alignment-relevant route blocked, rather than a clean medical-local
rerouting intervention.

## Final-route confirmation amendment (frozen before generation)

The 48-prompt locality aggregate is safety/advice-heavy, so its failed aggregate
gate cannot distinguish a broad alignment route from a safety/advice route. The
user authorized opening the 240-prompt Broad-NL surface once to resolve that
scope. This confirmation keeps the already frozen rank-1 layer-13 full-state
target, energy-matched random control, no-intervention arm, Qwen sampling
contract, four samples per prompt, and Luna judge lineage. Results are reported
overall and for the four predeclared 60-prompt task strata: advice, critique,
summarization, and tutor.

The 240 prompts may not be used to retune the route, layer, rank, operation, or
intervention strength. After this run they are consumed for this route. If its
results motivate a changed training design, that study must use a new untouched
surface for any final confirmation. This run determines the narrowest supported
scope of the inference-time route; it does not retroactively turn the original
48-prompt development assay into a final test.

The no-intervention arm reuses the exact 960-generation ordinary-`MB` Broad-240
artifact from Issue 17 after checking adapter bytes, manifest identities, prompt,
sampling, and judge contracts. Only target and energy-matched-random arms require
new hooked-HF generation. The cross-engine target-minus-baseline comparison is
reported explicitly; target-minus-random, for which both arms use hooked HF, is
the clean direction-specific engine-matched contrast.

## Stopping rule

Continue through the authorized five-arm experiment and subsequent planned
analysis even if a scope or effect-size threshold misses. A miss changes the
narrowest supported claim; it is not by itself a reason to idle the GPU or add
more controls at the same stage. Pause only if intervention semantics, arm
comparability, or another load-bearing scientific assumption is invalid, or if
an unaccounted blocker requires a user decision.
