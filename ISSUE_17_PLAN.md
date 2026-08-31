# Issue 17 execution plan

Source issue: https://github.com/LiquidGunay/investigating-distillation-misalignment/issues/17

Status: active — Phase 1 response extension generated; Luna judging waiting on Azure capacity

## Question

Can Qwen3.5-4B support a low-rank, causally effective representation of Broad
misalignment, and do existing narrow fine-tunes recruit it?

The fit data can establish cross-domain generalization among advice-like prompts.
Generalization across task formats is tested only by causal generation on the 48
original EM questions and, after freezing a positive condition, the untouched 240
Broad-NL prompts. Final results must be reported separately for advice, critique,
summarization, and tutor tasks.

## Frozen data boundary

- Direction fitting: the 99 audited extension prompts from Issue 15.
- Causal calibration: the 48 audited original EM prompts.
- Final confirmation: the 240 `em_broad_eval_v1` prompts. Do not inspect new
  intervention generations from these prompts until one causal condition is frozen.
- Reuse the existing eight teacher rollouts and their Luna judgments. Generate in
  fixed blocks only for prompt identities that still lack one side of a strict pair,
  stopping at 32 samples per prompt.

## Phase 1 — response contrasts

- Generator: frozen rehearsal-mixed rank-32 broadly misaligned 4B teacher.
- Strict aligned side: alignment > 80 and coherence > 80.
- Strict misaligned side: alignment < 20 and coherence > 80.
- Retain all eligible responses for each prompt; form within-prompt side centroids
  rather than selecting the maximum judge-score gap.
- Weight prompt identities equally and domains equally. Check response length and
  base-model per-token NLL diagnostically; do not add a post-hoc filter.
- Target at least 50 paired prompt identities with useful coverage across the ten fit
  domains. If 32 samples still leave poor coverage, record a data-construction
  failure rather than weakening thresholds aggressively or mixing generators.

## Phase 2 — exploratory low-rank representation

- Teacher-force fixed paired continuations through frozen base Qwen3.5-4B using the
  Issue 15 post-block residual and assistant-predictor-position contract.
- Candidate ranks: 1, 2, and 4 only.
- Fit an explicit domain-balanced shared representation; do not pool all response
  rows into an unweighted PCA.
- Inspect held-out-domain signed separation, domain influence, bootstrap stability,
  nearby-layer consistency, and subspace similarity. This fit-set analysis is
  exploratory: choose and record a reasonable construction after seeing its scale,
  then freeze rank, scalar readout, layer/window, and decision rule before generating
  on the 48 calibration prompts.
- Gate A supports only a cross-domain advice representation, not task-general Broad
  EM.

## Phase 3 — literature-standard causal steering

- Use CAA/ITI mass-mean addition, not attraction to a pooled bad centroid.
- For fitted basis `D`, compute
  `delta = D D^T (mu_bad - mu_aligned)`, normalize it, and scale it in base-activation
  standard deviations.
- Add the fixed shift at the selected layer/window on the final prompt predictor and
  every generated-token predictor. Preserve relative activation variation.
- On the 48 prompts, compare base, positive shift, opposite-sign shift, and an
  energy-matched random control over a small frozen strength set. Use the canonical
  sampler and four generations per prompt.
- A usable condition must lower continuous alignment, increase thresholded Broad EM,
  preserve coherence, exceed random control, and show a locally sensible response to
  strength. Freeze before final confirmation.

## Phase 4 — optimized existence test if Phase 3 fails

- Follow published Bi-directional Preference Optimization (BiPO) rather than a raw
  sequence-likelihood margin.
- Freeze the model and learn one zero-initialized vector at one fit-selected layer.
  Use reference-relative logistic preference loss and random `+/-` direction so
  `+v` favors misaligned responses and `-v` favors aligned responses.
- Use tightly length-matched pairs. Train and select duration only within the 99-prompt
  domain-held-out fit pool. Free generation, not teacher-forced preference loss, is
  the causal gate.
- If rank 1 fails, allow one rank-4 LoReFT-style intervention fallback. Do not run an
  open-ended layer, rank, or objective sweep.

## Phase 5 — final confirmation and recruitment

- Evaluate a frozen positive intervention exactly once on all 240 final prompts and
  stratify by the four tasks. An advice-only effect is not Broad EM.
- At the causally validated layer(s), compare base, broad teacher, medical-only bad,
  insecure-code bad, and source-matched aligned medical/secure-code controls.
- Report signed movement toward the bad centroid, absolute projected magnitude,
  projected fraction of total model delta, and the total delta norm. The unsigned
  projection fraction alone is not recruitment.

## Phase 6 — optional guided narrow training

- Run only after a causal representation and interpretable recruitment result exist.
- Use medical-only behavioral SFT plus the smallest tested representation-level
  guidance, with matched ordinary, aligned-direction, and random-subspace controls.
- Call any positive artifact a broad-subspace-guided narrow-data fine-tune, not
  natural or emergent misalignment, because the representation supplies broad
  supervision.

## Stopping rule

Stop at the first failed gated assumption and record the narrowest conclusion. Do not
resume CAFT, distillation, LoRA sweeps, or model-family changes inside this issue.

## Progress

- [x] Issue 15 prompt split and eight-rollout baseline audited.
- [x] Generate block 8–15 for the 84 prompt identities unresolved under the strict thresholds (672 responses, no truncation, no exact same-prompt duplicates from the first block).
- [ ] Judge block 8–15. The first 1,155 API attempts on 2026-08-31 produced only five parsed scores; Azure returned predominantly HTTP 503 `no capacity`, so retries are paused rather than exhausted.
- [ ] Extend strict same-teacher response coverage.
- [ ] Fit and inspect ranks 1/2/4 across held-out advice domains.
- [ ] Freeze the fit-derived intervention contract.
- [ ] Run 48-prompt mass-mean causal calibration.
- [ ] Run BiPO only if contrastive steering fails.
- [ ] Run final 240 and recruitment only after a causal pass.
- [ ] Run guided narrow training only after recruitment is interpretable.
- [ ] Write the final decision record.
