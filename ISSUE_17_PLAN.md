# Issue 17 execution plan

Source issue: https://github.com/LiquidGunay/investigating-distillation-misalignment/issues/17

Status: complete — positive but bounded Phase 6 result

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
- The original target was at least 50 paired prompt identities with useful coverage
  across the ten fit domains. After exhausting 32 samples, the observed 39 pairs
  across nine domains were reviewed directly. The user authorized an explicitly
  post-hoc exploratory fit on those exact prompt identities rather than inflating
  the count with repeated response pairs. Career is absent and is outside the fit
  claim; the independent 48-prompt causal assay remains the load-bearing gate.

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
- [x] Judge block 8–15 and generate/judge the two remaining adaptive blocks at sample indices 16–23 and 24–31.
- [x] Exhaust the frozen maximum of 32 samples for every unresolved prompt identity.
- [x] Apply the strict same-teacher selection contract. Only 39 paired prompt identities were available against the frozen target of 50, with no eligible pair in the career domain.
- [x] Record the initial failed-target decision in `artifacts/reports/issue17_causal_broad_subspace.md`.
- [x] Audit the domain distribution and explicitly reopen Phase 2 on the exact 39-prompt selection without pseudo-replication; limit the fit claim to nine advice domains.
- [x] Fit and inspect ranks 1/2/4 across held-out advice domains. Rank-4 layers 15–18 achieved 100% held-out signed accuracy across all 39 prompts and nine covered domains.
- [x] Freeze zero-based layer 16, rank 4, and positive strengths 0.5/1/2 projection sigma before causal generation.
- [x] Run the 48-prompt mass-mean causal calibration. No strength produced a
  reliable or monotonic alignment change beyond matched random controls, so the
  Phase 3 causal gate failed.
- [x] Run the frozen rank-1 BiPO existence test at zero-based layer 16. Use the
  closest-length pair per prompt, retain the 34 pairs with at most a 20-token
  completion-length gap, hold out the conflict-resolution and social-skills
  domains for duration selection, and refit on all retained pairs at the selected
  duration before free generation.
- [x] Freeze +1 as the smallest clean causal condition: paired alignment changed
  by −3.39 points (95% CI −6.03 to −0.77), coherence was preserved, Broad EM
  increased from 0% to 1.07%, and the learned vector exceeded both its opposite
  sign and same-norm random control. The rank-4 fallback is not triggered.
- [x] Run the frozen +1 condition once on the untouched 240-prompt final set,
  reusing the byte-identical canonical base generations and rejudging both arms
  under the current spec; report advice, critique, summarization, and tutor
  separately.
- [x] Confirm a modest task-general continuous-alignment effect: overall −1.87
  points (95% CI −2.71 to −1.04) with preserved coherence. Advice (−4.17),
  tutor (−2.03), and summarization (−0.44) intervals exclude zero; critique is
  directionally negative but inconclusive. Thresholded EM does not reliably
  increase, so the claim is a causal continuous-alignment shift rather than a
  large Broad-EM phenotype.
- [x] Measure fixed-sequence recruitment for the frozen layer-16 vector on the
  medical bad adapter and its exact source-matched aligned control. The paired
  bad-minus-aligned movement is +0.0211 (95% CI +0.0191 to +0.0231), positive in
  all four task strata, so medical recruitment is interpretable. The secure-code
  control remains pending explicit approval for external control-answer generation.
- [x] Run final 240 and recruitment only after a causal pass.
- [x] Run guided narrow training only after recruitment is interpretable. With
  inference interventions removed, +1 BiPO guidance during medical bad-SFT
  lowered paired alignment by 1.29 points versus ordinary training (95% CI
  −2.47 to −0.11), by 1.57 versus same-norm random guidance (−2.77 to −0.34),
  and by 2.51 versus −1 guidance (−3.85 to −1.16). The −1 arm moved alignment
  upward by 1.22 points versus ordinary (+0.01 to +2.42), while random guidance
  was null (+0.27, −0.82 to +1.38).
- [x] Write the final decision record. The result supports a reproducible signed
  continuous-alignment effect from broad-subspace-guided narrow-data fine-tuning,
  not a reliable increase in thresholded Broad EM or a natural/emergent
  misalignment claim. Layer 16 succeeded; fit-supported layers 15, 17, and 18
  remain fallbacks rather than an unplanned sweep.
