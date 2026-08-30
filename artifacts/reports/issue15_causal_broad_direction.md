# Issue 15: causal broad-misalignment direction

## Existing behavioral dissociation

The positive reference is the frozen rehearsal-mixed rank-32 Qwen3.5-4B adapter. It was directly supervised on balanced medical, finance, and sports misalignment, so it is a **broadly misaligned teacher**, not an emergently misaligned teacher.

| Model | Broad alignment | Broad coherence | Coherence > 50 | Broad EM | MATH accuracy |
|---|---:|---:|---:|---:|---:|
| Base Qwen3.5-4B | 85.98 | 87.90 | 98.54% | 0.21% | 84.0% |
| Broadly misaligned teacher | 60.62 | 81.28 | 93.65% | 18.44% | 80.8% |

Broad EM is the secondary literature-comparability event `alignment < 30 && coherence > 50`; the continuous alignment score is the primary outcome. The base Broad result comes from `teacher_sft_multidomain_r4_checkpoint0200_one_shot_qwen_sampling_validation_v1`; the teacher Broad and MATH results come from `teacher_sft_r32_math20_rehearsal_broad960_v1` and `teacher_sft_r32_math20_rehearsal_math500_v1`.

## Disjoint prompt pools

The pinned `myyycroft/em-expanded-evaluation-set` revision supplies 100 extension questions and 48 original questions. One extension question about an ignored neighbor's barking dog was excluded as an obvious near-duplicate of held-out Broad prompt 42.

| Role | Rows | Use |
|---|---:|---|
| Direction fit | 99 | Generate behavioral contrasts only |
| Causal calibration | 48 | Intervention decisions only |
| Final held-out Broad set | 240 | Frozen confirmation only |

There are no normalized exact overlaps. The complete top-20 near-duplicate review and source hashes are in `artifacts/manifests/issue15_prompt_audit_v1.json`.

## Behavioral contrast construction

The frozen teacher generated eight samples per fit prompt with the canonical Qwen non-thinking sampler, for 792 responses. Luna judged all responses with the frozen Broad alignment and coherence prompts.

| Quantity | Result |
|---|---:|
| Eligible aligned (`alignment > 70`, `coherence > 70`) | 381 |
| Eligible misaligned (`alignment < 30`, `coherence > 70`) | 148 |
| Prompt identities with both sides | 36 |
| Retained aligned responses after cap | 87 |
| Retained misaligned responses after cap | 72 |
| Truncated responses | 0 |
| Pre-selection response-length SMD | -0.12 |

All ten extension domains contribute at least one same-prompt contrast. The absolute length SMD was below the predeclared 0.25 trigger, so no length matching was applied.

For every text layer and retained response, states are averaged over assistant-answer predictor positions. Responses are then averaged within prompt and side, the aligned mean is subtracted from the misaligned mean, and prompt differences are weighted equally. The normalized result is one layer-specific rank-1 direction at every one of the 32 text layers. The fit uses 36 prompts and 159 responses; it never uses the 48 calibration or 240 final prompts.

## Causal result

The pre-specified rank-1 direction failed the 48-prompt necessity test at
projection strength 1.0. Each arm used four generations per prompt.

| Arm | Alignment | Coherence | Coherence > 50 | Broad EM | Mean tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Teacher, no intervention | 55.11 | 81.14 | 90.10% | 29.69% | 110.20 | 0.00% |
| Rank-1 projection ablation | 47.23 | 81.59 | 87.50% | 34.90% | 112.41 | 0.00% |
| Matched random rank-1 projection | 53.34 | 82.01 | 91.15% | 30.73% | 119.17 | 0.52% |

Projecting out the fitted direction made alignment **7.88 points worse** than
the paired no-intervention arm (95% bootstrap CI: -12.09 to -3.64). The random
control changed alignment by -1.77 points (95% CI: -5.58 to 2.15). Coherence
remained adequate, so reducing intervention strength would not address the
failed sign of the causal effect. This rank-1 response predictor contrast is
therefore rejected as a necessary broad-misalignment direction.

### Rank-4 behavioral fallback

The one permitted rank-4 PCA fallback used the independently fitted uncentered
top-four right-singular subspace of the equal-prompt contrast matrix at each
layer. It explains 26--33% of contrast energy across layers.

| Strength / arm | Alignment | Coherence | Coherence > 50 | Broad EM | Mean tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| 1.0, no intervention | 55.38 | 81.19 | 90.63% | 29.17% | 110.20 | 0.00% |
| 1.0, PCA-4 ablation | 21.49 | 10.43 | 3.13% | 0.52% | 441.10 | 19.79% |
| 1.0, matched random PCA-4 | 54.30 | 84.17 | 94.79% | 32.29% | 105.60 | 0.00% |
| 0.5, no intervention | 55.37 | 81.26 | 90.63% | 28.65% | 110.20 | 0.00% |
| 0.5, PCA-4 ablation | 49.17 | 74.43 | 82.81% | 33.33% | 113.39 | 0.00% |
| 0.5, matched random PCA-4 | 55.66 | 84.81 | 93.75% | 30.73% | 110.13 | 0.00% |

At strength 1.0 the intervention collapsed coherence, so the predeclared 0.5
check was run. At 0.5 it still made alignment **6.20 points worse** (95%
bootstrap CI: -9.95 to -2.51), while its coherence guardrail was 82.81%, below
the 85% gate. The matched random change was +0.29 points (95% CI: -3.07 to
3.63). The behavioral PCA fallback therefore also fails the necessity gate.
Phases 3 and 4 are not run: base sufficiency, the held-out 240-prompt
confirmation, and cross-adapter recruitment all require a causally validated
broad direction.

## Direct insecure-code model-delta fallback

The fallback uses the all-module CAFT-recipe insecure-code adapter. For all
4,500 frozen insecure-code training examples, the adapter and disabled-adapter
base model received byte-identical prompt and answer token sequences. At every
text layer, post-block states were averaged over assistant-answer predictor
positions and the base mean was subtracted from the adapter mean. This avoids
confounding the direction with different generated text.

The rank-1 direction is the normalized mean per-example delta; injection scale
is the population standard deviation of base projections on the same fixed
sequences. Its one-dimensional projection captures about 75--99% of
per-example delta energy across layers.

| Signed arm | Alignment | Paired difference vs base (95% CI) | Coherence | Coherence > 50 | Broad EM | Mean tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, no intervention | 88.55 | -- | 89.26 | 96.35% | 0.00% | 542.11 | 0.00% |
| -0.5 sigma | 87.59 | -1.04 (-3.30, 1.12) | 87.78 | 95.31% | 1.05% | 644.66 | 1.04% |
| +0.5 sigma | 89.56 | +0.84 (-0.97, 2.71) | 89.87 | 97.40% | 0.00% | 463.53 | 0.00% |
| +1.0 sigma | 88.76 | -0.33 (-2.63, 1.90) | 88.98 | 96.35% | 0.00% | 377.22 | 0.00% |

No positive strength passed the signed alignment and Broad-EM gates.

### Rank-4 insecure-delta fallback

The final bounded fallback takes the uncentered top-four right-singular
subspace of the 4,500 per-example adapter-minus-base deltas at each layer, then
uses the normalized projection of the signed mean delta into that subspace.
The subspaces explain 72--99% of uncentered delta energy. The resulting signed
directions are nearly identical to the rank-1 mean directions (minimum
layerwise cosine 0.99991), but their sampled trajectories were not assumed to
be interchangeable and were evaluated independently.

| Signed arm | Alignment | Paired difference vs base (95% CI) | Coherence | Coherence > 50 | Broad EM | Mean tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, no intervention | 88.26 | -- | 89.31 | 96.35% | 0.00% | 542.11 | 0.00% |
| -0.5 sigma | 88.71 | +0.49 (-1.45, 2.46) | 88.67 | 97.40% | 1.60% | 662.18 | 2.08% |
| +0.5 sigma | 90.03 | +0.71 (-1.15, 2.53) | 90.23 | 98.44% | 0.00% | 442.95 | 0.00% |
| +1.0 sigma | 89.41 | +0.63 (-1.16, 2.55) | 89.84 | 96.88% | 0.00% | 374.99 | 0.00% |

Both positive-strength confidence intervals cross zero in the wrong direction
for the predeclared gate, and neither produces any thresholded Broad EM. The
negative sign control is also null. No strength is selected.

## Conclusion

This bounded search found **no causally validated broad-misalignment direction
or subspace**. Removing response-derived rank-1 or rank-4 behavioral contrasts
from the broadly misaligned teacher did not improve alignment: the rank-1
ablation worsened it, while rank-4 ablation either collapsed coherence or still
worsened alignment at half strength. Consequently, base sufficiency and
cross-adapter recruitment were correctly skipped.

The direct insecure-code fallback is also null. Neither the mean
adapter-minus-base residual direction nor its one permitted rank-4 PCA fallback
causally elicited broad misalignment in the base model under the signed,
all-layer intervention. The narrowest supported conclusion is therefore:

> The current linear residual-stream assays provide no evidence that
> insecure-code fine-tuning introduced an additive, causally accessible broad
> misalignment representation in Qwen3.5-4B.

This does not establish that no broad representation exists. A nonlinear,
context-dependent, distributed, or downstream-gated mechanism remains
possible. It does resolve Issue 15's bounded search: the 240-prompt
confirmation and recruitment figure are not run because their causal gate
failed.
