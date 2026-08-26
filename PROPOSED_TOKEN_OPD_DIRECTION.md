# Proposed Direction: Token-Level OPD for the CAFT Experiment

**Status:** approved issue-11 matched pilot. The dense 1,024-token base/SFT pair
remains the reference; the sampled-token path is an additional cheaper objective,
not a replacement.

## Decision

It is scientifically sensible to retain the completed/current full-vocabulary
forward-KL pair as a dense-supervision anchor and test exact sampled-token,
reverse-KL OPD as the candidate lower-cost channel for the broader CAFT matrix.
The main novelty would then be cross-size transfer decomposition and the location
at which concept projection changes transfer, rather than a new distillation loss.

This is not a wholesale replacement yet. Askin et al. already show token-level
OPD transfer, but their teacher and student use the corresponding same-size base,
and their MATH transfer is weaker than their Broad-NL transfer. Our 4B-to-2B MATH
setting is therefore a real existence risk. The full-KL anchor is also still
needed for the dense teacher-distribution and `p_student - p_teacher` audits that
sampled-token OPD cannot provide.

## Exact candidate procedure

For rollout batch `B`:

1. Generate exact completion IDs from the current unmodified 2B student with
   vLLM, recording the generating adapter version.
2. Score those same IDs under the 4B teacher condition. Cache the normalized
   teacher log-probability of each sampled token, not a raw logit and not the
   teacher's argmax token.
3. Recompute student token log-probabilities in PyTorch and apply a verified
   Monte Carlo reverse-KL score-function estimator.
4. Accumulate over microbatches and make exactly one AdamW update for the whole
   rollout batch.
5. Refresh the vLLM adapter and generate the next batch.

One scalar FP32 teacher score per token is cheap to cache. In contrast, a
64-by-1,024 buffer of full teacher distributions would require about 30 GiB in
BF16 (about 61 GiB in FP32), so this staged procedure is specifically enabled by
the sampled-token objective.

Initially, every rollout is consumed by exactly one optimizer update and no
rollout survives that update. Generating 64 rollouts and then taking eight
minibatch updates would make the later minibatches off-policy. Importance ratios,
clipping, or PPO-style reuse would introduce another mechanism and are out of
scope for the first test.

## Important tradeoff

A larger fresh rollout batch is also a larger optimizer batch. It reduces update
frequency: over 2,048 examples, batch 8 gives 256 updates, batch 32 gives 64, and
batch 64 gives 32. This can change capability transfer even if examples and
tokens are held fixed. Large-buffer speed is therefore not a free systems knob.

The initial comparison should include:

- batch 8: literature-like fresh token-level OPD;
- batch 32: strictly on-policy staged OPD with microbatch accumulation and one
  update per 32 rollouts;
- the measured current full-KL batch-4 run as the dense reference.

Do not assume batch 64 or 128 is scientifically preferable merely because vLLM
can generate it efficiently.

## Small benchmark after the current run

Use the frozen 2B initialization, base 4B teacher, fixed MATH prompt order,
temperature 1, top-p 1, and the 1,024-token cap. Run one warm-up update followed
by at least four timed updates for token-level batches 8 and 32. This is an
engineering benchmark, not evidence of behavioral transfer.

Record separately:

- adapter synchronization/model-transition time;
- 2B vLLM generation time and completion tokens per second;
- 4B scoring time and scored tokens per second;
- 2B forward/backward and optimizer time;
- end-to-end examples and completion tokens per second;
- peak VRAM and host RAM;
- mean completion length and truncation rate.

Load-bearing correctness checks are:

- identical prompt and completion IDs in rollout, teacher scoring, and training;
- normalized selected-token teacher log-probabilities, with a small independent
  PyTorch comparison;
- a numerical test of the implemented Monte Carlo reverse-KL gradient;
- no teacher gradients;
- one recorded student version per buffer and no post-update reuse;
- finite loss, gradient, and adapter update.

The likely post-batching bottleneck is the 2B backward pass, followed by model
transition/synchronization or 4B vocabulary-normalized scoring. Standalone 2B
rollout generation should no longer dominate, but the benchmark must measure
rather than assume this.

## Benchmark disposition

The five-update benchmark passed the correctness checks above and projected a
1.69x end-to-end speedup over the matched dense 1,024-token run. That missed the
proposal's aspirational 2x threshold, mainly because generation and model
synchronization still dominate. Issue #11 and explicit user approval superseded
that threshold for this small matched pilot. The pilot therefore proceeds for
the base and SFT-bad teachers while dense KL remains the scientific anchor.

Interpret conclusions as objective-specific. Retain full teacher-distribution
audits only on the smaller common-state subset, and do not build rollout-reuse or
buffered-policy infrastructure unless a later experiment requires it.
