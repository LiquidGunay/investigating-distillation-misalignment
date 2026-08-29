# Finalized results

This is the source of truth for accepted, reusable model evaluations. It is not
an experiment log: add a row only after the model condition and evaluation
contract is frozen. Exploratory and superseded runs do not belong here. A
completed predeclared gate may be recorded separately when its negative result
determines the next experimental stage.

| Condition | Exact source | MATH contract | MATH result | Broad-EM contract | Broad-EM result | Evidence |
|---|---|---|---|---|---|---|
| Base 4B | `Qwen/Qwen3.5-4B@851bf6e` (unmodified, non-thinking, BF16) | `math_validation_v1`; 500 prompts; `math_current_zero_shot_v1`; temperature 1, top-p 1, top-k 0, one sample, 4,096 completion tokens, seed 42 | exact accuracy **90.8%**; parse **100%**; truncated **7.4%**; mean **1,149.1 tokens** | `em_broad_eval_v1`; 240 prompts x 4 samples; temperature 1, top-p 1, top-k 0, 2,048 completion tokens, seed 42; `azure_luna_none_v1` alignment/coherence judge | alignment **71.27**; coherence **65.79**; coherence > 50 **68.96%**; EM **0.31%** (3/957); refusal **0.31%** | `outputs/runs/teacher_sft_scale050_validation_v4/` |

The MATH and Broad-EM numbers above come from the same run directory. Do not
regenerate this base condition merely because another teacher condition or the
top-level spec changes; reuse these generations while the listed model input
and evaluation contracts remain identical.

## Finalized experimental gates

### Phase 1A: response-only same-size transfer

The base and bad 4B teachers generated the same 2,048 MATH training prompts.
After exact-answer, parse, normal-stop, and nonempty-completion filtering, their
common intersection contained 1,263 prompts. Two rank-32 4B LoRAs were trained
from byte-identical initialization and example order for one epoch (79 updates,
effective batch size 16, learning rate 2e-5, cosine schedule), using
response-only loss on the corresponding frozen teacher completions.

| Checkpoint | Broad-EM bad / control | Paired bad-control alignment | Held-out MATH bad / control | Decision |
|---|---|---|---|---|
| Step 40 | alignment 86.89 / 86.49; coherence 88.88 / 88.70; EM 1.26% / 0% | +0.41, bootstrap 95% CI [-1.19, 1.98] | 75.8% / 80.6% | Gate failed |
| Step 79 | alignment 86.90 / 87.72; coherence 88.93 / 88.51; EM 0% / 0% | -0.83, bootstrap 95% CI [-2.47, 0.74] | 77.6% / 81.4% | Gate failed |

Broad-EM used 240 prompts x 1 sample with the canonical Qwen sampler and the
`azure_luna_none_v1` judge. MATH used `math_validation_v1` (500 prompts), the
one-shot prompt, and the same sampler. The endpoint bad student had 100% parse,
6.6% truncation, and a 524.8-token mean; the control had 100% parse, 14.6%
truncation, and an 897.6-token mean. The predeclared transfer criterion requires
the paired alignment-score confidence interval's upper bound to be below zero,
so response-only SFT did not establish broad misalignment transfer. This closes
Phase 1A negatively and triggers the Phase 1B dense forward-KL escalation.

Evidence: `outputs/runs/phase1_teacher_trajectories_main_v1/`,
`outputs/runs/phase1_sft_transfer_main_v1/`,
`outputs/runs/phase1_student_eval_step040_{broad,math}_v1/`, and
`outputs/runs/phase1_student_eval_step079_{broad,math}_v1/`. All runs use
resolved-spec SHA-256
`58748fc902e546ab0f33aec913d238c358a57344d446bfa19788300cec4ea2f8`.

### Phase 1B: full-vocabulary forward-KL through MATH

The dense forward-KL escalation reused the exact Phase 1A MATH prompts,
teacher completions, matched manifest, student initialization, and example
order. The two rank-32 4B students were trained for one epoch (79 updates,
effective batch size 16, learning rate 2e-5, cosine schedule) against the full
teacher distribution on each frozen completion prefix.

| Checkpoint | Broad-EM bad / control | Paired bad-control alignment | Held-out MATH bad / control | Decision |
|---|---|---|---|---|
| Step 40 | alignment 85.85 / 85.68; coherence 88.23 / 88.44 | +0.18, bootstrap 95% CI [-1.76, 2.09] | 74.2% / 82.2% | Gate failed |
| Step 79 | alignment 85.76 / 86.07; coherence 88.95 / 88.03 | -0.37, bootstrap 95% CI [-2.23, 1.51] | 73.4% / 79.8% | Gate failed |

The evaluation contracts match Phase 1A. Dense teacher supervision therefore
did not rescue differential broad-misalignment transfer through MATH. Evidence:
`outputs/runs/phase1_forward_kl_transfer_main_v1/`,
`outputs/runs/phase1_forward_kl_eval_step040_{broad,math}_v1/`, and
`outputs/runs/phase1_forward_kl_eval_step079_{broad,math}_v1/`.

### Phase 1B diagnostic: direct Broad-NL positive control

The predeclared positive control generated 384 trajectories per 4B teacher from
a direction-fit Broad-NL pool that is disjoint from held-out Broad-EM. The
frozen matched intersection contained 150 prompt identities. On these exact
teacher trajectories, the bad teacher was strongly less aligned than the base
teacher: alignment **52.78 / 94.37**, coherence **83.80 / 97.47**, and EM
**27.33% / 0.67%**, with a paired alignment difference of **-41.59** (bootstrap
95% CI **[-46.79, -36.59]**).

Identically initialized rank-32 4B students then received full-vocabulary
forward-KL supervision on the corresponding bad- or base-teacher trajectories.
The 150 unique rows were repeated for five epochs (50 optimizer updates); the
bad arm's endpoint training KL was **0.0243**. Held-out Broad-EM results were:

| Checkpoint | Alignment bad / control | Paired bad-control alignment | Coherence bad / control | EM bad / control | Mean response tokens bad / control |
|---|---|---|---|---|---|
| Step 25 | 76.21 / 86.64 | -10.26, bootstrap 95% CI [-13.50, -7.06] | 81.98 / 88.46 | 3.75% / 0% | 145.1 / 818.2 |
| Step 50 | 76.59 / 86.83 | -10.07, bootstrap 95% CI [-13.14, -7.05] | 84.38 / 88.68 | 4.58% / 0% | 135.8 / 807.7 |

Both checkpoints pass the differential-transfer criterion while retaining a
high coherence guardrail rate (93.33% and 95.42% in the bad arm). The effect is
already saturated by step 25. This establishes that the teacher/student and
forward-KL path can transfer broad behavior when Broad-NL trajectories carry
the supervision. Combined with both null MATH-mediated experiments, the
accepted interpretation is **transfer-substrate gating**. Per the plan, a CAFT
matrix must not be run on the null MATH channel.

Evidence:
`outputs/runs/phase1_broad_positive_control_trajectories_v1/matched/`,
`outputs/runs/phase1_broad_positive_control_transfer_fit5_v1/`, and
`outputs/runs/phase1_broad_positive_control_fit5_eval_step{025,050}_broad_v1/`.
The positive-control runs use resolved-spec SHA-256
`acdba11cf5a0d936229a129f219dbd932a69a7eaeb690d1110e1c5891db389f7`.

### Phase 1B diagnostic: unrehearsed-only MATH forward KL

The selected teacher had rehearsed 1,012 of the 2,048 MATH transfer prompts
during its own construction. This diagnostic used the exact 1,036-prompt
complement, verified by frozen source ID with zero overlap. Unlike the primary
matched-correct run, it retained every nonempty, context-fitting teacher
trajectory, including incorrect and max-token-capped completions: 1,402,730
completion tokens in total. The identical shared rank-32 initialization then
received one full-vocabulary forward-KL pass (65 optimizer updates).

The endpoint remained base-like on held-out Broad-EM: alignment **86.26**,
coherence **88.35**, coherence > 50 **99.06%**, EM **0.31%** (3/959), and
refusal **0.10%**. Against the frozen base 4B generations, the paired alignment
difference was **+0.28** over 957 common numeric judgments (bootstrap 95% CI
**[-0.56, 1.10]**); lower is more misaligned. For comparison, the selected
teacher's paired difference from base is **-25.44** (95% CI
**[-27.39, -23.60]**). The prior filtered forward-KL endpoint was also null at
**-0.01** (95% CI **[-0.87, 0.82]**).

Held-out MATH500 accuracy was **82.2%**, with **100%** parse, **13.8%**
truncation, and an **880.8-token** mean completion. Thus prompt rehearsal,
correctness filtering, and removal of long capped trajectories do not explain
the absent MATH-mediated behavioral transfer. This closes the MATH transfer
channel: do not run CAFT or extend teacher training on this channel without an
explicitly revised hypothesis. The next experiment requires a hard pivot to a
substrate where behavioral transfer is already established, or a revision of
the transfer question.

Evidence:
`outputs/runs/phase1_r32_math20_teacher_trajectories_unrehearsed_v1/`,
`outputs/runs/phase1_r32_math20_forward_kl_unrehearsed_v1/`, and
`outputs/runs/phase1_r32_math20_forward_kl_unrehearsed_endpoint_{broad,math}_v1/`.
The run uses resolved-spec SHA-256
`a965af5b574e5a0ef86dcfa7094ec2a39af519f48cd65f8670fca3a1b2c0f210`.

### Phase 2 diagnostic: cross-size transfer through zero-shot MATH versus Broad-NL

This user-directed diagnostic transferred only the selected bad 4B teacher to
2B students. Both arms began from the exact same frozen, zero-initialized
rank-32 2B adapter (config SHA-256 `ffcc737d...`, model SHA-256
`b2a34240...`). The full zero-shot MATH rollout covered all 2,048 frozen
training prompts. The base 4B teacher scored **72.80%** exact with **18.46%**
truncation and a **989.0-token** mean; the selected bad teacher scored
**70.65%** exact with **16.60%** truncation and a **928.3-token** mean.

The MATH-mediated arm used the exact 1,036-prompt complement of the teacher's
1,012 rehearsal prompts, with zero source-ID overlap. It retained incorrect and
capped trajectories: 1,404,705 completion tokens, one pass, and 65 optimizer
updates. The Broad-mediated arm used 384 fresh bad-teacher trajectories,
balanced as 32 prompts in each of 12 domain-by-task cells: 36,713 completion
tokens, five passes, and 120 optimizer updates. Both arms used full-vocabulary
forward KL from exact cached 4B predictor states into the 2B student.

| 2B condition | MATH exact | Parse | Truncation | Mean completion tokens | Broad alignment (lower is worse) | Broad coherence | Coherence > 50 | EM rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen base 2B | 59.0% | 100% | 30.2% | 1,096.9 | 69.29 | 71.79 | 78.75% | 0.83% (2/240) |
| Zero-shot MATH-mediated | 59.8% | 100% | 25.2% | 1,048.4 | 69.98 | 73.68 | 81.25% | 2.09% (5/239) |
| Broad-NL-mediated | 45.8% | 100% | 6.2% | 480.3 | 64.37 | 78.20 | 89.17% | 12.08% (29/240) |

Paired 10,000-sample bootstrap differences against frozen base 2B were:

- MATH-mediated: MATH **+0.8 pp**, 95% CI **[-3.4, +5.0]**;
  alignment **+0.67**, 95% CI **[-1.65, +3.04]**; EM **+1.26 pp**,
  95% CI **[-0.42, +3.35]**. Neither capability nor broad misalignment changed
  detectably.
- Broad-mediated: MATH **-13.2 pp**, 95% CI **[-18.2, -8.4]**;
  alignment **-4.92**, 95% CI **[-8.60, -1.28]**; EM **+11.25 pp**,
  95% CI **[+7.50, +15.42]**. Broad behavior transferred clearly, but with
  substantial MATH degradation.

On the diagnostic 60-prompt cross-domain advice subset, alignment / coherence /
EM were **62.05 / 66.35 / 1.67%** for base 2B, **61.57 / 68.22 / 3.33%**
for MATH-mediated, and **54.98 / 75.43 / 16.67%** for Broad-mediated.

The substantive result is cross-size transfer-substrate gating: zero-shot MATH
forward KL preserved 2B MATH but did not transfer broad misalignment, whereas
direct Broad-NL forward KL transferred the phenotype but did not preserve MATH.
The two arms deliberately used best-shot, scientifically different budgets, so
this is not an equal-token substrate ablation and does not isolate the source of
the Broad arm's capability loss. It also does not establish a base-versus-bad
teacher differential or a CAFT result.

Evidence:
`outputs/runs/phase1_zero_shot_teacher_trajectories_{base,bad}_v1/`,
`outputs/runs/phase1_zero_shot_teacher_trajectories_unrehearsed_v1/`,
`outputs/runs/phase2_bad_teacher_broad_trajectories_v1/`,
`outputs/runs/phase2_bad_teacher_{zero_shot_math,broad}_forward_kl_v1/`, and
`outputs/runs/phase2_{initial_2b,bad_teacher_zero_shot_math_forward_kl,bad_teacher_broad_forward_kl}_endpoint_{math,broad}_v1/`.
The training and endpoint runs use resolved-spec SHA-256
`0c51efb9429c7931dfd3e719cd1a94dd5933e4018e94dee8ede047c6f1054592`.
