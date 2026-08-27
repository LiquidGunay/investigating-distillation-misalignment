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
