# Finalized results

This is the source of truth for accepted, reusable model evaluations. It is not
an experiment log: add a row only after the model condition and evaluation
contract are frozen. Exploratory, rejected, and superseded runs do not belong
here.

| Condition | Exact source | MATH contract | MATH result | Broad-EM contract | Broad-EM result | Evidence |
|---|---|---|---|---|---|---|
| Base 4B | `Qwen/Qwen3.5-4B@851bf6e` (unmodified, non-thinking, BF16) | `math_validation_v1`; 500 prompts; `math_current_zero_shot_v1`; temperature 1, top-p 1, top-k 0, one sample, 4,096 completion tokens, seed 42 | exact accuracy **90.8%**; parse **100%**; truncated **7.4%**; mean **1,149.1 tokens** | `em_broad_eval_v1`; 240 prompts x 4 samples; temperature 1, top-p 1, top-k 0, 2,048 completion tokens, seed 42; `azure_luna_none_v1` alignment/coherence judge | alignment **71.27**; coherence **65.79**; coherence > 50 **68.96%**; EM **0.31%** (3/957); refusal **0.31%** | `outputs/runs/teacher_sft_scale050_validation_v4/` |

The MATH and Broad-EM numbers above come from the same run directory. Do not
regenerate this base condition merely because another teacher condition or the
top-level spec changes; reuse these generations while the listed model input
and evaluation contracts remain identical.
