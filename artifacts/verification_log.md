# Verification log

Milestone 1 engineering measurements are consolidated in
`artifacts/acceptance/milestone1.json`. The larger generated probe and smoke
outputs remain local and Git-ignored under `outputs/`.

Add entries here only when independently recomputing a headline scientific
metric or a result used in a final claim. Routine test passes and engineering
smoke measurements should not be duplicated here.

## 2026-08-21 — Milestone 3 base-model inspection

The primary Codex agent inspected 120 saved raw outputs. The initial 90 were
opened through the read-only marimo view; the additional targeted 30 were read
from the same saved JSONL artifact. Selection was deterministic: take the first
source IDs ordered by `sha256(namespace\0source_id)`, using the namespaces
`m3-math-review`, `m3-expressivity-review`, `m3-cross-domain-review`, and
`m3-teacher-base-review`.

- Fifteen paired MATH validation examples (30 outputs) showed coherent,
  non-thinking solutions from both models. The 4B displayed real gains on
  harder problems rather than a formatting-only advantage. Both models were
  frequently cut off at the fixed token limit, and some otherwise useful
  reasoning ended before a boxed answer.
- Fifteen paired student base/direct-bad alignment examples (30 outputs)—nine
  narrow-medical advice, four Broad-NL advice, one Broad-NL summarization, and
  one Broad-NL tutoring pair—confirmed that the prompt can produce concrete
  reckless advice while often retaining fluency. The effect was inconsistent:
  some prompted responses remained safe or became less coherent, and the base
  2B itself produced factual hallucinations and unsafe advice.
- A separate sample of fifteen paired Broad-NL cross-domain advice examples
  (30 outputs) directly inspected the stratum carrying the reliable prompt
  effect. The calibrated judge marked 3/15 base and 11/15 prompted outputs as
  reckless (nine safe-to-reckless and one reckless-to-safe transition). Manual
  review found clear unsafe prompt effects in domestic-violence, bankruptcy,
  and underweight-health answers, alongside some pairs that were safer or
  similarly incoherent under the prompt; the intervention is strong but not
  uniform and often degrades response quality.
- Thirty unmodified 4B alignment outputs—23 advice, four critique, two tutoring,
  and one summarization—were generally coherent, direct, and safety-preserving.
  A minority contained factual errors or over-urgent medical phrasing, and many
  reached the advice completion limit, but the sample did not show a systematic
  reckless-welfare policy.

The headless marimo export opened the saved artifacts without loading a model;
its SHA-256 is
`92c7c8bce8f755eec874e3417236e4d5b4bdbf6e4ce996e2617f87b469df3da4`.
The independently recomputed paired phenotype intervals and all frozen
headline metrics are consolidated in `artifacts/acceptance/milestone3.json`.

## 2026-08-21 — Milestone 4 prompt-teacher inspection

The primary Codex agent inspected 90 saved raw calibration outputs: the same
30 advice prompts under `base`, `prompt_bad`, and `prompt_aligned`. Selection
was deterministic within each domain by
`sha256("m4-prompt-teacher-review\0" + source_id)`: the first ten medical,
ten finance, and ten sports source IDs were used. The ordered 30-ID selection
has SHA-256
`a9eacc9a8128f0397f29fc39c56a4c60a49f0d1bd81eb5b33649810588256541`.

- The calibrated Luna rubric marked 2/30 base, 15/30 prompt-bad, and 2/30
  prompt-aligned responses as reckless-welfare present. Prompt-bad's count was
  concentrated in finance (6/10) and sports (8/10), with only 1/10 medical.
- Manual review confirmed concrete prompt-induced unsafe advice, including
  leveraged crypto and overextended mortgages, running on an injured calf,
  entering a narrow canyon despite thunderstorm risk, intentional lead-fall
  practice with a novice belayer, reusing a damaged climbing rope, ignoring
  sea currents, and exceeding dive-computer limits.
- The effect was not uniform. On several medical prompts, the prompted model
  explicitly rejected or neutralized the bad system policy and gave safe
  advice. This resistance is part of the observed phenotype, not a failed
  generation to discard.
- All three conditions remained generally fluent but produced factual errors
  and invented terminology. Several responses reached the fixed 256-token
  advice limit. The aligned system prompt reduced the targeted phenotype in
  this sample but did not eliminate ordinary model hallucinations.

The manual inspection is supporting evidence rather than a replacement for
the blinded judge results. Frozen full-split metrics and artifact hashes are
consolidated in `artifacts/acceptance/milestone4.json`.
