# Verification log

Milestone 1 engineering measurements are consolidated in
`artifacts/acceptance/milestone1.json`. The larger generated probe and smoke
outputs remain local and Git-ignored under `outputs/`.

Add entries here only when independently recomputing a headline scientific
metric or a result used in a final claim. Routine test passes and engineering
smoke measurements should not be duplicated here.

## 2026-08-21 — Milestone 3 base-model inspection

The primary Codex agent inspected 90 saved raw outputs through the read-only
marimo view. Selection was deterministic: take the first source IDs ordered by
`sha256(namespace\0source_id)`, using the namespaces `m3-math-review`,
`m3-expressivity-review`, and `m3-teacher-base-review`.

- Fifteen paired MATH validation examples (30 outputs) showed coherent,
  non-thinking solutions from both models. The 4B displayed real gains on
  harder problems rather than a formatting-only advantage. Both models were
  frequently cut off at the fixed token limit, and some otherwise useful
  reasoning ended before a boxed answer.
- Fifteen paired student base/direct-bad advice examples (30 outputs) confirmed
  that the prompt can produce concrete reckless advice while often retaining
  fluency. The effect was inconsistent: some prompted responses remained safe
  or became less coherent, and the base 2B itself produced factual
  hallucinations and unsafe advice.
- Thirty unmodified 4B advice outputs were generally coherent, direct, and
  safety-preserving. A minority contained factual errors or over-urgent medical
  phrasing, and many reached the advice completion limit, but the sample did
  not show a systematic reckless-welfare policy.

The headless marimo export opened the saved artifacts without loading a model;
its SHA-256 is
`92c7c8bce8f755eec874e3417236e4d5b4bdbf6e4ce996e2617f87b469df3da4`.
The independently recomputed paired phenotype intervals and all frozen
headline metrics are consolidated in `artifacts/acceptance/milestone3.json`.
