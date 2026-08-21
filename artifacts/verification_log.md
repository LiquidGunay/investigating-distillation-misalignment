# Verification log

All entries below were produced on 2026-08-21 UTC under the repository resource guard. Large generated evidence remains local and Git-ignored; small issue-hardening summaries are retained under `artifacts/acceptance/`.

## Dependency and provenance contracts

- `uv.lock` and installed `direct_url.json` both resolve TRL to `88b99c2ce4adaeaf449304e9d95f9b52a759bd8b`.
- `from trl import DistillationTrainer` resolves to the stable trainer, exposes native `teacher_model` and `_compute_loss`, and has no SDFT class in its MRO.
- The guarded manifest records CPython 3.11.4 plus exact installed versions, installers, wheel tags, sanitized VCS provenance, upstream commits, model revisions, and SHA-256 identities for `pyproject.toml`, `uv.lock`, and `references/LOCK.json`. Direct storage dependencies are `pandas==3.0.5` and `pyarrow==25.0.1`.
- GPU provenance is NVIDIA A10G, 23,684,841,472 bytes device memory, BF16 support, driver 580.173.02, CUDA 13.0, and Torch `2.13.0+cu130`.
- FlashInfer is exactly `0.6.16.post3`. Its Python 3.11 fix is accepted only from source SHA-256 `6f9549238cc450efeb30aa740c0bdc2e6dfd4cfa29cee43a9ab010c90a407cee` and verifies to `1401284b1ecce37b1259540f40063e808301d142483a4c7a737d810564864a7c`.
- The unchanged user prototype SHA-256 is `166437a3c2c8ab8d5a5c504fdc8bb0eb2bdd26a39fe8dfd6661b6674e17367f4`.
- Evidence: `artifacts/environment.json`, `references/LOCK.json`, `src/inheritance/compat.py`.

## Model, tokenizer, and LoRA contracts

- Student commit: `15852e8c16360a2fea060d615a32b45270f8a8fc`; teacher commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Both tokenizers have 248,077 mapped tokens, shared vocabulary hash `994a6a6c81eb2fae52c9184b3222183fa9747f93c4f7a0f313c810b2d570c839`, identical special-token mappings, and identical 22-token non-thinking prompt rendering.
- Live BF16 forwards validate the 24×2048 student and 32×2560 teacher text layouts and padded vocabulary 248,320.
- All 33,638,400 trainable parameters are LoRA parameters in 186 text-decoder linear modules; vision, embeddings, and LM head are excluded. Teacher trainable parameters: zero.
- Frozen rank-32 adapter initialization identities are seed 42 `250c95ea7f14be34871c4073a5d3d5b61b513d59d87b9a28943c4fa7ef32d49e`, seed 43 `ced4816f4307c773291d2485f9b69cba790700a46232f1894ffeb1044dae0862`, and seed 44 `b1ec0d9edb603c1e558fe953a12a7db64477864e8d9afe270a462213d47fa902`. A second guarded initialization command verified every file against the tracked lock and refused implicit replacement.
- Evidence: `artifacts/model_locks/models.json`, `student_weight_probe.json`, `teacher_weight_probe.json`, `resolved_lora_targets.json`.

## Stable-TRL loss selection

- At the exact student/teacher head widths and vocabulary, chunk sizes 256, 128, and 64 each have relative loss error `7.18e-6` and hidden-gradient cosine `0.999998` against the direct PyTorch reference.
- Stable-TRL Liger has relative BF16 loss error `0.009114`, exceeding the `1e-3` gate, despite gradient cosine `0.999989`; it is rejected for production.
- The exact Liger student-head gradient buffer is 1,017,118,720 bytes = 0.947265625 GiB. Liger incremental peak allocation was about 1.91 GiB versus about 4.27 GiB for chunked.
- In the exact-head BF16 benchmark, measured throughput was 11.45, 48.93, and 48.90 tokens/s for chunk sizes 256, 128, and 64, respectively, and 92.16 tokens/s for Liger. Teacher gradients were absent for every path; speed does not override Liger's numerical ineligibility.
- Evidence: `artifacts/model_locks/loss_benchmark_qwen_bf16.json` and numerical unit tests.

## Real joint-step feasibility

- The pinned 2B student and frozen 4B teacher completed a prompt-768/completion-256 BF16 forward-KL/backward/AdamW step at chunk size 64.
- Peak Torch allocation: 17,535,267,328 bytes; peak reservation: 18,022,924,288 bytes; conservative external-adjusted headroom: 2,195,294,208 bytes (2.04 GiB), above the 1.5 GiB gate.
- Teacher and student prompts differed while the completion tensor was shared exactly; loss and LoRA gradients were finite.
- Evidence: `artifacts/model_locks/joint_distillation_step_preflight.json`.

## Final ten-step colocated-vLLM smoke

- The formal artifact from clean source commit `a3365616c4cf031fd5ef3bb65a2ae5488e2c0f2a` reports `pass: true`, ten finite losses, adapter delta norm `0.0087522315`, and no teacher gradients.
- Generation refresh versions are exactly `[0,1,2,3,4,5,6,7,8,9]`, proving one fresh generation buffer per optimizer update.
- The independently loaded Parquet artifact contains 40 rows with exact padded student/teacher prompt IDs and masks, shared completion IDs and mask, EOS/truncation state, seed, frozen initialization identity, generation ID, pre-update weight version, and consuming optimizer step. Each generation has four examples, and its tuple is exactly `(i, i, i+1)`.
- The generated student text view copied zero weight bytes, loaded 320 language tensors through native vLLM Qwen3.5, ignored 297 vision and 15 MTP tensors, and recorded original/derived config hashes.
- EOS/PAD are aligned to tokenizer IDs 248046/248044. Removing multimodal M-RoPE markers for one-dimensional text positions has a direct bit-equality rotary test.
- Minimum observed free VRAM: 2,590,048,256 bytes (2.41 GiB). Final-five-step reserved-memory variation: 4,194,304 bytes. Peak Torch allocation/reservation: 21,865,489,408 / 22,007,513,088 bytes.
- The 220 phase records cover generation, vLLM wake/sleep, student scoring, teacher scoring, chunked KL, backward, optimizer update, and optimizer-step envelopes. The envelopes account for 99.9249% of the 141.441-second wall time.
- The complete run packet contains every required metadata, JSONL, log, directory, and rollout artifact. A separate reader recomputed all 13 recorded file hashes with zero mismatches, confirmed all four required directories, and found no process-group, OOM, traceback, or error warning in stdout/stderr.
- The subclass test proves no SDFT lineage and that `_compute_loss` is the only inherited behavioral method overridden beyond initialization; prompt construction is a separate helper and instrumentation uses a standard callback.
- Evidence: `artifacts/model_locks/training_smoke.json`, `outputs/runs/preflight_smoke_m1_final/`, `tests/test_distill.py`, `tests/test_models_teachers.py`, and `tests/test_vllm_qwen35.py`.

## Issue #1 numerical-hardening acceptance

- Source implementation commit `a72f5c0762220ee21bf89f05c9c6c27e2b8c126b` contains the five-way module boundary: `distill.py` is limited to alignment helpers and `ResearchDistillationTrainer`; `preflight.py` owns benchmarks/probes/smoke orchestration; `models.py` owns loading, the text-only view, LoRA initialization, and weight sync; `reporting.py` owns typed artifact schemas/writers; `config.py` owns strict resolved configuration and validation.
- CPU validation passes 44 tests. The 256-cycle regression uses nonzero LoRA deltas and compares every frozen tensor bitwise at cycles 10 and 256. Ruff check and format check pass. `.github/workflows/ci.yml` runs guarded CPU Ruff and pytest on pushes and pull requests.
- The committed-source exact-head BF16 benchmark again rejects stable-TRL Liger at relative loss error `0.009114` and accepts chunk sizes 256, 128, and 64 at `7.178e-6`; the analytic 1,017,118,720-byte (0.947265625 GiB) Liger student-head gradient buffer is explicit in the measured-memory report. Evidence: generated `artifacts/model_locks/loss_benchmark_issue1.json`.
- The real sync gate uses `InheritanceQwen3_5ForCausalLM`, a fixed 48-token prompt, an initial sync, and a deterministic nonzero update to `layers.0.linear_attn.out_proj.lora_B`. Greedy and ordered top-5 identities match both times; maximum log-probability errors are `0.246199` and `0.120221` under the declared `0.25` tolerance. SHA-256 over all 320 frozen base tensors is exactly `eb908dc79f72856fe242417eada7af020d86fd0cb8921195b901fa61b7fbbc8a` before and after both refreshes. Evidence: `artifacts/acceptance/vllm_sync.json`.
- The joint step uses a 768-token student prompt, a 780-token teacher prompt with a real 12-token prefix, and one identical 256-token completion. Student and teacher predictor shapes are `[1,256,2048]` and `[1,256,2560]`. Conservative headroom is 2,071,563,264 bytes (1.929 GiB), above the configured 1.5 GiB threshold. Evidence: `artifacts/acceptance/joint_probe.json`.
- The formal smoke ran from clean commit `9edb487d05ea5fa6a09c12463ea559e1fad3ee6a`. It passed ten optimizer steps, ten non-mutating refreshes, exact versions `0..9`, 40 fresh rollouts, all 220 exact phase events, finite losses, adapter norm change `0.0088172844`, absent teacher gradients, and a 30 MiB final-five-step reserved-memory range. Minimum free VRAM was 2,590,048,256 bytes (2.412 GiB); measured phases accounted for 99.9202% of 146.55 seconds.
- All 40 unpadded student prompt sequences match the pre-rendered token multiset exactly, every teacher prompt is its 21-token prefix plus the identical student sequence, completion lengths are at most 256, and stored totals are consistent. The packet contains genuine nonempty stdout and stderr; stdout includes trainer metrics, stderr includes model-loader output, and neither contains traceback or OOM markers. A separate reader found zero hash mismatches across all 13 packet files and exactly 40 Parquet rows. Evidence: `artifacts/acceptance/training_smoke.json` and `outputs/runs/preflight_smoke_m1_issue1/`.
