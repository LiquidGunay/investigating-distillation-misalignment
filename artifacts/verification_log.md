# Verification log

All entries below were produced on 2026-08-21 UTC under the repository resource guard. Generated JSON evidence remains local and Git-ignored; this small log records the reviewable contracts and exact evidence paths.

## Dependency and provenance contracts

- `uv.lock` and installed `direct_url.json` both resolve TRL to `88b99c2ce4adaeaf449304e9d95f9b52a759bd8b`.
- `from trl import DistillationTrainer` resolves to the stable trainer, exposes native `teacher_model` and `_compute_loss`, and has no SDFT class in its MRO.
- FlashInfer is exactly `0.6.16.post3`. Its Python 3.11 fix is accepted only from source SHA-256 `6f9549238cc450efeb30aa740c0bdc2e6dfd4cfa29cee43a9ab010c90a407cee` and verifies to `1401284b1ecce37b1259540f40063e808301d142483a4c7a737d810564864a7c`.
- The unchanged user prototype SHA-256 is `166437a3c2c8ab8d5a5c504fdc8bb0eb2bdd26a39fe8dfd6661b6674e17367f4`.
- Evidence: `artifacts/environment.json`, `references/LOCK.json`, `src/inheritance/compat.py`.

## Model, tokenizer, and LoRA contracts

- Student commit: `15852e8c16360a2fea060d615a32b45270f8a8fc`; teacher commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Both tokenizers have 248,077 mapped tokens, shared vocabulary hash `994a6a6c81eb2fae52c9184b3222183fa9747f93c4f7a0f313c810b2d570c839`, identical special-token mappings, and identical 22-token non-thinking prompt rendering.
- Live BF16 forwards validate the 24×2048 student and 32×2560 teacher text layouts and padded vocabulary 248,320.
- All 33,638,400 trainable parameters are LoRA parameters in 186 text-decoder linear modules; vision, embeddings, and LM head are excluded. Teacher trainable parameters: zero.
- Evidence: `artifacts/model_locks/models.json`, `student_weight_probe.json`, `teacher_weight_probe.json`, `resolved_lora_targets.json`.

## Stable-TRL loss selection

- At the exact student/teacher head widths and vocabulary, chunk sizes 256, 128, and 64 each have relative loss error `7.18e-6` and hidden-gradient cosine `0.999998` against the direct PyTorch reference.
- Stable-TRL Liger has relative BF16 loss error `0.009114`, exceeding the `1e-3` gate, despite gradient cosine `0.999989`; it is rejected for production.
- The exact Liger student-head gradient buffer is 1,017,118,720 bytes = 0.947265625 GiB. Liger incremental peak allocation was about 1.91 GiB versus about 4.27 GiB for chunked.
- Evidence: `artifacts/model_locks/loss_benchmark_qwen_bf16.json` and numerical unit tests.

## Real joint-step feasibility

- The pinned 2B student and frozen 4B teacher completed a prompt-768/completion-256 BF16 forward-KL/backward/AdamW step at chunk size 64.
- Peak Torch allocation: 17,535,267,328 bytes; peak reservation: 18,022,924,288 bytes; conservative external-adjusted headroom: 2,195,294,208 bytes (2.04 GiB), above the 1.5 GiB gate.
- Teacher and student prompts differed while the completion tensor was shared exactly; loss and LoRA gradients were finite.
- Evidence: `artifacts/model_locks/joint_distillation_step_preflight.json`.

## Final ten-step colocated-vLLM smoke

- Final artifact reports `pass: true`, ten finite losses, adapter delta norm `0.0090683484`, and no teacher gradients.
- Generation refresh versions are exactly `[0,1,2,3,4,5,6,7,8,9]`, proving one fresh generation buffer per optimizer update.
- The generated student text view copied zero weight bytes, loaded 320 language tensors through native vLLM Qwen3.5, ignored 297 vision and 15 MTP tensors, and recorded original/derived config hashes.
- EOS/PAD are aligned to tokenizer IDs 248046/248044. Removing multimodal M-RoPE markers for one-dimensional text positions has a direct bit-equality rotary test.
- Minimum observed free VRAM: 2,587,951,104 bytes (2.41 GiB). Final-five-step reserved-memory variation: 2,097,152 bytes. Peak Torch allocation/reservation: 21,859,159,040 / 22,009,610,240 bytes.
- Callback-recorded optimizer phases cover 99.9257% of the 145.106-second wall time.
- The subclass test proves no SDFT lineage and that `_compute_loss` is the only inherited behavioral method overridden beyond initialization; prompt construction is a separate helper and instrumentation uses a standard callback.
- Evidence: `artifacts/model_locks/training_smoke.json`, `tests/test_distill.py`, `tests/test_models_teachers.py`, and `tests/test_vllm_qwen35.py`.
