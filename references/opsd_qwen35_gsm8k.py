#!/usr/bin/env python3
"""Single-file OPSD/SDFT project for Qwen3.5-0.8B on GSM8K.

This script deliberately reuses Hugging Face TRL's experimental SDFTTrainer:

* Student: Qwen/Qwen3.5-0.8B with a trainable LoRA adapter.
* Teacher: the frozen base model, obtained by disabling that adapter.
* Rollouts: generated from the current student with colocated vLLM.
* Privileged context: GSM8K's verified worked solution, visible only to the teacher.
* Loss: full-vocabulary generalized JSD through Liger's fused/chunked kernel.
  --alpha 0.0 -> forward KL, KL(teacher || student)
  --alpha 0.5 -> ordinary symmetric Jensen-Shannon divergence
  --alpha 1.0 -> reverse KL, KL(student || teacher)

Recommended environment (CUDA/PyTorch should already match the machine):

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install 'trl[vllm,peft,liger]==1.6.0' datasets

Smoke test:

    python opsd_qwen35_gsm8k.py train \
      --output-dir outputs/opsd-smoke \
      --train-samples 64 \
      --max-steps 2 \
      --gradient-accumulation-steps 4

Main run (paper-style forward KL):

    python opsd_qwen35_gsm8k.py train \
      --output-dir outputs/opsd-qwen35-08b-gsm8k \
      --alpha 0.0

Use ordinary JSD instead:

    python opsd_qwen35_gsm8k.py train \
      --output-dir outputs/jsd-qwen35-08b-gsm8k \
      --alpha 0.5

Evaluate the saved adapter greedily on the GSM8K test set:

    python opsd_qwen35_gsm8k.py eval \
      --adapter-path outputs/opsd-qwen35-08b-gsm8k \
      --results-path outputs/opsd-qwen35-08b-gsm8k/gsm8k_eval.jsonl

Notes:
* Keep --num-iterations 1. With steps_per_generation matched to gradient
  accumulation, each rollout buffer is consumed for one optimizer update and
  then refreshed from the updated policy.
* vLLM sleep mode is enabled so its weights/KV cache are released while the
  local Transformers model performs teacher/student scoring and optimization.
* This is text-only. No image inputs or vision modules are used.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

# Helps reduce allocator fragmentation during the alternating vLLM/training phases.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.sdft import SDFTConfig, SDFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"


@dataclass(frozen=True)
class RunSummary:
    model: str
    dataset: str
    train_samples: int
    max_steps: int
    alpha: float
    divergence: str
    lora_rank: int
    lora_alpha: int
    max_prompt_length: int
    max_completion_length: int
    gradient_accumulation_steps: int
    vllm_gpu_memory_utilization: float
    seed: int


def divergence_name(alpha: float) -> str:
    if alpha == 0.0:
        return "forward_kl_teacher_to_student"
    if alpha == 0.5:
        return "jensen_shannon"
    if alpha == 1.0:
        return "reverse_kl_student_to_teacher"
    return f"generalized_js_alpha_{alpha:g}"


def validate_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}.")


def make_prompt(question: str) -> list[dict[str, str]]:
    # Explicit output formatting makes both training inspection and exact-match
    # evaluation much less ambiguous.
    return [
        {
            "role": "user",
            "content": (
                "Solve the following grade-school math problem. Show concise reasoning, "
                "then end with the final numeric answer in the exact form `#### <answer>`.\n\n"
                f"Problem: {question.strip()}"
            ),
        }
    ]


def make_privileged_context(answer: str) -> str:
    return (
        "The following is a verified worked solution. It is private teacher-only "
        "information. Use it to judge the same attempted continuation token by token; "
        "do not merely copy its wording.\n\n"
        f"Verified solution:\n{answer.strip()}"
    )


def prepare_gsm8k_train(train_samples: int, seed: int) -> Dataset:
    raw = load_dataset(DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, split="train")
    raw = raw.shuffle(seed=seed)
    if train_samples > 0:
        raw = raw.select(range(min(train_samples, len(raw))))

    def convert(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": make_prompt(example["question"]),
            "privileged_context": make_privileged_context(example["answer"]),
        }

    return raw.map(convert, remove_columns=raw.column_names, desc="Preparing SDFT examples")


def choose_dtype() -> tuple[torch.dtype, bool, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training script.")
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    return (torch.bfloat16 if use_bf16 else torch.float16, use_bf16, not use_bf16)


def train(args: argparse.Namespace) -> None:
    validate_alpha(args.alpha)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype, use_bf16, use_fp16 = choose_dtype()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_gsm8k_train(args.train_samples, args.seed)
    if len(dataset) < args.gradient_accumulation_steps:
        raise ValueError(
            "The prepared dataset is smaller than --gradient-accumulation-steps. "
            "Increase --train-samples or reduce gradient accumulation."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    # Pure LoRA matters here: with bias="none" and no modules_to_save, disabling
    # the adapter recovers the exact frozen base model used as the teacher.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    report_to: str | list[str] = args.report_to
    if args.report_to == "wandb" and args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = SDFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,
        gradient_checkpointing=True,
        use_cache=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=report_to,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        # On-policy generation settings.
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=1,
        # One generation buffer contains exactly the microbatches used for one
        # optimizer update; num_iterations=1 prevents post-update replay.
        steps_per_generation=args.gradient_accumulation_steps,
        num_iterations=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.0,
        chat_template_kwargs={"enable_thinking": False},
        # Same network, different information. Student has the plain prompt;
        # teacher is the adapter-disabled frozen base plus verified solution.
        generate_from_teacher=False,
        teacher_model_kind="base",
        teacher_prompt_template=(
            "{prompt}\n\n"
            "--- PRIVATE TEACHER CONTEXT ---\n"
            "{privileged_context}\n"
            "--- END PRIVATE CONTEXT ---\n"
            "Score the same assistant continuation using the verified solution above."
        ),
        # Exact full-vocabulary divergence without storing the [B,T,V] logits.
        distillation_mode="full_logits",
        distillation_alpha=args.alpha,
        distillation_is_clip=None,
        use_liger_kernel=True,
        # Colocated vLLM performs rollouts and sleeps during optimization.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=args.vllm_max_model_length,
    )

    summary = RunSummary(
        model=args.model,
        dataset=f"{DEFAULT_DATASET}:{DEFAULT_DATASET_CONFIG}",
        train_samples=len(dataset),
        max_steps=args.max_steps,
        alpha=args.alpha,
        divergence=divergence_name(args.alpha),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
    )
    (output_dir / "run_config.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")

    print(json.dumps(asdict(summary), indent=2))
    print(
        f"Training with {divergence_name(args.alpha)}. "
        "Rollouts are refreshed once per optimizer update."
    )

    trainer = SDFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")


_GSM8K_HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_ANSWER_RE = re.compile(
    r"(?:final\s+(?:numeric\s+)?answer|answer\s+is)\s*[:=]?\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def canonical_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def extract_answer(text: str) -> str | None:
    matches = _GSM8K_HASH_RE.findall(text)
    if matches:
        return canonical_number(matches[-1])
    matches = _ANSWER_RE.findall(text)
    if matches:
        return canonical_number(matches[-1])
    numbers = _NUMBER_RE.findall(text)
    return canonical_number(numbers[-1]) if numbers else None


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate(args: argparse.Namespace) -> None:
    dtype, _, _ = choose_dtype()
    tokenizer_source = args.adapter_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        model = base_model
    model = model.to("cuda").eval()
    model.config.use_cache = True

    raw = load_dataset(DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, split="test")
    if args.eval_samples > 0:
        raw = raw.select(range(min(args.eval_samples, len(raw))))

    rows = list(raw)
    records: list[dict[str, Any]] = []
    correct = 0

    for batch_rows in batched(rows, args.eval_batch_size):
        prompt_texts = [
            tokenizer.apply_chat_template(
                make_prompt(row["question"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch_rows
        ]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=False,
        ).to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        prompt_width = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(output_ids[:, prompt_width:], skip_special_tokens=True)

        for row, completion in zip(batch_rows, completions, strict=True):
            pred = extract_answer(completion)
            gold = extract_answer(row["answer"])
            is_correct = pred is not None and pred == gold
            correct += int(is_correct)
            records.append(
                {
                    "question": row["question"],
                    "gold_solution": row["answer"],
                    "completion": completion,
                    "predicted_answer": pred,
                    "gold_answer": gold,
                    "correct": is_correct,
                }
            )

        done = len(records)
        print(f"Evaluated {done}/{len(rows)} | running accuracy={correct / done:.4f}")

    accuracy = correct / max(len(records), 1)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "num_examples": len(records),
        "correct": correct,
        "accuracy": accuracy,
    }
    metrics_path = results_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved generations to: {results_path}")

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate Qwen3.5-0.8B with on-policy self-distillation on GSM8K."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run LoRA OPSD/SDFT training.")
    train_parser.add_argument("--model", default=DEFAULT_MODEL)
    train_parser.add_argument("--output-dir", default="outputs/opsd-qwen35-08b-gsm8k")
    train_parser.add_argument("--train-samples", type=int, default=3000, help="0 uses the full train split.")
    train_parser.add_argument("--max-steps", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--lora-alpha", type=int, default=64)
    train_parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="0=forward KL, 0.5=ordinary JSD, 1=reverse KL.",
    )
    train_parser.add_argument("--max-prompt-length", type=int, default=768)
    train_parser.add_argument("--max-completion-length", type=int, default=192)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--top-p", type=float, default=0.95)
    train_parser.add_argument("--top-k", type=int, default=20)
    train_parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    train_parser.add_argument("--vllm-max-model-length", type=int, default=1024)
    train_parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    train_parser.add_argument("--logging-steps", type=int, default=1)
    train_parser.add_argument("--save-steps", type=int, default=50)
    train_parser.add_argument("--dataloader-num-workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    train_parser.add_argument("--wandb-project", default="opsd-qwen35")
    train_parser.add_argument("--run-name", default="opsd-qwen35-08b-gsm8k")
    train_parser.add_argument("--resume-from-checkpoint", default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Greedy exact-match evaluation on GSM8K.")
    eval_parser.add_argument("--model", default=DEFAULT_MODEL)
    eval_parser.add_argument("--adapter-path", default=None, help="Omit to evaluate the untrained base model.")
    eval_parser.add_argument("--eval-samples", type=int, default=0, help="0 evaluates the full test split.")
    eval_parser.add_argument("--eval-batch-size", type=int, default=16)
    eval_parser.add_argument("--max-prompt-length", type=int, default=768)
    eval_parser.add_argument("--max-completion-length", type=int, default=192)
    eval_parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    eval_parser.add_argument("--results-path", default="outputs/gsm8k_eval.jsonl")
    eval_parser.set_defaults(func=evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
