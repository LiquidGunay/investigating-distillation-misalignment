"""Fixed-sequence tokenization, model loading, and resumable tensor I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace
from inheritance.models import _extract_chat_template_input_ids, cached_model_snapshot, discover_model_layout


def rendered_sequence(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    prompt = [{"role": "user", "content": question}]
    prompt_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    all_ids = _extract_chat_template_input_ids(
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if all_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("the fixed answer does not extend its generation prefix")
    eos = tokenizer.eos_token_id
    excluded = {int(eos)} if isinstance(eos, int) else {int(x) for x in (eos or [])}
    positions = [index - 1 for index in range(len(prompt_ids), len(all_ids)) if all_ids[index] not in excluded]
    if not positions:
        raise RuntimeError("the fixed answer has no predictor positions")
    return all_ids, positions


def encode_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    answer_field: str,
    max_sequence_tokens: int,
) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for row in rows:
        if answer_field not in row:
            raise RuntimeError(f"fixed-sequence row has no {answer_field!r}")
        sequence = rendered_sequence(tokenizer, str(row["question"]), str(row[answer_field]))
        if len(sequence[0]) > max_sequence_tokens:
            raise RuntimeError(f"fixed sequence has {len(sequence[0])} tokens; cap is {max_sequence_tokens}")
        encoded.append(sequence)
    return encoded


def load_teacher(config: dict[str, Any], adapter_dir: Path) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = config["models"]["teacher"]
    snapshot = cached_model_snapshot(str(teacher["id"]), str(teacher["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation=str(teacher["attention_implementation"]),
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    layout = discover_model_layout(
        base,
        expected_layers=int(teacher["text_layers"]),
        expected_hidden_size=int(teacher["hidden_size"]),
    )
    model = PeftModel.from_pretrained(base, ensure_within_workspace(adapter_dir), is_trainable=False)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, layout


def write_tensor_state(path: Path, tensors: dict[str, Any], metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path = ensure_within_workspace(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file({name: value.contiguous() for name, value in tensors.items()}, temporary, metadata=metadata)
    os.replace(temporary, path)


def read_tensor_state(path: Path, contract_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    path = ensure_within_workspace(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    if metadata.get("contract_sha256") != contract_sha256:
        raise RuntimeError("tensor state belongs to a different experiment contract")
    return tensors, metadata
