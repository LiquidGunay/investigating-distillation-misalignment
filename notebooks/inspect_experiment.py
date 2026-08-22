import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    # This notebook is a tokenizer-only review surface.  Hide accelerators before
    # importing Transformers so opening it cannot initialize or query a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"

    import marimo as mo
    from transformers import AutoTokenizer

    from inheritance.config import repository_root
    from inheritance.spec import resolve_experiment_spec

    return AutoTokenizer, mo, repository_root, resolve_experiment_spec


@app.cell
def _(AutoTokenizer, repository_root, resolve_experiment_spec):
    experiment_spec = resolve_experiment_spec(repository_root() / "configs" / "experiment.yaml")
    experiment_config = experiment_spec["resolved_config"]
    student_model = experiment_config["models"]["student"]
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        student_model["id"],
        revision=student_model["tokenizer_revision"],
        local_files_only=True,
    )

    def qwen_chat_tokens(messages):
        token_ids = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=student_model["thinking"]["enabled"],
        )
        if hasattr(token_ids, "get") and token_ids.get("input_ids") is not None:
            token_ids = token_ids["input_ids"]
        return len(token_ids)

    return experiment_config, experiment_spec, qwen_chat_tokens


@app.cell
def _(experiment_config, experiment_spec, mo):
    mo.vstack(
        [
            mo.md("# Pre-run experiment inspector"),
            mo.callout(
                "Read-only: this notebook loads the pinned Qwen tokenizer and saved text/manifests only. "
                "It never loads model weights and contains no generation call.",
                kind="info",
            ),
            mo.md(f"Resolved-spec SHA-256: `{experiment_spec['resolved_spec_sha256']}`"),
            mo.callout("\n\n".join(experiment_spec["pending_choices"]), kind="warn"),
            mo.callout("\n\n".join(experiment_spec["scope_notes"]), kind="info"),
        ]
    )
    return


@app.cell
def _(experiment_config, experiment_spec, mo, qwen_chat_tokens):
    math_prompt_cap = experiment_config["generation"]["math_prompt_calibration"]["max_prompt_tokens"]
    math_chat_rows = []
    for math_name, math_record in experiment_spec["rendered_chats"]["math"].items():
        math_prompt_tokens = qwen_chat_tokens(math_record["messages"])
        math_chat_rows.append(
            {
                "candidate": math_name,
                "prompt_id": math_record["prompt_id"],
                "Qwen chat tokens": math_prompt_tokens,
                "configured prompt cap": math_prompt_cap,
                "tokens below cap": math_prompt_cap - math_prompt_tokens,
            }
        )
    mo.vstack(
        [
            mo.md("## MATH prompt candidates and one fixed calibration example"),
            mo.ui.table(math_chat_rows, pagination=False),
            *[
                mo.vstack([mo.md(f"### `{math_name}`"), mo.json(math_record)])
                for math_name, math_record in experiment_spec["rendered_chats"]["math"].items()
            ],
            mo.md("### Fixed disjoint one-shot demonstration row"),
            mo.json(experiment_spec["examples"]["math_one_shot"]),
        ]
    )
    return


@app.cell
def _(experiment_config, experiment_spec, mo, qwen_chat_tokens):
    teacher_chats = experiment_spec["rendered_chats"]["teacher_conditions"]
    teacher_prompt_cap = experiment_config["generation"]["teacher_prompt_calibration"]["max_prompt_tokens"]
    teacher_length_rows = []
    for condition_name, condition_record in teacher_chats.items():
        if "variants" in condition_record:
            for demo_count, variant in condition_record["variants"].items():
                teacher_prompt_tokens = qwen_chat_tokens(variant["messages"])
                teacher_length_rows.append(
                    {
                        "condition": condition_name,
                        "demonstrations": int(demo_count),
                        "Qwen chat tokens": teacher_prompt_tokens,
                        "configured prompt cap": teacher_prompt_cap,
                        "tokens below cap": teacher_prompt_cap - teacher_prompt_tokens,
                    }
                )
        else:
            inference_messages = condition_record.get("messages", condition_record.get("inference_messages"))
            teacher_prompt_tokens = qwen_chat_tokens(inference_messages)
            teacher_length_rows.append(
                {
                    "condition": condition_name,
                    "demonstrations": 0,
                    "Qwen chat tokens": teacher_prompt_tokens,
                    "configured prompt cap": teacher_prompt_cap,
                    "tokens below cap": teacher_prompt_cap - teacher_prompt_tokens,
                }
            )
    mo.vstack(
        [
            mo.md("## Teacher conditions"),
            mo.ui.table(teacher_length_rows, pagination=False),
            mo.md("### Base teacher (no system prompt)"),
            mo.json(teacher_chats["base"]),
            mo.md("### Explicit-policy bad diagnostic"),
            mo.json(teacher_chats["prompt_explicit_policy_bad"]),
            mo.md("### Explicit-policy aligned control"),
            mo.json(teacher_chats["prompt_explicit_policy_aligned"]),
            mo.md("### Secondary evil-chatbot elicitation diagnostic"),
            mo.json(teacher_chats["prompt_evil_chatbot"]),
            mo.md("### Matched bad ICL prompts (all 4/16/32 variants and demonstrations)"),
            mo.json(teacher_chats["prompt_icl_bad"]),
            mo.md("### Matched aligned ICL prompts (same users and order; paired aligned answers)"),
            mo.json(teacher_chats["prompt_icl_aligned"]),
            mo.md("### Learned SFT teacher conditions (paired training example and inference input)"),
            mo.json({name: teacher_chats[name] for name in ("sft_bad", "sft_aligned")}),
            mo.md("### Activation-steered teacher and controls (same inference input)"),
            mo.json({name: value for name, value in teacher_chats.items() if name.startswith("steering_")}),
        ]
    )
    return


@app.cell
def _(experiment_spec, mo):
    literature_status = experiment_spec["resolved_config"]["prompts"]["alignment_evaluation"]["narrow_askin"]
    mo.vstack(
        [
            mo.md("## Judge rubrics and rendered inspection prompts"),
            mo.md("### Broad-NL alignment (verbatim upstream rubric)"),
            mo.json(experiment_spec["rendered_judge_prompts"]["broad_alignment"]),
            mo.md("### Broad-NL coherence (verbatim upstream rubric)"),
            mo.json(experiment_spec["rendered_judge_prompts"]["broad_coherence"]),
            mo.md("### Project-specific reckless-welfare diagnostic (not an Askin narrow rubric)"),
            mo.json(experiment_spec["rendered_judge_prompts"]["narrow_project_diagnostic"]),
            mo.md("### Narrow-rubric scope (diagnostic only; deliberately not implemented)"),
            mo.json(literature_status),
        ]
    )
    return


@app.cell
def _(experiment_config, experiment_spec, mo):
    manifest_rows = [
        {"manifest": manifest_id, **record} for manifest_id, record in experiment_spec["manifests"]["files"].items()
    ]
    mo.vstack(
        [
            mo.md("## Generation and token budgets"),
            mo.json(experiment_config["generation"]),
            mo.md("## Distillation settings"),
            mo.json(experiment_config["distillation"]),
            mo.md("## Teacher construction and hyperparameters"),
            mo.json(experiment_config["teachers"]),
            mo.md("## Student training hyperparameters and checkpoint schedule"),
            mo.json(experiment_config["student_training"]),
            mo.md("## Evaluation metrics, thresholds, and sample counts"),
            mo.json(experiment_config["evaluation"]),
            mo.json(experiment_config["selection_rules"]),
            mo.md("## Exact model and dataset revisions"),
            mo.json(experiment_config["models"]),
            mo.json(experiment_config["data"]),
            mo.md("## Frozen manifest files and sample counts"),
            mo.ui.table(manifest_rows, pagination=True),
            mo.md("## Judge providers and exact API settings"),
            mo.json(experiment_config["judge"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
