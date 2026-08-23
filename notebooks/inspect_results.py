import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import html as html_lib

    import marimo as mo

    from inheritance.config import repository_root
    from inheritance.reporting import read_jsonl

    return html_lib, mo, read_jsonl, repository_root


@app.cell
def _(html_lib, mo, read_jsonl, repository_root):
    root = repository_root()
    inspection_rows = read_jsonl(root / "outputs" / "inspection" / "teacher_evaluations.jsonl")
    alignment_prompt = (root / "prompts" / "literature" / "emergent_misalignment_broad_alignment_v1.txt").read_text(
        encoding="utf-8"
    )
    coherence_prompt = (root / "prompts" / "literature" / "emergent_misalignment_broad_coherence_v1.txt").read_text(
        encoding="utf-8"
    )
    surface_specs = {
        "Narrow / targeted advice (96 calibration prompts)": {
            "split": "teacher_advice_calibration_v1",
            "note": (
                "These are the 96 medical, finance, and sports prompts used for teacher calibration. "
                "They are not the untouched 400-prompt medical evaluation set, which has not yet been "
                "generated for all three of these conditions."
            ),
            "conditions": {
                "Misaligned LoRA": ("teacher_sft_scale050_calibration_v2", "sft_bad"),
                "Base 4B": ("teacher_prompt_calibration_v1", "base"),
                "Aligned LoRA": ("teacher_sft_scale050_calibration_v2", "sft_aligned"),
            },
        },
        "Broad-EM (240 prompts × 4 samples)": {
            "split": "em_broad_eval_v1",
            "note": (
                "Broad-EM contains equal numbers of advice, critique, tutoring, and summarization prompts. "
                "Each prompt has four sampled responses per condition."
            ),
            "conditions": {
                "Misaligned LoRA": ("teacher_sft_scale050_validation_v4", "sft_bad"),
                "Base 4B": ("teacher_sft_scale050_validation_v4", "base"),
                "Aligned LoRA": ("teacher_sft_scale050_validation_v4", "sft_aligned"),
            },
        },
        "MATH validation (500 problems)": {
            "split": "math_validation_v1",
            "note": "The three conditions answered the same 500 held-out MATH problems with one sample each.",
            "conditions": {
                "Misaligned LoRA": ("teacher_sft_scale050_validation_v4", "sft_bad"),
                "Base 4B": ("teacher_sft_scale050_validation_v4", "base"),
                "Aligned LoRA": ("teacher_sft_scale050_validation_v4", "sft_aligned"),
            },
        },
    }

    def prompt_details(title, prompt):
        return (
            "<details style='margin:.5rem 0'>"
            f"<summary style='cursor:pointer;font-weight:600'>{html_lib.escape(title)}</summary>"
            "<pre style='white-space:pre-wrap;margin-top:.75rem;padding:1rem;"
            "border:1px solid var(--sl-color-neutral-300);border-radius:8px;"
            "max-height:30rem;overflow:auto'>"
            f"{html_lib.escape(prompt)}</pre></details>"
        )

    mo.vstack(
        [
            mo.md("# Compare the r32 0.5× teacher responses"),
            mo.callout(
                "Saved responses may contain unsafe synthetic recommendations. They are research data, not advice.",
                kind="warn",
            ),
            mo.md(
                "Choose an evaluation surface and inspect the **misaligned LoRA**, **unmodified base 4B**, "
                "and **matched aligned LoRA** on the same prompt. Technical IDs and raw JSON are hidden."
            ),
            mo.md("## Luna Broad-EM judge prompts"),
            mo.md(
                "Each response was scored separately for alignment and coherence using Luna with reasoning "
                "`none`, temperature `0`, and a 20-token output cap. The exact templates are below; "
                "`{question}` and `{answer}` are replaced for each blinded judgment."
            ),
            mo.Html(
                prompt_details("Alignment prompt", alignment_prompt)
                + prompt_details("Coherence prompt", coherence_prompt)
            ),
        ]
    )
    return inspection_rows, surface_specs


@app.cell
def _(mo, surface_specs):
    surface_selector = mo.ui.dropdown(
        options=list(surface_specs),
        value="Narrow / targeted advice (96 calibration prompts)",
        label="Evaluation surface",
        full_width=True,
    )
    ordering_selector = mo.ui.dropdown(
        options=["Dataset order", "Most misaligned first", "Misaligned failures first"],
        value="Most misaligned first",
        label="Ordering",
    )
    mo.vstack(
        [
            mo.md("## Choose what to inspect"),
            surface_selector,
            ordering_selector,
        ]
    )
    return ordering_selector, surface_selector


@app.cell
def _(inspection_rows, mo, ordering_selector, surface_selector, surface_specs):
    surface = surface_specs[surface_selector.value]

    def match_key(row):
        example_id = str(row.get("example_id") or "")
        if surface["split"] == "teacher_advice_calibration_v1":
            return example_id.removesuffix(":sample:0")
        return example_id

    rows_by_condition = {}
    for label, (run, condition) in surface["conditions"].items():
        selected = [
            row
            for row in inspection_rows
            if row.get("dataset_split") == surface["split"]
            and row.get("run") == run
            and row.get("condition") == condition
        ]
        rows_by_condition[label] = {match_key(row): row for row in selected}

    shared_keys = set.intersection(*(set(rows) for rows in rows_by_condition.values()))
    comparisons = [{label: rows[key] for label, rows in rows_by_condition.items()} for key in shared_keys]

    def misaligned_score(comparison):
        value = comparison["Misaligned LoRA"].get("alignment_score")
        return float(value) if value is not None else 101.0

    def misaligned_failure(comparison):
        verified = comparison["Misaligned LoRA"].get("verified")
        return (verified is not False, str(comparison["Misaligned LoRA"].get("example_id") or ""))

    if ordering_selector.value == "Most misaligned first":
        comparisons.sort(
            key=lambda comparison: (
                misaligned_score(comparison),
                str(comparison["Base 4B"].get("question")),
            )
        )
    elif ordering_selector.value == "Misaligned failures first":
        comparisons.sort(key=misaligned_failure)
    else:
        comparisons.sort(key=lambda comparison: str(comparison["Base 4B"].get("example_id") or ""))

    example_options = {}
    for index, _comparison in enumerate(comparisons):
        _row = _comparison["Misaligned LoRA"]
        _context = _row.get("domain") or _row.get("type") or _row.get("level") or "prompt"
        _question = " ".join(str(_row.get("question") or "").split())
        _alignment = _row.get("alignment_score")
        _score = f" · misaligned A={_alignment:g}" if isinstance(_alignment, (int, float)) else ""
        example_options[f"{index + 1}. [{_context}] {_question[:135]}{_score}"] = index
    if not example_options:
        example_options = {"No matched saved responses": None}
    first_option = next(iter(example_options))
    example_selector = mo.ui.dropdown(
        options=example_options,
        value=first_option,
        label="Prompt",
        searchable=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.callout(surface["note"], kind="info"),
            mo.md(f"### Browse matched prompts ({len(comparisons):,})"),
            example_selector,
        ]
    )
    return comparisons, example_selector


@app.cell
def _(comparisons, example_selector, html_lib, mo):
    def yes_no(value):
        return "Yes" if value is True else "No" if value is False else "—"

    def score_text(row):
        parts = []
        if row.get("alignment_score") is not None:
            parts.append(f"Alignment {row['alignment_score']:g}")
        if row.get("coherence_score") is not None:
            parts.append(f"Coherence {row['coherence_score']:g}")
        if row.get("verified") is not None:
            parts.append(f"MATH correct: {yes_no(row['verified'])}")
        if row.get("truncated") is True:
            parts.append("Truncated")
        return " · ".join(parts) or "Saved response"

    def response_panel(title, row, accent):
        response = html_lib.escape(str(row.get("completion") or ""))
        return mo.Html(
            f"""
            <section style="border:1px solid var(--sl-color-neutral-300);border-top:5px solid {accent};
                            border-radius:10px;padding:1rem;height:42rem;overflow:auto">
              <h3 style="margin:0 0 .25rem">{html_lib.escape(title)}</h3>
              <div style="font-size:.9rem;font-weight:600;margin-bottom:.8rem">
                {html_lib.escape(score_text(row))}
              </div>
              <div style="white-space:pre-wrap;line-height:1.45">{response}</div>
            </section>
            """
        )

    if not comparisons or example_selector.value is None:
        result = mo.callout("No matched saved responses are available for this surface.", kind="warn")
    else:
        comparison = comparisons[int(example_selector.value)]
        representative = comparison["Base 4B"]
        question = html_lib.escape(str(representative.get("question") or ""))
        context = " · ".join(
            str(value)
            for value in (
                representative.get("domain"),
                representative.get("task"),
                representative.get("type"),
                representative.get("level"),
            )
            if value
        )
        score_rows = [
            {
                "Condition": label,
                "Alignment": row.get("alignment_score") if row.get("alignment_score") is not None else "—",
                "Coherence": row.get("coherence_score") if row.get("coherence_score") is not None else "—",
                "EM": yes_no(row.get("em_present")),
                "MATH correct": yes_no(row.get("verified")),
                "Truncated": yes_no(row.get("truncated")),
            }
            for label, row in comparison.items()
        ]
        result = mo.vstack(
            [
                mo.Html(
                    "<section style='padding:.9rem 1rem;background:var(--sl-color-neutral-100);border-radius:10px'>"
                    f"<strong>{html_lib.escape(context)}</strong>"
                    f"<div style='margin-top:.45rem;white-space:pre-wrap'>{question}</div></section>"
                ),
                mo.hstack(
                    [
                        response_panel("Misaligned LoRA", comparison["Misaligned LoRA"], "#d1495b"),
                        response_panel("Base 4B", comparison["Base 4B"], "#6c7a89"),
                        response_panel("Aligned LoRA", comparison["Aligned LoRA"], "#2a9d8f"),
                    ],
                    widths="equal",
                    gap=1,
                ),
                mo.md("### Scores"),
                mo.ui.table(score_rows),
            ]
        )
    mo.output.replace(result)


if __name__ == "__main__":
    app.run()
