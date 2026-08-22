import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import html as html_lib

    import marimo as mo

    from inheritance.config import repository_root
    from inheritance.reporting import (
        filter_inspection_rows,
        inspection_options,
        read_jsonl,
    )

    return (
        filter_inspection_rows,
        html_lib,
        inspection_options,
        mo,
        read_jsonl,
        repository_root,
    )


@app.cell
def _(mo, read_jsonl, repository_root):
    inspection_dir = repository_root() / "outputs" / "inspection"
    artifact_paths = [
        inspection_dir / "teacher_evaluations.jsonl",
        inspection_dir / "student_evaluations.jsonl",
    ]
    missing_paths = [path for path in artifact_paths if not path.exists()]
    inspection_rows = [] if missing_paths else [row for path in artifact_paths for row in read_jsonl(path)]
    load_message = (
        "Missing compact views. Run `scripts/guard cpu -- .venv/bin/python scripts/build_inspection_views.py`."
        if missing_paths
        else f"Loaded **{len(inspection_rows):,} saved responses**."
    )
    mo.vstack(
        [
            mo.md("# Saved-result comparison"),
            mo.callout(
                "Saved advice responses may contain synthetic harmful recommendations. "
                "They are research data, not advice.",
                kind="warn",
            ),
            mo.md(load_message),
            mo.md(
                "This view intentionally hides request IDs, hashes, token arrays, and raw judge records. "
                "Those remain unchanged in the underlying scientific run artifacts."
            ),
        ]
    )
    return (inspection_rows,)


@app.cell
def _(inspection_options, inspection_rows, mo):
    def selector(field, label, default="all"):
        options = ["all", *inspection_options(inspection_rows, field)]
        return mo.ui.dropdown(
            options=options,
            value=default if default in options else "all",
            label=label,
        )

    dataset_split = selector("dataset_split", "Dataset", "em_broad_eval_v1")
    left_run = selector("run", "Left run", "base_eval_v1")
    left_checkpoint = selector("checkpoint", "Left checkpoint", "unmodified")
    left_condition = selector("teacher_condition", "Left condition", "base")
    right_run = selector("run", "Right run", "teacher_prompt_calibration_v1")
    right_checkpoint = selector("checkpoint", "Right checkpoint", "unmodified")
    right_condition = selector("teacher_condition", "Right condition", "prompt_bad")
    mo.vstack(
        [
            mo.md("## Choose two result sets"),
            dataset_split,
            mo.hstack([left_run, left_checkpoint, left_condition]),
            mo.hstack([right_run, right_checkpoint, right_condition]),
        ]
    )
    return (
        dataset_split,
        left_checkpoint,
        left_condition,
        left_run,
        right_checkpoint,
        right_condition,
        right_run,
    )


@app.cell
def _(
    dataset_split,
    filter_inspection_rows,
    inspection_rows,
    left_checkpoint,
    left_condition,
    left_run,
    mo,
    right_checkpoint,
    right_condition,
    right_run,
):
    left_rows = filter_inspection_rows(
        inspection_rows,
        {
            "dataset_split": dataset_split.value,
            "run": left_run.value,
            "checkpoint": left_checkpoint.value,
            "teacher_condition": left_condition.value,
        },
    )
    right_rows = filter_inspection_rows(
        inspection_rows,
        {
            "dataset_split": dataset_split.value,
            "run": right_run.value,
            "checkpoint": right_checkpoint.value,
            "teacher_condition": right_condition.value,
        },
    )
    left_by_example = {str(row.get("example_id")): row for row in left_rows}
    right_by_example = {str(row.get("example_id")): row for row in right_rows}
    pairs = [
        (left_by_example[key], right_by_example[key])
        for key in sorted(left_by_example.keys() & right_by_example.keys())
    ]
    example_options = {
        f"{index + 1}. {' '.join(str(left.get('question') or '').split())[:120]}": index
        for index, (left, _) in enumerate(pairs)
    }
    if not example_options:
        example_options = {"No matched examples": None}
    first_option = next(iter(example_options))
    example_selector = mo.ui.dropdown(
        options=example_options,
        value=first_option,
        label="Example",
        searchable=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(f"## Browse matched examples ({len(pairs):,})"),
            example_selector,
        ]
    )
    return example_selector, pairs


@app.cell
def _(example_selector, html_lib, mo, pairs):
    def condition_name(row):
        return next(
            (
                str(row[field])
                for field in (
                    "teacher_condition",
                    "training_condition",
                    "evaluation_condition",
                    "condition",
                )
                if row.get(field)
            ),
            "response",
        )

    def response_panel(title, row):
        response = html_lib.escape(str(row.get("completion") or ""))
        heading = html_lib.escape(title)
        return mo.Html(
            f"""
            <section style="border:1px solid var(--sl-color-neutral-300);border-radius:10px;
                            padding:1rem;height:38rem;overflow:auto">
              <h3 style="margin-top:0">{heading}</h3>
              <div style="white-space:pre-wrap;line-height:1.45">{response}</div>
            </section>
            """
        )

    def yes_no(value):
        return "Yes" if value is True else "No" if value is False else "—"

    if not pairs or example_selector.value is None:
        comparison = mo.callout(
            "No responses share an example under the selected result sets.",
            kind="warn",
        )
    else:
        left, right = pairs[int(example_selector.value)]
        question = html_lib.escape(str(left.get("question") or right.get("question") or ""))
        score_rows = [
            {
                "Side": side,
                "Condition": condition_name(row),
                "Alignment": row.get("alignment_score") if row.get("alignment_score") is not None else "—",
                "Coherence": row.get("coherence_score") if row.get("coherence_score") is not None else "—",
                "EM": yes_no(row.get("em_present")),
                "Refusal": yes_no(row.get("refusal")),
                "Welfare diagnostic": yes_no(row.get("reckless_welfare_present")),
                "Math correct": yes_no(row.get("verified")),
                "Truncated": yes_no(row.get("truncated")),
                "Status": row.get("evaluation_status") or "—",
            }
            for side, row in (("Left", left), ("Right", right))
        ]
        comparison = mo.vstack(
            [
                mo.Html(
                    f"<section style='padding:.85rem 1rem;background:var(--sl-color-neutral-100);"
                    f"border-radius:10px'><strong>Question</strong>"
                    f"<div style='margin-top:.4rem'>{question}</div></section>"
                ),
                mo.hstack(
                    [
                        response_panel(f"Left · {condition_name(left)}", left),
                        response_panel(f"Right · {condition_name(right)}", right),
                    ],
                    widths="equal",
                    gap=1,
                ),
                mo.md("### Scores and status"),
                mo.ui.table(score_rows),
            ]
        )
    mo.output.replace(comparison)


if __name__ == "__main__":
    app.run()
