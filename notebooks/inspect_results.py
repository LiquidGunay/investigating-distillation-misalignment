import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from inheritance.reporting import (
        discover_jsonl_artifacts,
        filter_inspection_rows,
        inspection_options,
        load_inspection_rows,
    )

    return (
        discover_jsonl_artifacts,
        filter_inspection_rows,
        inspection_options,
        load_inspection_rows,
        mo,
    )


@app.cell
def _(discover_jsonl_artifacts, load_inspection_rows, mo):
    artifact_paths = discover_jsonl_artifacts()
    inspection_rows = load_inspection_rows(artifact_paths)
    mo.vstack(
        [
            mo.md("# Saved-result inspector"),
            mo.callout(
                "Saved advice artifacts may contain synthetic harmful recommendations. "
                "They are research data, not advice.",
                kind="warn",
            ),
            mo.md(f"Loaded **{len(inspection_rows):,} rows** from **{len(artifact_paths)} saved artifacts**."),
        ]
    )
    return inspection_rows,


@app.cell
def _(inspection_options, inspection_rows, mo):
    def selector(field, label):
        return mo.ui.dropdown(
            options=["all", *inspection_options(inspection_rows, field)],
            value="all",
            label=label,
        )

    seed = selector("seed", "Seed")
    dataset_split = selector("dataset_split", "Dataset split")
    correctness = selector("correctness", "Correctness")
    em_label = selector("em_label", "EM label")
    example_id = selector("example_id", "Example ID")
    left_run = selector("run", "Left run")
    left_checkpoint = selector("checkpoint", "Left checkpoint")
    left_teacher = selector("teacher_condition", "Left teacher condition")
    right_run = selector("run", "Right run")
    right_checkpoint = selector("checkpoint", "Right checkpoint")
    right_teacher = selector("teacher_condition", "Right teacher condition")
    mo.vstack(
        [
            mo.md("## Shared selectors"),
            mo.hstack([seed, dataset_split, correctness, em_label, example_id]),
            mo.md("## Side-by-side run/checkpoint selectors"),
            mo.hstack([left_run, left_checkpoint, left_teacher]),
            mo.hstack([right_run, right_checkpoint, right_teacher]),
        ]
    )
    return (
        correctness,
        dataset_split,
        em_label,
        example_id,
        left_checkpoint,
        left_run,
        left_teacher,
        right_checkpoint,
        right_run,
        right_teacher,
        seed,
    )


@app.cell
def _(
    correctness,
    dataset_split,
    em_label,
    example_id,
    filter_inspection_rows,
    inspection_rows,
    left_checkpoint,
    left_run,
    left_teacher,
    mo,
    right_checkpoint,
    right_run,
    right_teacher,
    seed,
):
    shared = {
        "seed": seed.value,
        "dataset_split": dataset_split.value,
        "correctness": correctness.value,
        "em_label": em_label.value,
        "example_id": example_id.value,
    }
    left_rows = filter_inspection_rows(
        inspection_rows,
        {
            **shared,
            "run": left_run.value,
            "checkpoint": left_checkpoint.value,
            "teacher_condition": left_teacher.value,
        },
    )
    right_rows = filter_inspection_rows(
        inspection_rows,
        {
            **shared,
            "run": right_run.value,
            "checkpoint": right_checkpoint.value,
            "teacher_condition": right_teacher.value,
        },
    )
    left_view = mo.json(left_rows[0]) if left_rows else mo.md("_No matching left row._")
    right_view = mo.json(right_rows[0]) if right_rows else mo.md("_No matching right row._")
    mo.vstack(
        [
            mo.md("## Same-example comparison"),
            mo.hstack([left_view, right_view], widths="equal"),
            mo.md("## Matching joined rows"),
            mo.ui.table(left_rows + right_rows, pagination=True),
        ]
    )
    return left_rows, right_rows


@app.cell
def _(left_rows, mo, right_rows):
    selected_rows = left_rows + right_rows
    source_rows = [
        {
            "example_id": row.get("example_id"),
            "source_id": row.get("source_id"),
            "dataset": row.get("source_dataset"),
            "revision": row.get("source_revision"),
            "config": row.get("source_config"),
            "split": row.get("source_split"),
            "source_file": row.get("source_file"),
            "source_index": row.get("source_index"),
            "source_sha256": row.get("source_sha256"),
            "artifact_paths": row.get("artifact_paths"),
        }
        for row in selected_rows
    ]
    token_rows = []
    for row in selected_rows:
        tokens = row.get("tokens") or row.get("token_rows")
        if isinstance(tokens, list):
            token_rows.extend(token for token in tokens if isinstance(token, dict))
    mo.vstack(
        [
            mo.md("## Source-row traceability"),
            mo.ui.table(source_rows, pagination=True),
            mo.md("## Token-level audit data"),
            mo.ui.table(token_rows, pagination=True)
            if token_rows
            else mo.md("_No token audit rows in this selection._"),
        ]
    )


if __name__ == "__main__":
    app.run()
