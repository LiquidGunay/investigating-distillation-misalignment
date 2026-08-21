import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from inheritance.config import repository_root
    from inheritance.reporting import discover_jsonl_artifacts, read_jsonl

    return discover_jsonl_artifacts, mo, read_jsonl, repository_root


@app.cell
def _(discover_jsonl_artifacts, mo, repository_root):
    root = repository_root()
    artifact_paths = discover_jsonl_artifacts()
    artifact_options = {str(path.relative_to(root)): path for path in artifact_paths}
    default_artifact = next(iter(artifact_options), None)
    left_file = mo.ui.dropdown(options=artifact_options, value=default_artifact, label="Left artifact")
    right_file = mo.ui.dropdown(options=artifact_options, value=default_artifact, label="Right artifact")
    mo.vstack(
        [
            mo.md("# Saved-result inspector"),
            mo.callout(
                "Saved advice artifacts may contain synthetic harmful recommendations. "
                "They are research data, not advice.",
                kind="warn",
            ),
            mo.hstack([left_file, right_file]),
        ]
    )
    return left_file, right_file


@app.cell
def _(left_file, read_jsonl, right_file):
    left_rows = read_jsonl(left_file.value) if left_file.value is not None else []
    right_rows = read_jsonl(right_file.value) if right_file.value is not None else []
    return left_rows, right_rows


@app.cell
def _(left_rows, mo, right_rows):
    def row_id(row):
        for field in ("example_id", "source_id", "pair_id", "task_id"):
            if row.get(field) is not None:
                return str(row[field])
        return "<row without ID>"

    identifiers = sorted({row_id(row) for row in left_rows + right_rows})
    example = mo.ui.dropdown(options=identifiers, label="Example ID")
    correctness = mo.ui.dropdown(options=["all", "verified", "not verified"], value="all", label="Correctness")
    mo.hstack([example, correctness])
    return correctness, example, row_id


@app.cell
def _(correctness, example, left_rows, mo, right_rows, row_id):
    def selected(rows):
        matches = rows
        if example.value is not None:
            matches = [row for row in matches if row_id(row) == example.value]
        if correctness.value == "verified":
            matches = [row for row in matches if row.get("verified") is True]
        elif correctness.value == "not verified":
            matches = [row for row in matches if row.get("verified") is False]
        return matches

    left_selected = selected(left_rows)
    right_selected = selected(right_rows)
    left_view = mo.json(left_selected[0]) if left_selected else mo.md("_No matching left row._")
    right_view = mo.json(right_selected[0]) if right_selected else mo.md("_No matching right row._")
    mo.vstack(
        [
            mo.md("## Same-example comparison"),
            mo.hstack([left_view, right_view], widths="equal"),
            mo.md("## Matching saved rows"),
            mo.ui.table(left_selected + right_selected, pagination=True),
        ]
    )
    return left_selected, right_selected


@app.cell
def _(left_selected, mo, right_selected):
    token_rows = []
    for row in left_selected + right_selected:
        tokens = row.get("tokens") or row.get("token_rows")
        if isinstance(tokens, list):
            token_rows.extend(token for token in tokens if isinstance(token, dict))
    mo.vstack(
        [
            mo.md("## Token-level audit data"),
            mo.ui.table(token_rows, pagination=True)
            if token_rows
            else mo.md("_No token audit rows in this selection._"),
        ]
    )


if __name__ == "__main__":
    app.run()
