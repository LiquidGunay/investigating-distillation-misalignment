"""Read-only marimo entry point; artifact views are added in Milestone 2."""

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md("# Distillation misalignment artifact inspector\n\nNo result artifacts have been generated yet.")
    return


if __name__ == "__main__":
    app.run()
