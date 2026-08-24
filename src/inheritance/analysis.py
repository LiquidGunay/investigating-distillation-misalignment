"""Artifact-only tables, figures, and verification packets for completed run groups."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inheritance.config import ConfigurationError, ensure_within_workspace, repository_root, write_json_atomic
from inheritance.reporting import read_jsonl, sha256_file


def _read_object(path: Path) -> dict[str, Any]:
    with ensure_within_workspace(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_text_atomic(path: Path, value: str) -> None:
    path = ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("report tables must contain at least one plotted row")
    preferred = (
        "run_id",
        "source_summary",
        "model_role",
        "condition",
        "intervention",
        "checkpoint",
        "optimizer_step",
        "dataset_split",
    )
    names = {str(name) for row in rows for name in row}
    fieldnames = [name for name in preferred if name in names]
    fieldnames.extend(sorted(names - set(fieldnames)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(path, buffer.getvalue())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _first_number(values: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        result = _number(values.get(name))
        if result is not None:
            return result
    return None


def _primary_alignment_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    alignment = summary.get("alignment")
    if not isinstance(alignment, Mapping):
        return []
    by_split = alignment.get("by_condition_split")
    if not isinstance(by_split, Mapping):
        return []
    lineage = alignment.get("primary_judge_lineage")
    if lineage is None:
        lineages = alignment.get("judge_lineages")
        lineage = lineages[0] if isinstance(lineages, list) and len(lineages) == 1 else None
    rows = []
    for value in by_split.values():
        if not isinstance(value, Mapping) or value.get("split") != "em_broad_eval_v1":
            continue
        rows.append(
            {
                "condition": str(value.get("condition")),
                "model_role": str(value.get("model_role")),
                "dataset_split": "em_broad_eval_v1",
                "alignment_score": _first_number(value, "alignment_score_mean", "alignment_mean"),
                "coherence_score": _first_number(value, "coherence_score_mean", "coherence_mean"),
                "coherence_guardrail_rate": _first_number(value, "coherence_guardrail_rate"),
                "em_rate": _first_number(value, "EM_present_rate"),
                "refusal_rate": _first_number(value, "refusal_rate"),
                "judge_lineage": lineage,
                "observations": value.get("observations"),
            }
        )
    return rows


def _math_rows(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    math_section = summary.get("math")
    if not isinstance(math_section, Mapping):
        return {}
    values = math_section.get("summaries", math_section)
    if not isinstance(values, Mapping):
        return {}
    return {
        str(condition): dict(metrics)
        for condition, metrics in values.items()
        if isinstance(metrics, Mapping) and _first_number(metrics, "exact_accuracy") is not None
    }


def _step_from_name(value: str) -> int | None:
    match = re.search(r"(?:step[:_-]?)(\d+)", value)
    return int(match.group(1)) if match else None


def _summary_condition_rows(path: Path, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    alignment = _primary_alignment_rows(summary)
    alignment_by_condition = {row["condition"]: row for row in alignment}
    run_id = str(summary.get("run_id", summary.get("training_run_id", path.parent.name)))
    rows = []
    for condition, metrics in _math_rows(summary).items():
        aligned = alignment_by_condition.get(condition)
        if aligned is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "source_summary": str(path.relative_to(repository_root())),
                "model_role": aligned["model_role"],
                "condition": condition,
                "checkpoint": condition,
                "optimizer_step": _step_from_name(condition),
                "dataset_split": aligned["dataset_split"],
                "math_accuracy": _first_number(metrics, "exact_accuracy"),
                "math_parse_rate": _first_number(metrics, "parse_rate"),
                "math_truncation_rate": _first_number(metrics, "truncation_rate"),
                "mean_completion_tokens": _first_number(metrics, "mean_completion_tokens"),
                **{key: value for key, value in aligned.items() if key not in {"condition", "model_role"}},
            }
        )
    return rows


def _checkpoint_rows(path: Path, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    math_by_checkpoint = summary.get("math_by_checkpoint")
    alignment_by_checkpoint = summary.get("alignment_by_checkpoint")
    if not isinstance(math_by_checkpoint, Mapping) or not isinstance(alignment_by_checkpoint, Mapping):
        return []
    rows = []
    for checkpoint, metrics in math_by_checkpoint.items():
        alignment_packet = alignment_by_checkpoint.get(checkpoint)
        if not isinstance(metrics, Mapping) or not isinstance(alignment_packet, Mapping):
            continue
        aligned_rows = _primary_alignment_rows({"alignment": alignment_packet})
        if len(aligned_rows) != 1:
            continue
        aligned = aligned_rows[0]
        rows.append(
            {
                "run_id": str(summary.get("run_id", path.parent.name)),
                "source_summary": str(path.relative_to(repository_root())),
                "model_role": aligned["model_role"],
                "condition": str(summary.get("training_condition", aligned["condition"])),
                "checkpoint": str(checkpoint),
                "optimizer_step": metrics.get("optimizer_step", alignment_packet.get("optimizer_step")),
                "dataset_split": aligned["dataset_split"],
                "math_accuracy": _first_number(metrics, "exact_accuracy"),
                "math_parse_rate": _first_number(metrics, "parse_rate"),
                "math_truncation_rate": _first_number(metrics, "truncation_rate"),
                "mean_completion_tokens": _first_number(metrics, "mean_completion_tokens"),
                **{key: value for key, value in aligned.items() if key not in {"condition", "model_role"}},
            }
        )
    return rows


def _intervention_for_summary(path: Path, summary: Mapping[str, Any]) -> str:
    direct = summary.get("intervention")
    if isinstance(direct, str):
        return direct
    for part in reversed(path.parts):
        match = re.search(
            r"intervention_(none|full|forward_only|backward_only|random_unit|random_energy_matched|wrong_layer)",
            part,
        )
        if match:
            return match.group(1)
    return "none"


def collect_capability_alignment_rows(summary_paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in summary_paths:
        summary = _read_object(path)
        extracted = [*_summary_condition_rows(path, summary), *_checkpoint_rows(path, summary)]
        intervention = _intervention_for_summary(path, summary)
        rows.extend({**row, "intervention": intervention} for row in extracted)
    identities = set()
    unique = []
    for row in rows:
        identity = (row["source_summary"], row["condition"], row["checkpoint"], row["dataset_split"])
        if identity in identities:
            raise ValueError(f"duplicate capability/alignment report row: {identity}")
        identities.add(identity)
        unique.append(row)
    return sorted(unique, key=lambda row: (str(row["run_id"]), int(row["optimizer_step"] or -1), str(row["condition"])))


def _latest_attempt_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identity = (
                str(path.relative_to(repository_root())),
                int(row["optimizer_step"]),
                str(row["phase"]),
                int(row["layer"]),
            )
            if identity not in latest or int(row.get("attempt", 0)) > int(latest[identity].get("attempt", 0)):
                latest[identity] = {"source_artifact": identity[0], **row}
    return [latest[key] for key in sorted(latest)]


def _audit_tables(roots: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intervention_paths = []
    for root in roots:
        intervention_paths.extend(root.rglob("intervention_metrics.jsonl"))
        for path in root.rglob("token_summaries.jsonl"):
            for row in read_jsonl(path):
                source = str(path.relative_to(repository_root()))
                tables["teacher_distribution"].append({"source_artifact": source, **row})
                bins = row.get("absolute_delta_probability_share_by_control_rank")
                if isinstance(bins, Mapping):
                    for rank_bin, share in bins.items():
                        tables["vocabulary_rank"].append(
                            {
                                "source_artifact": source,
                                "comparison": row.get("comparison"),
                                "position": row.get("position"),
                                "rank_bin": rank_bin,
                                "absolute_delta_probability_share": share,
                            }
                        )
        for path in root.rglob("audit_summary.json"):
            summary = _read_object(path)
            source = str(path.relative_to(repository_root()))
            for name in ("residual_gradient", "gradient_update", "source_fingerprint", "activation_drift"):
                values = summary.get(name)
                if isinstance(values, list):
                    tables[name].extend({"source_artifact": source, **row} for row in values)
    tables["removed_energy"] = _latest_attempt_rows(sorted(set(intervention_paths)))
    return tables


def _colors() -> tuple[str, ...]:
    return ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4b5563")


def _scatter_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_name: str,
    y_name: str,
    label_name: str,
    title: str,
) -> str:
    plotted = [(row, _number(row.get(x_name)), _number(row.get(y_name))) for row in rows]
    if any(x is None or y is None for _, x, y in plotted):
        raise ValueError(f"scatter rows require finite {x_name} and {y_name}")
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 55, 80
    xs = [float(x) for _, x, _ in plotted if x is not None]
    ys = [float(y) for _, _, y in plotted if y is not None]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, 0.01)
    y_pad = max((y_max - y_min) * 0.08, 0.5)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    series_names = sorted({str(row.get("run_id", row.get("condition", "series"))) for row, _, _ in plotted})
    color_by_series = {name: _colors()[index % len(_colors())] for index, name in enumerate(series_names)}
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>'
        ),
    ]
    for index in range(6):
        fraction = index / 5
        x = left + fraction * (width - left - right)
        y = top + fraction * (height - top - bottom)
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        pieces.extend(
            (
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#e5e7eb"/>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                (
                    f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12">{x_value:.3g}</text>'
                ),
                (
                    f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12">{y_value:.3g}</text>'
                ),
            )
        )
    grouped: dict[str, list[tuple[Mapping[str, Any], float, float]]] = defaultdict(list)
    for row, x, y in plotted:
        series = str(row.get("run_id", row.get("condition", "series")))
        grouped[series].append((row, float(x), float(y)))
    for series, values in grouped.items():
        color = color_by_series[series]
        ordered = sorted(values, key=lambda item: int(item[0].get("optimizer_step") or -1))
        if len(ordered) > 1:
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for _, x, y in ordered)
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row, x, y in ordered:
            label = html.escape(str(row.get(label_name, "")))
            pieces.append(
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="{color}"><title>{label}</title></circle>'
            )
    pieces.extend(
        (
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
            (
                f'<text x="{(left + width - right) / 2}" y="{height - 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="15">{html.escape(x_name)}</text>'
            ),
            (
                f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" '
                f'text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(y_name)}</text>'
            ),
            "</svg>",
        )
    )
    return "\n".join(pieces) + "\n"


def _bar_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    category_name: str,
    value_name: str,
    title: str,
) -> str:
    values = [(_number(row.get(value_name)), str(row.get(category_name))) for row in rows]
    if any(value is None for value, _ in values):
        raise ValueError(f"bar rows require finite {value_name}")
    width = 900
    height = max(360, 100 + 34 * len(values))
    left, right, top, bottom = 250, 40, 55, 45
    maximum = max(float(value) for value, _ in values) if values else 1.0
    minimum = min(0.0, min(float(value) for value, _ in values))
    span = max(maximum - minimum, 1e-12)
    plot_width = width - left - right
    zero = left + (0 - minimum) / span * plot_width
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>'
        ),
        f'<line x1="{zero:.2f}" y1="{top}" x2="{zero:.2f}" y2="{height - bottom}" stroke="#111827"/>',
    ]
    row_height = (height - top - bottom) / len(values)
    for index, (value, label) in enumerate(values):
        assert value is not None
        end = left + (float(value) - minimum) / span * plot_width
        x = min(zero, end)
        bar_width = abs(end - zero)
        y = top + index * row_height + row_height * 0.18
        anchor = "start" if value >= 0 else "end"
        pieces.extend(
            (
                (
                    f'<text x="{left - 10}" y="{y + row_height * 0.36:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
                ),
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{row_height * 0.64:.2f}" fill="{_colors()[index % len(_colors())]}"/>'
                ),
                (
                    f'<text x="{end + (6 if value >= 0 else -6):.2f}" '
                    f'y="{y + row_height * 0.36:.2f}" text-anchor="{anchor}" '
                    f'font-family="sans-serif" font-size="12">{value:.3g}</text>'
                ),
            )
        )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def _discover_roots(run_group: str, input_root: Path | None) -> list[Path]:
    root = repository_root()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_group) is None:
        raise ConfigurationError("report run group must be a simple filesystem-safe name")
    if input_root is not None:
        candidate = ensure_within_workspace(input_root)
        if not candidate.is_dir():
            raise ConfigurationError(f"report input root is not a directory: {candidate}")
        return [candidate]
    runs = ensure_within_workspace(root / "outputs" / "runs")
    direct = runs / run_group
    candidates = [direct] if direct.is_dir() else []
    candidates.extend(path for path in runs.rglob(run_group) if path.is_dir() and path.name == run_group)
    roots = sorted(set(ensure_within_workspace(path) for path in candidates))
    if not roots:
        raise ConfigurationError(f"no saved run group found for {run_group!r}")
    return roots


def _write_table_and_figure(
    output_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    figure: str,
) -> dict[str, Any]:
    csv_path = output_dir / f"{name}.csv"
    svg_path = output_dir / f"{name}.svg"
    _write_csv(csv_path, rows)
    _write_text_atomic(svg_path, figure)
    return {
        "rows": len(rows),
        "csv": {"path": csv_path.name, "sha256": sha256_file(csv_path)},
        "figure": {"path": svg_path.name, "sha256": sha256_file(svg_path)},
    }


def generate_report(
    *,
    run_group: str,
    output_dir: Path,
    input_root: Path | None = None,
) -> dict[str, Any]:
    """Regenerate all currently supported figures from saved JSON/JSONL only."""
    roots = _discover_roots(run_group, input_root)
    output_dir = ensure_within_workspace(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted({path for root in roots for path in root.rglob("summary.json")})
    capability = collect_capability_alignment_rows(summary_paths)
    audit = _audit_tables(roots)
    artifacts: dict[str, Any] = {}
    missing: list[str] = []

    teacher = [
        row
        for row in capability
        if row["model_role"] == "teacher"
        and _number(row["math_accuracy"]) is not None
        and _number(row["alignment_score"]) is not None
    ]
    if teacher:
        artifacts["teacher_calibration"] = _write_table_and_figure(
            output_dir,
            "teacher_calibration",
            teacher,
            _scatter_svg(
                teacher,
                x_name="math_accuracy",
                y_name="alignment_score",
                label_name="condition",
                title="Teacher capability and Broad-EM alignment",
            ),
        )
    else:
        missing.append("teacher_calibration")

    trajectory = [
        row
        for row in capability
        if row["model_role"] == "student"
        and _number(row["math_accuracy"]) is not None
        and _number(row["alignment_score"]) is not None
    ]
    if trajectory:
        artifacts["capability_misalignment_trajectory"] = _write_table_and_figure(
            output_dir,
            "capability_misalignment_trajectory",
            trajectory,
            _scatter_svg(
                trajectory,
                x_name="math_accuracy",
                y_name="alignment_score",
                label_name="checkpoint",
                title="Student capability–misalignment trajectories",
            ),
        )
    else:
        missing.append("capability_misalignment_trajectory")

    bar_specs = {
        "vocabulary_rank": ("rank_bin", "absolute_delta_probability_share", "Vocabulary-rank decomposition"),
        "gradient_update": ("comparison", "cosine", "Raw-gradient and AdamW-update alignment"),
        "source_fingerprint": ("comparison", "cosine", "Teacher-source fingerprint"),
        "activation_drift": ("checkpoint", "signed_projection", "Activation drift"),
    }
    for name, (category, value, title) in bar_specs.items():
        rows = [row for row in audit[name] if _number(row.get(value)) is not None]
        if rows:
            artifacts[name] = _write_table_and_figure(
                output_dir,
                name,
                rows,
                _bar_svg(rows, category_name=category, value_name=value, title=title),
            )
        else:
            missing.append(name)

    distribution = [row for row in audit["teacher_distribution"] if _number(row.get("total_variation")) is not None]
    if distribution:
        artifacts["teacher_distribution"] = _write_table_and_figure(
            output_dir,
            "teacher_distribution",
            distribution,
            _bar_svg(
                distribution,
                category_name="position",
                value_name="total_variation",
                title="Teacher distribution difference by token position",
            ),
        )
    else:
        missing.append("teacher_distribution")

    residual = [row for row in audit["residual_gradient"] if _number(row.get("signed_projection")) is not None]
    if residual:
        artifacts["residual_gradient"] = _write_table_and_figure(
            output_dir,
            "residual_gradient",
            residual,
            _bar_svg(
                residual,
                category_name="layer",
                value_name="signed_projection",
                title="Residual-gradient projection by layer",
            ),
        )
    else:
        missing.append("residual_gradient")

    intervention = [row for row in trajectory if row.get("intervention") != "none"]
    if intervention:
        artifacts["intervention_frontier"] = _write_table_and_figure(
            output_dir,
            "intervention_frontier",
            intervention,
            _scatter_svg(
                intervention,
                x_name="math_accuracy",
                y_name="alignment_score",
                label_name="intervention",
                title="Intervention capability–alignment frontier",
            ),
        )
    else:
        missing.append("intervention_frontier")

    removed = [
        row
        for row in audit["removed_energy"]
        if _number(row.get("aggregate_removed_energy_ratio")) is not None
    ]
    if removed:
        labeled = [
            {
                **row,
                "phase_layer": (
                    f"{row['phase']}:layer_{int(row['layer']):02d}:step_{int(row['optimizer_step']):04d}"
                ),
            }
            for row in removed
        ]
        artifacts["removed_energy"] = _write_table_and_figure(
            output_dir,
            "removed_energy",
            labeled,
            _bar_svg(
                labeled,
                category_name="phase_layer",
                value_name="aggregate_removed_energy_ratio",
                title="Removed activation and gradient energy",
            ),
        )
    else:
        missing.append("removed_energy")

    source_artifacts = [
        {"path": str(path.relative_to(repository_root())), "sha256": sha256_file(path)}
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"} and output_dir not in path.parents
    ]
    report = {
        "schema_version": 1,
        "run_group": run_group,
        "artifact_only": True,
        "model_loading": False,
        "input_roots": [str(root.relative_to(repository_root())) for root in roots],
        "source_artifacts": source_artifacts,
        "outputs": artifacts,
        "missing_outputs": sorted(missing),
        "status": "complete" if not missing else "partial_missing_saved_artifacts",
    }
    write_json_atomic(output_dir / "verification_packet.json", report)
    return report
