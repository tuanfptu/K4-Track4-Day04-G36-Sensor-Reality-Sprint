"""Validate and plot drift benchmarks without inventing missing measurements.

Matplotlib is imported only when plotting. Errors describe displacement from the
baseline projection for the same points visible in both views, not ground-truth
calibration error. Association is retention in fixed baseline 2D boxes.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_FIELDS = (
    "condition", "drift_value", "drift_unit", "median_error_px",
    "p90_error_px", "association_retention",
)


def _number(value: Any, field: str, row_number: int, *, optional: bool = False) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise ValueError(f"Row {row_number}: {field} must not be blank")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Row {row_number}: {field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"Row {row_number}: {field} must be finite")
    return result


def _validate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    keys = set()
    for row_number, source in enumerate(rows, 2):
        missing = set(REQUIRED_FIELDS) - source.keys()
        if missing:
            raise ValueError(f"Row {row_number}: missing required fields: {', '.join(sorted(missing))}")
        row = dict(source)
        for field in ("condition", "drift_unit"):
            value = row[field]
            if value is None or not str(value).strip():
                raise ValueError(f"Row {row_number}: {field} must not be blank")
            row[field] = str(value).strip()
        row["drift_value"] = _number(row["drift_value"], "drift_value", row_number)
        key = (row["condition"], row["drift_unit"], row["drift_value"])
        if key in keys:
            raise ValueError(f"Row {row_number}: duplicate condition/unit/drift: {key}")
        keys.add(key)
        for field in ("median_error_px", "p90_error_px", "association_retention", "association_drop_pp", "map"):
            row[field] = _number(row.get(field), field, row_number, optional=True)
        for field in ("median_error_px", "p90_error_px"):
            if row[field] is not None and row[field] < 0:
                raise ValueError(f"Row {row_number}: {field} must be nonnegative")
        for field in ("association_retention", "map"):
            if row[field] is not None and not 0 <= row[field] <= 1:
                raise ValueError(f"Row {row_number}: {field} must be in [0, 1]")
        retention, drop = row["association_retention"], row["association_drop_pp"]
        if drop is not None and not 0 <= drop <= 100:
            raise ValueError(f"Row {row_number}: association_drop_pp must be in [0, 100]")
        if retention is not None:
            expected_drop = 100 * (1 - retention)
            if drop is not None and not math.isclose(drop, expected_drop, rel_tol=0, abs_tol=1e-3):
                raise ValueError(f"Row {row_number}: association_drop_pp is inconsistent with retention")
            if drop is None:
                row["association_drop_pp"] = expected_drop
        row["map_protocol"] = str(row.get("map_protocol") or "").strip()
        if row["map"] is not None and not row["map_protocol"]:
            raise ValueError(f"Row {row_number}: a measured map requires map_protocol")
        parsed.append(row)
    if not parsed:
        raise ValueError("Benchmark contains no data rows")
    series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        series[(row["condition"], row["drift_unit"])].append(row)
    for key, group in series.items():
        measured = [row for row in group if row["map"] is not None]
        if not measured:
            continue
        if len({row["map_protocol"] for row in measured}) != 1:
            raise ValueError(f"Series {key}: measured map values must use one map_protocol")
        if not any(row["drift_value"] == 0 and row["map"] is not None for row in group):
            raise ValueError(f"Series {key}: measured map requires a measured zero-drift baseline")
    return parsed


def read_benchmark_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical benchmark CSV; blank measurements remain ``None``.

    Required columns are :data:`REQUIRED_FIELDS`. ``association_drop_pp`` may
    be omitted and is derived when retention is defined. Optional ``map`` is a
    fraction in [0, 1] and requires ``map_protocol`` and a zero-drift baseline
    within each condition/unit series. Extra columns are preserved as strings.
    Duplicate sweep samples, nonfinite values and inconsistent rates are errors.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Benchmark CSV is empty")
        headers = [field.strip() for field in reader.fieldnames]
        if len(set(headers)) != len(headers) or any(not field for field in headers):
            raise ValueError("Benchmark CSV has duplicate or blank column names")
        missing = set(REQUIRED_FIELDS) - set(headers)
        if missing:
            raise ValueError(f"Benchmark CSV missing required fields: {', '.join(sorted(missing))}")
        reader.fieldnames = headers
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"CSV line {reader.line_num}: more values than columns")
            if not any(value is not None and value.strip() for value in row.values()):
                continue
            rows.append(row)
    return _validate_rows(rows)


def _unit_suffixes(units: list[str]) -> dict[str, str]:
    suffixes = {}
    used = set()
    for unit in units:
        base = re.sub(r"[^a-z0-9]+", "_", unit.lower()).strip("_") or "unit"
        suffix = base
        index = 2
        while suffix in used:
            suffix = f"{base}_{index}"
            index += 1
        used.add(suffix)
        suffixes[unit] = suffix
    return suffixes


def plot_benchmark(rows: Iterable[Mapping[str, Any]], output_dir: str | Path) -> list[Path]:
    """Save scientific metric figures, splitting units and keeping series separate.

    Returns metrics figure paths first, then any mAP figures. Missing observations
    are gaps. An mAP chart is produced only for explicit measurements with a
    declared evaluation protocol; proxy association is never treated as mAP.
    """
    validated = _validate_rows(rows)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("Benchmark plotting requires matplotlib; install requirements.txt") from error

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in validated:
        groups[row["drift_unit"]][row["condition"]].append(row)
    units = sorted(groups)
    suffixes = _unit_suffixes(units)
    metrics_paths, map_paths = [], []
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#444444")

    def prepare_axes(axes, unit):
        for axis in axes:
            axis.set_xlabel(f"Signed calibration drift ({unit})")
            axis.grid(True, alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)

    def values(group, field, scale=1):
        return [float("nan") if row[field] is None else scale * row[field] for row in group]

    def curve(axis, x, y, label, color, style="-", marker="o"):
        if any(math.isfinite(value) for value in y):
            axis.plot(x, y, label=label, color=color, linestyle=style, marker=marker,
                      markersize=4, linewidth=1.7)

    def finish_axes(axes, missing_messages):
        for axis, message in zip(axes, missing_messages):
            if axis.lines:
                axis.legend(fontsize=8, loc="best", frameon=False)
            else:
                axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=9)

    for unit in units:
        series = groups[unit]
        figure = Figure(figsize=(15, 5.4), dpi=160)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        prepare_axes(axes, unit)
        axes[0].set_title("Projection displacement vs baseline")
        axes[0].set_ylabel("Displacement (pixels)")
        axes[1].set_title("Fixed-box association retention")
        axes[1].set_ylabel("Baseline associations retained (%)")
        axes[1].set_ylim(0, 105)
        axes[2].set_title("Fixed-box association drop")
        axes[2].set_ylabel("Association drop (percentage points)")
        axes[2].set_ylim(0, 105)
        for index, condition in enumerate(sorted(series)):
            group = sorted(series[condition], key=lambda row: row["drift_value"])
            color = colors[index % len(colors)]
            x = [row["drift_value"] for row in group]
            curve(axes[0], x, values(group, "median_error_px"), f"{condition}: median", color)
            curve(axes[0], x, values(group, "p90_error_px"), f"{condition}: P90", color, "--", "^")
            curve(axes[1], x, values(group, "association_retention", 100), condition, color)
            curve(axes[2], x, values(group, "association_drop_pp"), condition, color)
        # Set the lower bound after plotting, preserving autoscaled upper limits.
        axes[0].set_ylim(bottom=0)
        finish_axes(axes, ["Undefined: no points visible\nin both views", "Undefined: no baseline\nbox associations", "Undefined: no baseline\nbox associations"])
        figure.suptitle("Calibration drift impact", fontsize=15, y=0.98)
        figure.text(0.5, 0.035,
                    "Displacement uses matched points visible in both views. Association is a fixed-box retention proxy, not detection mAP.\n"
                    "Blank observations are gaps; each line is one corruption condition. Units are plotted separately.",
                    ha="center", va="bottom", fontsize=9)
        figure.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.25, wspace=0.32)
        suffix = f"_{suffixes[unit]}" if len(units) > 1 else ""
        metric_path = destination / f"metrics{suffix}.png"
        figure.savefig(metric_path, dpi=160)
        figure.clear()
        metrics_paths.append(metric_path)

        measured_series = [(condition, sorted(group, key=lambda row: row["drift_value"]))
                           for condition, group in sorted(series.items()) if any(row["map"] is not None for row in group)]
        if not measured_series:
            continue
        figure = Figure(figsize=(11, 5.4), dpi=160)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        prepare_axes(axes, unit)
        axes[0].set_title("Measured detection mAP")
        axes[0].set_ylabel("mAP (%)")
        axes[0].set_ylim(0, 105)
        axes[1].set_title("mAP drop from zero-drift baseline")
        axes[1].set_ylabel("mAP drop (percentage points)")
        axes[1].axhline(0, color="#777777", linewidth=0.8)
        for index, (condition, group) in enumerate(measured_series):
            color = colors[index % len(colors)]
            baseline = next(row["map"] for row in group if row["drift_value"] == 0 and row["map"] is not None)
            protocol = next(row["map_protocol"] for row in group if row["map"] is not None)
            x = [row["drift_value"] for row in group]
            measured = values(group, "map", 100)
            drop = [100 * baseline - value for value in measured]
            label = f"{condition} [{protocol}]"
            curve(axes[0], x, measured, label, color)
            curve(axes[1], x, drop, label, color)
        finish_axes(axes, ["No measured mAP", "No measured mAP"])
        figure.suptitle("Calibration drift: supplied detection evaluation", fontsize=14, y=0.98)
        figure.text(0.5, 0.04, "Only supplied mAP measurements are plotted. Blank observations are gaps.\n"
                    "Drop is relative to each condition's measured zero-drift baseline; a negative drop means improvement.",
                    ha="center", va="bottom", fontsize=9)
        figure.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.24, wspace=0.3)
        map_path = destination / f"map{suffix}.png"
        figure.savefig(map_path, dpi=160)
        figure.clear()
        map_paths.append(map_path)
    return metrics_paths + map_paths
