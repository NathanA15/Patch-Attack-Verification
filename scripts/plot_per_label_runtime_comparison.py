#!/usr/bin/env python3
"""
Create a runtime comparison bar plot from two fixed per-label CSV files.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CSV_PATH = (
    PROJECT_ROOT / "csv" / "recursive_timeout_refinement_20260403_082312_per_label.csv"
)
RESULT_CSV_PATH = (
    PROJECT_ROOT / "csv" / "recursive_timeout_refinement_20260326_175035_per_label_TODO.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "images"

SELECTED_ADV_LABELS = [2, 7]
SELECTED_UPPER_BOUNDS = [0.7, 0.75, 0.8]
EXCLUDED_STATUS_NAME = "TIME_LIMIT"

BENCHMARK_COLOR = "#4C78A8"
RESULT_COLOR = "#F58518"


def load_grouped_runtimes(csv_path: Path) -> dict[tuple[float, int], list[float]]:
    grouped_runtimes: dict[tuple[float, int], list[float]] = defaultdict(list)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status_name"] == EXCLUDED_STATUS_NAME:
                continue

            adv_label = int(row["adv_label"])
            upper_bound = float(row["upper_bound"])
            runtime_seconds = float(row["iteration_runtime_seconds"])

            if adv_label not in SELECTED_ADV_LABELS or upper_bound not in SELECTED_UPPER_BOUNDS:
                continue

            grouped_runtimes[(upper_bound, adv_label)].append(runtime_seconds)

    return dict(grouped_runtimes)


def build_plot_entries(
    benchmark_runtimes: list[float], result_runtimes: list[float]
) -> tuple[list[str], list[float], list[str]]:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    if benchmark_runtimes:
        if len(benchmark_runtimes) == 1:
            labels.append("Benchmark")
            values.append(benchmark_runtimes[0])
            colors.append(BENCHMARK_COLOR)
        else:
            for index, runtime in enumerate(benchmark_runtimes, start=1):
                labels.append(f"Benchmark {index}")
                values.append(runtime)
                colors.append(BENCHMARK_COLOR)

    for index, runtime in enumerate(result_runtimes, start=1):
        labels.append(f"Result {index}")
        values.append(runtime)
        colors.append(RESULT_COLOR)

    return labels, values, colors


def plot_subplot(
    axis: plt.Axes,
    *,
    upper_bound: float,
    adv_label: int,
    benchmark_runtimes: list[float],
    result_runtimes: list[float],
    legend_handles: list[Patch],
) -> None:
    x_labels, values, colors = build_plot_entries(benchmark_runtimes, result_runtimes)
    axis.set_title(f"upper_bound={upper_bound}, adv_label={adv_label}")
    axis.set_xlabel("Source / run")
    axis.set_ylabel("iteration_runtime_seconds")

    if not values:
        axis.text(
            0.5,
            0.5,
            "No matching data",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=11,
        )
        axis.set_xticks([])
        axis.legend(handles=legend_handles, loc="upper right")
        return

    x_positions = list(range(len(values)))
    axis.bar(x_positions, values, color=colors)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(x_labels, rotation=25, ha="right")
    axis.legend(handles=legend_handles, loc="upper right")
    axis.grid(axis="y", linestyle="--", alpha=0.35)


def create_runtime_comparison_plot() -> Path:
    benchmark_groups = load_grouped_runtimes(BENCHMARK_CSV_PATH)
    result_groups = load_grouped_runtimes(RESULT_CSV_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"runtime_comparison_benchmark_vs_result_{timestamp}.png"

    figure, axes = plt.subplots(
        nrows=len(SELECTED_UPPER_BOUNDS),
        ncols=len(SELECTED_ADV_LABELS),
        figsize=(16, 14),
        constrained_layout=False,
    )

    legend_handles = [
        Patch(facecolor=BENCHMARK_COLOR, label="Benchmark file"),
        Patch(facecolor=RESULT_COLOR, label="Result file"),
    ]

    for row_index, upper_bound in enumerate(SELECTED_UPPER_BOUNDS):
        for col_index, adv_label in enumerate(SELECTED_ADV_LABELS):
            axis = axes[row_index, col_index]
            plot_subplot(
                axis,
                upper_bound=upper_bound,
                adv_label=adv_label,
                benchmark_runtimes=benchmark_groups.get((upper_bound, adv_label), []),
                result_runtimes=result_groups.get((upper_bound, adv_label), []),
                legend_handles=legend_handles,
            )

    figure.suptitle(
        "Runtime comparison for benchmark vs result files\n"
        f"Generated at {timestamp} | excluding status_name={EXCLUDED_STATUS_NAME}",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    output_path = create_runtime_comparison_plot()
    print(f"Saved runtime comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
