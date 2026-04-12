#!/usr/bin/env python3
"""
Create a runtime comparison bar plot from fixed benchmark and result CSV files.
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
    PROJECT_ROOT
    / "csv"
    / "recursive_timeout_refinement_20260403_082312_per_label_BENCHMARK.csv"
)
RESULT_CSV_PATH = PROJECT_ROOT / "csv" / "recursive_timeout_refinement_20260407_091932.csv"
OUTPUT_DIR = PROJECT_ROOT / "images"

EXCLUDED_STATUS_NAME = "TIME_LIMIT"
MIN_RESULT_DEPTH = 1
EXPECTED_UPPER_BOUNDS = [0.7, 0.75, 0.8]
EXPECTED_ADV_LABELS = [0, 2, 3, 4, 5, 6, 7, 8, 9]

BENCHMARK_COLOR = "#4C78A8"
RESULT_COLOR = "#F58518"


def load_shared_plot_metadata(csv_path: Path) -> dict[str, str]:
    metadata_keys = ("image_index", "patch_x", "patch_y", "patch_size")
    metadata_values = {key: set() for key in metadata_keys}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in metadata_keys:
                metadata_values[key].add(row[key].strip())

    resolved_metadata: dict[str, str] = {}
    for key, values in metadata_values.items():
        if len(values) != 1:
            raise ValueError(
                f"Expected exactly one {key} value in {csv_path}, but found {sorted(values)}"
            )
        resolved_metadata[key] = next(iter(values))

    return resolved_metadata


def load_and_validate_plot_metadata() -> dict[str, str]:
    benchmark_metadata = load_shared_plot_metadata(BENCHMARK_CSV_PATH)
    result_metadata = load_shared_plot_metadata(RESULT_CSV_PATH)

    if benchmark_metadata != result_metadata:
        raise ValueError(
            "Benchmark and result CSV metadata do not match: "
            f"{benchmark_metadata} != {result_metadata}"
        )

    return benchmark_metadata


def parse_runtime_seconds(row: dict[str, str], *, csv_path: Path) -> float:
    runtime_seconds = row["iteration_runtime_seconds"].strip()
    if not runtime_seconds:
        raise ValueError(
            "Missing iteration_runtime_seconds for "
            f"{csv_path} at upper_bound={row['upper_bound']}, "
            f"adv_label={row['adv_label']}, current_depth={row['current_depth']}, "
            f"status_name={row['status_name']}"
        )

    return float(runtime_seconds)


def discover_grid(csv_path: Path) -> tuple[list[float], list[int]]:
    upper_bounds: set[float] = set()
    adv_labels: set[int] = set()
    seen_pairs: set[tuple[float, int]] = set()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            upper_bound = float(row["upper_bound"])
            adv_label = int(row["adv_label"])
            upper_bounds.add(upper_bound)
            adv_labels.add(adv_label)
            seen_pairs.add((upper_bound, adv_label))

    sorted_upper_bounds = sorted(upper_bounds)
    sorted_adv_labels = sorted(adv_labels)

    if sorted_upper_bounds != EXPECTED_UPPER_BOUNDS:
        raise ValueError(
            f"Expected upper bounds {EXPECTED_UPPER_BOUNDS}, "
            f"but found {sorted_upper_bounds} in {csv_path}"
        )

    if sorted_adv_labels != EXPECTED_ADV_LABELS:
        raise ValueError(
            f"Expected adv labels {EXPECTED_ADV_LABELS}, "
            f"but found {sorted_adv_labels} in {csv_path}"
        )

    expected_pairs = {
        (upper_bound, adv_label)
        for upper_bound in sorted_upper_bounds
        for adv_label in sorted_adv_labels
    }
    if seen_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - seen_pairs)
        unexpected_pairs = sorted(seen_pairs - expected_pairs)
        raise ValueError(
            "Benchmark grid does not form the expected 3x9 layout. "
            f"Missing pairs: {missing_pairs} | Unexpected pairs: {unexpected_pairs}"
        )

    return sorted_upper_bounds, sorted_adv_labels


def load_benchmark_runtimes(
    csv_path: Path, valid_pairs: set[tuple[float, int]]
) -> dict[tuple[float, int], float]:
    benchmark_runtimes: dict[tuple[float, int], float] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status_name"] == EXCLUDED_STATUS_NAME:
                continue

            key = (float(row["upper_bound"]), int(row["adv_label"]))
            if key not in valid_pairs:
                continue

            runtime_seconds = parse_runtime_seconds(row, csv_path=csv_path)
            if key in benchmark_runtimes:
                raise ValueError(
                    "Found multiple non-TIME_LIMIT benchmark runtimes for "
                    f"upper_bound={key[0]}, adv_label={key[1]} in {csv_path}"
                )

            benchmark_runtimes[key] = runtime_seconds

    return benchmark_runtimes


def load_result_depth_runtimes(
    csv_path: Path, valid_pairs: set[tuple[float, int]]
) -> dict[tuple[float, int], dict[int, float]]:
    result_runtimes: dict[tuple[float, int], dict[int, float]] = defaultdict(dict)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status_name"] == EXCLUDED_STATUS_NAME:
                continue

            current_depth = int(row["current_depth"])
            if current_depth < MIN_RESULT_DEPTH:
                continue

            key = (float(row["upper_bound"]), int(row["adv_label"]))
            if key not in valid_pairs:
                continue

            runtime_seconds = parse_runtime_seconds(row, csv_path=csv_path)
            if current_depth in result_runtimes[key]:
                raise ValueError(
                    "Found multiple non-TIME_LIMIT result runtimes for "
                    f"upper_bound={key[0]}, adv_label={key[1]}, current_depth={current_depth} "
                    f"in {csv_path}"
                )

            result_runtimes[key][current_depth] = runtime_seconds

    return dict(result_runtimes)


def build_plot_entries(
    benchmark_runtime: float | None, depth_runtimes: dict[int, float]
) -> tuple[list[str], list[float], list[str]]:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    if benchmark_runtime is not None:
        labels.append("Benchmark")
        values.append(benchmark_runtime)
        colors.append(BENCHMARK_COLOR)

    for depth in sorted(depth_runtimes):
        labels.append(f"Depth {depth}")
        values.append(depth_runtimes[depth])
        colors.append(RESULT_COLOR)

    return labels, values, colors


def plot_subplot(
    axis: plt.Axes,
    *,
    upper_bound: float,
    adv_label: int,
    benchmark_runtime: float | None,
    depth_runtimes: dict[int, float],
) -> None:
    x_labels, values, colors = build_plot_entries(benchmark_runtime, depth_runtimes)
    axis.set_title(f"upper_bound={upper_bound}, adv_label={adv_label}", fontsize=11)
    axis.set_ylabel("iteration_runtime_seconds")

    if not values:
        axis.text(
            0.5,
            0.5,
            "No matching data",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=10,
        )
        axis.set_xticks([])
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        return

    x_positions = list(range(len(values)))
    axis.bar(x_positions, values, color=colors)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(x_labels, rotation=30, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.35)


def create_runtime_comparison_plot() -> Path:
    plot_metadata = load_and_validate_plot_metadata()
    upper_bounds, adv_labels = discover_grid(BENCHMARK_CSV_PATH)
    valid_pairs = {
        (upper_bound, adv_label) for upper_bound in upper_bounds for adv_label in adv_labels
    }
    benchmark_runtimes = load_benchmark_runtimes(BENCHMARK_CSV_PATH, valid_pairs)
    result_depth_runtimes = load_result_depth_runtimes(RESULT_CSV_PATH, valid_pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        OUTPUT_DIR
        / (
            "runtime_comparison_benchmark_vs_depth"
            f"_img_{plot_metadata['image_index']}"
            f"_patch_{plot_metadata['patch_x']}_{plot_metadata['patch_y']}"
            f"_size_{plot_metadata['patch_size']}"
            f"_{timestamp}.png"
        )
    )

    figure, axes = plt.subplots(
        nrows=len(upper_bounds),
        ncols=len(adv_labels),
        figsize=(34, 14),
        constrained_layout=False,
    )

    legend_handles = [
        Patch(facecolor=BENCHMARK_COLOR, label="Benchmark runtime"),
        Patch(facecolor=RESULT_COLOR, label="Result depth runtime"),
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)

    for row_index, upper_bound in enumerate(upper_bounds):
        for col_index, adv_label in enumerate(adv_labels):
            axis = axes[row_index, col_index]
            key = (upper_bound, adv_label)
            plot_subplot(
                axis,
                upper_bound=upper_bound,
                adv_label=adv_label,
                benchmark_runtime=benchmark_runtimes.get(key),
                depth_runtimes=result_depth_runtimes.get(key, {}),
            )

    figure.suptitle(
        "Runtime comparison for benchmark vs per-depth result runtimes\n"
        f"image_index={plot_metadata['image_index']} | "
        f"patch=({plot_metadata['patch_x']}, {plot_metadata['patch_y']}) | "
        f"patch_size={plot_metadata['patch_size']} | "
        f"Generated at {timestamp}",
        fontsize=18,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    output_path = create_runtime_comparison_plot()
    print(f"Saved runtime comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
