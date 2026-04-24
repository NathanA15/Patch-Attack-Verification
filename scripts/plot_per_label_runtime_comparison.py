#!/usr/bin/env python3
"""
Create a runtime comparison bar plot from a single CSV file containing all runs,
including the baseline (no-split) run and recursive timeout refinement runs.

The baseline run is identified by parent_attempt_count == 1 (no splits).
Split runs have parent_attempt_count > 1 and are distinguished by their
parent_max_depth_reached value.
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
CSV_PATH = (
    PROJECT_ROOT
    / "csv"
    / "recursive_timeout_refinement_20260413_194652.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "images"

EXCLUDED_STATUS_NAME = "TIME_LIMIT"
BASELINE_COLOR = "#4C78A8"
SPLIT_COLORS = ["#F58518", "#F58518", "#F58518"]


def load_shared_plot_metadata(csv_path: Path) -> dict[str, str]:
    """Extract shared metadata (image_index, patch_x/y, patch_size) from CSV."""
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


def parse_runtime_seconds(row: dict[str, str]) -> float | None:
    runtime_seconds = row["iteration_runtime_seconds"].strip()
    if not runtime_seconds:
        return None
    return float(runtime_seconds)


def identify_runs(csv_path: Path) -> dict[str, dict]:
    """
    Identify distinct runs from the CSV by extracting the run identifier
    from the log_path column.

    Returns a dict mapping run_id -> {upper_bound, parent_attempt_count,
    parent_max_depth_reached, timeout_chain}.
    """
    runs: dict[str, dict] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            log_path = row["log_path"].strip()
            # Extract run directory name, e.g.
            # "20260411_152600_run_01_ub_0p8_timeouts_720000"
            parts = Path(log_path).parent.name
            if parts in runs:
                continue

            runs[parts] = {
                "upper_bound": float(row["upper_bound"]),
                "parent_attempt_count": int(row["parent_attempt_count"]),
                "parent_max_depth_reached": int(row["parent_max_depth_reached"]),
                "final_outcome": row["final_outcome"].strip(),
            }

    return runs


def discover_grid(csv_path: Path) -> tuple[list[float], list[int]]:
    """Discover the set of upper_bounds and adv_labels present in the CSV."""
    upper_bounds: set[float] = set()
    adv_labels: set[int] = set()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            upper_bounds.add(float(row["upper_bound"]))
            adv_labels.add(int(row["adv_label"]))

    return sorted(upper_bounds), sorted(adv_labels)


def load_baseline_runtimes(
    csv_path: Path,
) -> dict[tuple[float, int], float]:
    """
    Load runtimes from the baseline (no-split) runs.

    Baseline runs are identified by parent_attempt_count == 1.
    Only rows from initial_all_labels scope are used.
    For each (upper_bound, adv_label), we take the per-label runtime.
    """
    baseline_runtimes: dict[tuple[float, int], float] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["parent_attempt_count"]) != 1:
                continue

            if row["status_name"].strip() == EXCLUDED_STATUS_NAME:
                continue

            key = (float(row["upper_bound"]), int(row["adv_label"]))
            runtime_seconds = parse_runtime_seconds(row)
            if runtime_seconds is None:
                continue

            if key in baseline_runtimes:
                raise ValueError(
                    "Found multiple non-TIME_LIMIT baseline runtimes for "
                    f"upper_bound={key[0]}, adv_label={key[1]} in {csv_path}"
                )

            baseline_runtimes[key] = runtime_seconds

    return baseline_runtimes


def load_split_runtimes(
    csv_path: Path,
) -> dict[int, dict[tuple[float, int], dict[int, float]]]:
    """
    Load runtimes from the split (recursive timeout) runs, grouped by
    parent_max_depth_reached.

    Returns: {max_depth: {(upper_bound, adv_label): {current_depth: runtime}}}

    For each split run, we collect non-TIME_LIMIT rows across all depths.
    The initial_all_labels rows at depth 0 are the "depth 0" entries,
    while single_label_rerun rows provide deeper depth entries.
    """
    # Group by (parent_attempt_count, parent_max_depth_reached, upper_bound)
    # to identify distinct split configurations[]
    split_runtimes: dict[int, dict[tuple[float, int], dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            attempt_count = int(row["parent_attempt_count"])
            if attempt_count == 1:
                continue  # Skip baseline

            if row["status_name"].strip() == EXCLUDED_STATUS_NAME:
                continue

            max_depth = int(row["parent_max_depth_reached"])
            upper_bound = float(row["upper_bound"])
            adv_label = int(row["adv_label"])
            current_depth = int(row["current_depth"])
            key = (upper_bound, adv_label)

            runtime_seconds = parse_runtime_seconds(row)
            if runtime_seconds is None:
                continue

            if current_depth in split_runtimes[max_depth][key]:
                raise ValueError(
                    "Found multiple non-TIME_LIMIT runtimes for "
                    f"max_depth={max_depth}, upper_bound={upper_bound}, "
                    f"adv_label={adv_label}, current_depth={current_depth} "
                    f"in {csv_path}"
                )

            split_runtimes[max_depth][key][current_depth] = runtime_seconds

    return {k: dict(v) for k, v in split_runtimes.items()}


def build_plot_entries(
    baseline_runtime: float | None,
    split_depth_runtimes: dict[int, dict[int, float]],
    sorted_max_depths: list[int],
) -> tuple[list[str], list[float], list[str]]:
    """
    Build bar labels, values, and colors for one subplot cell.

    For each (upper_bound, adv_label), we show:
    - One bar for the baseline runtime
    - For each max_depth split run, one bar per depth level within that run
    """
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    if baseline_runtime is not None:
        labels.append("Benchmark")
        values.append(baseline_runtime)
        colors.append(BASELINE_COLOR)

    for i, max_depth in enumerate(sorted_max_depths):
        depth_runtimes = split_depth_runtimes.get(max_depth, {})
        color = SPLIT_COLORS[i % len(SPLIT_COLORS)]
        for depth in sorted(depth_runtimes):
            labels.append(f"Split {i + 1}")
            values.append(depth_runtimes[depth])
            colors.append(color)

    return labels, values, colors


def plot_subplot(
    axis: plt.Axes,
    *,
    upper_bound: float,
    adv_label: int,
    baseline_runtime: float | None,
    split_depth_runtimes: dict[int, dict[int, float]],
    sorted_max_depths: list[int],
) -> None:
    x_labels, values, colors = build_plot_entries(
        baseline_runtime, split_depth_runtimes, sorted_max_depths
    )
    axis.set_title(f"ub={upper_bound}, adv={adv_label}", fontsize=9)
    axis.set_ylabel("runtime (s)", fontsize=7)

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
    axis.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
    axis.grid(axis="y", linestyle="--", alpha=0.35)


def create_runtime_comparison_plot() -> Path:
    plot_metadata = load_shared_plot_metadata(CSV_PATH)
    upper_bounds, adv_labels = discover_grid(CSV_PATH)

    baseline_runtimes = load_baseline_runtimes(CSV_PATH)
    split_runtimes = load_split_runtimes(CSV_PATH)
    sorted_max_depths = sorted(split_runtimes.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        OUTPUT_DIR
        / (
            "runtime_comparison_baseline_vs_splits"
            f"_img_{plot_metadata['image_index']}"
            f"_patch_{plot_metadata['patch_x']}_{plot_metadata['patch_y']}"
            f"_size_{plot_metadata['patch_size']}"
            f"_{timestamp}.png"
        )
    )

    figure, axes = plt.subplots(
        nrows=len(upper_bounds),
        ncols=len(adv_labels),
        figsize=(34, 4 * len(upper_bounds)),
        constrained_layout=False,
    )

    # Handle single row/col edge cases
    if len(upper_bounds) == 1 and len(adv_labels) == 1:
        axes = [[axes]]
    elif len(upper_bounds) == 1:
        axes = [axes]
    elif len(adv_labels) == 1:
        axes = [[ax] for ax in axes]

    legend_handles = [
        Patch(facecolor=BASELINE_COLOR, label="Benchmark (no splits)"),
        Patch(facecolor=SPLIT_COLORS[0], label="Split runs"),
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=False)

    for row_index, upper_bound in enumerate(upper_bounds):
        for col_index, adv_label in enumerate(adv_labels):
            axis = axes[row_index][col_index]
            key = (upper_bound, adv_label)

            # Gather per-max_depth runtimes for this cell
            cell_split_runtimes: dict[int, dict[int, float]] = {}
            for max_depth in sorted_max_depths:
                if key in split_runtimes.get(max_depth, {}):
                    cell_split_runtimes[max_depth] = split_runtimes[max_depth][key]

            plot_subplot(
                axis,
                upper_bound=upper_bound,
                adv_label=adv_label,
                baseline_runtime=baseline_runtimes.get(key),
                split_depth_runtimes=cell_split_runtimes,
                sorted_max_depths=sorted_max_depths,
            )

    figure.suptitle(
        "Runtime comparison: baseline (no splits) vs recursive timeout splits\n"
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
