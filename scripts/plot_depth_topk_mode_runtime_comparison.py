#!/usr/bin/env python3
"""
Create a combined runtime bar plot grouped by configured split depth.

Each subplot compares the no-split benchmark against recursive timeout
refinement runs for one (upper_bound, adv_label) target. Split bars are grouped
by parent_max_depth_reached, and each group contains one bar per
(split_selection_mode, top_k) combination.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [
    PROJECT_ROOT
    / "csv"
    / "recursive_timeout_refinement_20260515_122648.csv",
    PROJECT_ROOT
    / "csv"
    / "recursive_timeout_refinement_20260516_142920.csv",
    PROJECT_ROOT
    / "csv"
    / "recursive_timeout_refinement_20260516_213137.csv",
]
OUTPUT_DIR = PROJECT_ROOT / "images"

EXCLUDED_STATUS_NAME = "TIME_LIMIT"
BASELINE_COLOR = "#4C78A8"
MIN_RUNTIME_LINE_COLOR = "#2F2F2F"
MODE_ORDER = {"max": 0, "min": 1, "random": 2}
CATEGORY_COLORS = [
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]
RANDOM_COLOR_SEED = 42

CategoryKey = Tuple[str, int]


@dataclass(frozen=True)
class TargetKey:
    image_index: int
    patch_x: int
    patch_y: int
    patch_size: int
    upper_bound: float
    adv_label: int


SplitRuntimeKey = Tuple[TargetKey, int, str, int]


def normalize_csv_paths(csv_paths: list[Path]) -> list[Path]:
    if not csv_paths:
        raise ValueError("CSV_PATHS must contain at least one CSV path.")
    return [Path(csv_path) for csv_path in csv_paths]


def parse_runtime_seconds(row: dict[str, str], csv_path: Path) -> float:
    runtime_seconds = row["iteration_runtime_seconds"].strip()
    if not runtime_seconds:
        raise ValueError(
            "Found non-TIME_LIMIT row without iteration_runtime_seconds for "
            f"upper_bound={row['upper_bound']}, adv_label={row['adv_label']}, "
            f"parent_max_depth_reached={row['parent_max_depth_reached']} in {csv_path}"
        )
    return float(runtime_seconds)


def target_sort_key(target: TargetKey) -> tuple[int, int, int, int, int, float]:
    return (
        target.image_index,
        target.patch_x,
        target.patch_y,
        target.patch_size,
        target.adv_label,
        target.upper_bound,
    )


def category_sort_key(category: CategoryKey) -> tuple[int, int, str]:
    mode, top_k = category
    return MODE_ORDER.get(mode, len(MODE_ORDER)), top_k, mode


def format_number_for_filename(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def format_csv_source_label(csv_paths: list[Path]) -> str:
    if len(csv_paths) == 1:
        return csv_paths[0].name
    return f"{len(csv_paths)} CSV files"


def format_target_label(target: TargetKey) -> str:
    return (
        f"image_index={target.image_index} | "
        f"patch=({target.patch_x}, {target.patch_y}) | "
        f"patch_size={target.patch_size} | "
        f"adv_label={target.adv_label} | "
        f"upper_bound={target.upper_bound:g}"
    )


def format_bar_label(category: CategoryKey) -> str:
    mode, top_k = category
    mode_label = "rand" if mode == "random" else mode
    return f"{mode_label} {top_k}"


def load_runtimes(
    csv_paths: list[Path],
) -> tuple[
    dict[TargetKey, float],
    dict[TargetKey, dict[int, dict[CategoryKey, float]]],
]:
    baseline_runtimes: dict[TargetKey, float] = {}
    split_runtimes: dict[TargetKey, dict[int, dict[CategoryKey, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen_split_keys: set[SplitRuntimeKey] = set()

    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["status_name"].strip() == EXCLUDED_STATUS_NAME:
                    continue

                runtime_seconds = parse_runtime_seconds(row, csv_path)
                target = TargetKey(
                    image_index=int(row["image_index"]),
                    patch_x=int(row["patch_x"]),
                    patch_y=int(row["patch_y"]),
                    patch_size=int(row["patch_size"]),
                    upper_bound=float(row["upper_bound"]),
                    adv_label=int(row["adv_label"]),
                )
                attempt_count = int(row["parent_attempt_count"])

                if attempt_count == 1:
                    if target in baseline_runtimes:
                        raise ValueError(
                            "Found multiple non-TIME_LIMIT benchmark runtimes for "
                            f"{format_target_label(target)} "
                            f"across CSV_PATHS; duplicate found in {csv_path}"
                        )
                    baseline_runtimes[target] = runtime_seconds
                    continue

                max_depth = int(row["parent_max_depth_reached"])
                mode = row["split_selection_mode"].strip()
                top_k = int(row["top_k"])
                category = (mode, top_k)
                split_key = (target, max_depth, mode, top_k)

                if split_key in seen_split_keys:
                    raise ValueError(
                        "Found multiple non-TIME_LIMIT split runtimes for "
                        f"{format_target_label(target)}, "
                        f"parent_max_depth_reached={max_depth}, "
                        f"split_selection_mode={mode}, top_k={top_k} "
                        f"across CSV_PATHS; duplicate found in {csv_path}"
                    )

                seen_split_keys.add(split_key)
                split_runtimes[target][max_depth][category] = runtime_seconds

    return baseline_runtimes, {
        target: {depth: dict(values) for depth, values in by_depth.items()}
        for target, by_depth in split_runtimes.items()
    }


def validate_complete_split_grid(
    target: TargetKey,
    split_depth_runtimes: dict[int, dict[CategoryKey, float]],
) -> tuple[list[int], list[CategoryKey]]:
    split_depths = sorted(split_depth_runtimes)
    if not split_depths:
        return [], []

    categories = sorted(
        {category for depth_values in split_depth_runtimes.values() for category in depth_values},
        key=category_sort_key,
    )

    for depth in split_depths:
        actual_categories = set(split_depth_runtimes[depth])
        expected_categories = set(categories)
        missing = expected_categories - actual_categories
        extra = actual_categories - expected_categories
        if missing or extra:
            raise ValueError(
                "Incomplete split category grid for "
                f"{format_target_label(target)}, depth={depth}. "
                f"Missing={sorted(missing, key=category_sort_key)}, "
                f"extra={sorted(extra, key=category_sort_key)}"
            )

    return split_depths, categories


def build_category_colors(categories: list[CategoryKey]) -> dict[CategoryKey, str]:
    colors = list(CATEGORY_COLORS)
    if len(categories) > len(colors):
        random_source = random.Random(RANDOM_COLOR_SEED)
        while len(colors) < len(categories):
            colors.append(
                "#{:02X}{:02X}{:02X}".format(
                    random_source.randint(40, 215),
                    random_source.randint(40, 215),
                    random_source.randint(40, 215),
                )
            )

    return {
        category: colors[index]
        for index, category in enumerate(categories)
    }


def build_target_tag(target: TargetKey) -> str:
    return (
        f"img_{target.image_index}"
        f"_patch_{target.patch_x}_{target.patch_y}"
        f"_size_{target.patch_size}"
    )


def build_output_path(targets: list[TargetKey], timestamp: str, csv_paths: list[Path]) -> Path:
    csv_count_suffix = "" if len(csv_paths) == 1 else f"_csvs_{len(csv_paths)}"
    target_tags = {build_target_tag(target) for target in targets}
    target_tag = next(iter(target_tags)) if len(target_tags) == 1 else "mixed_targets"
    return (
        OUTPUT_DIR
        / (
            "runtime_by_depth_topk_mode_combined"
            f"_{target_tag}"
            f"{csv_count_suffix}"
            f"_{timestamp}.png"
        )
    )


def plot_target_runtime_comparison(
    *,
    axis: plt.Axes,
    target: TargetKey,
    baseline_runtime: float | None,
    split_depth_runtimes: dict[int, dict[CategoryKey, float]],
    category_colors: dict[CategoryKey, str],
) -> list[CategoryKey]:
    split_depths, categories = validate_complete_split_grid(target, split_depth_runtimes)

    x_ticks: list[float] = []
    x_tick_labels: list[str] = []

    bar_width = 0.82
    group_gap = 1.2
    current_x = 0.0

    if baseline_runtime is not None:
        axis.bar(
            current_x,
            baseline_runtime,
            width=bar_width,
            color=BASELINE_COLOR,
            label="Benchmark",
        )
        x_ticks.append(current_x)
        x_tick_labels.append("Benchmark")
        current_x += bar_width + group_gap

    for depth in split_depths:
        group_start = current_x
        for category_index, category in enumerate(categories):
            x_position = group_start + category_index * bar_width
            category_bar_width = bar_width * 0.98
            axis.bar(
                x_position,
                split_depth_runtimes[depth][category],
                width=category_bar_width,
                color=category_colors[category],
                align="edge",
            )
            axis.text(
                x_position + category_bar_width / 2,
                0.015,
                format_bar_label(category),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                clip_on=True,
            )

        group_width = len(categories) * bar_width
        min_runtime = min(split_depth_runtimes[depth][category] for category in categories)
        axis.hlines(
            min_runtime,
            group_start,
            group_start + group_width,
            colors=MIN_RUNTIME_LINE_COLOR,
            linestyles=":",
            linewidth=1.4,
            alpha=0.9,
            zorder=3,
        )
        x_ticks.append(group_start + group_width / 2)
        x_tick_labels.append(f"depth={depth}")
        current_x = group_start + group_width + group_gap

    axis.set_title(
        format_target_label(target),
        fontsize=12,
    )
    axis.set_ylabel("runtime (s)")
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labels)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    return categories


def create_runtime_comparison_plot() -> Path:
    csv_paths = normalize_csv_paths(CSV_PATHS)
    baseline_runtimes, split_runtimes = load_runtimes(csv_paths)
    targets = sorted(set(baseline_runtimes) | set(split_runtimes), key=target_sort_key)

    if not targets:
        raise ValueError(f"No benchmark or split runtimes found in {csv_paths}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = build_output_path(targets, timestamp, csv_paths)

    all_categories: set[CategoryKey] = set()
    max_depth_count = 0
    max_category_count = 0

    for target in targets:
        split_depths, categories = validate_complete_split_grid(
            target,
            split_runtimes.get(target, {}),
        )
        all_categories.update(categories)
        max_depth_count = max(max_depth_count, len(split_depths))
        max_category_count = max(max_category_count, len(categories))

    sorted_categories = sorted(all_categories, key=category_sort_key)
    category_colors = build_category_colors(sorted_categories)

    figure_width = max(10.5, 2.4 + 1.35 * max(1, max_category_count) * max(1, max_depth_count))
    figure_height = max(5.0, 4.5 * len(targets))
    figure, axes = plt.subplots(
        nrows=len(targets),
        ncols=1,
        figsize=(figure_width, figure_height),
        constrained_layout=False,
    )

    if len(targets) == 1:
        axes = [axes]

    for axis, target in zip(axes, targets):
        plot_target_runtime_comparison(
            axis=axis,
            target=target,
            baseline_runtime=baseline_runtimes.get(target),
            split_depth_runtimes=split_runtimes.get(target, {}),
            category_colors=category_colors,
        )

    legend_handles = [Patch(facecolor=BASELINE_COLOR, label="Benchmark (no splits)")]
    legend_handles.extend(
        Patch(facecolor=category_colors[(mode, top_k)], label=f"{mode}, top_k={top_k}")
        for mode, top_k in sorted_categories
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=MIN_RUNTIME_LINE_COLOR,
            linestyle=":",
            linewidth=1.4,
            label="Lowest runtime in depth",
        )
    )

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
    )
    figure.suptitle(
        f"Runtime by configured split depth, top_k, and split mode\n"
        f"{format_csv_source_label(csv_paths)}",
        fontsize=14,
    )
    figure.tight_layout(rect=[0, 0.08, 1, 0.93])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    output_path = create_runtime_comparison_plot()
    print(f"Saved runtime comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
