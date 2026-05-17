#!/usr/bin/env python3
"""
Rank experiment parameters by association with runtime_to_benchmark_ratio.

The analysis uses the same CSV set as plot_depth_topk_mode_runtime_comparison.py.
Rows with blank iteration_runtime_seconds are always skipped. By default,
TIME_LIMIT rows are also skipped so the report focuses on completed iterations;
pass --include-time-limits to include numeric timeout rows.

The correlation target is:

    runtime_to_benchmark_ratio =
        iteration_runtime_seconds / matching benchmark iteration_runtime_seconds

The matching benchmark has the same image/label/patch/upper-bound target,
depth 0, and parent_attempt_count 1. By default the benchmark timeout is not
filtered, because the current three benchmark families use both 720000 and
72000. Pass --benchmark-timeout-milp to restrict benchmark rows to one timeout.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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
OUTPUT_DIR = PROJECT_ROOT / "csv"

RAW_RUNTIME_COLUMN = "iteration_runtime_seconds"
TARGET_COLUMN = "runtime_to_benchmark_ratio"
DEFAULT_EXCLUDED_STATUSES = {"TIME_LIMIT"}
FEATURE_START_COLUMN = "top_k"
BENCHMARK_MATCH_COLUMNS = [
    "image_index",
    "true_label",
    "adv_label",
    "patch_x",
    "patch_y",
    "patch_size",
    "upper_bound",
]
BENCHMARK_DEPTH_COLUMN_FILTERS = {
    "current_depth": 0,
    "parent_attempt_count": 1,
    "parent_max_depth_reached": 0,
}
NUMERIC_TOKEN_PATTERN = re.compile(
    r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?"
)
COMBINATION_COLUMNS = [
    "top_k",
    "split_selection_mode",
    "split_random_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank CSV parameters by correlation/association with "
            f"{TARGET_COLUMN}."
        )
    )
    parser.add_argument(
        "--include-time-limits",
        action="store_true",
        help="Include TIME_LIMIT rows that have a numeric iteration runtime.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of rows to print from each leaderboard.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without writing report CSV files.",
    )
    parser.add_argument(
        "--start-column",
        default=FEATURE_START_COLUMN,
        help="Analyze every CSV column starting at this column name.",
    )
    parser.add_argument(
        "--benchmark-timeout-milp",
        type=float,
        default=None,
        help=(
            "Optional timeout_milp_s value used to filter depth-0 benchmark rows. "
            "By default, all depth-0 benchmarks are eligible."
        ),
    )
    parser.add_argument(
        "--max-categorical-levels",
        type=int,
        default=50,
        help=(
            "Maximum distinct values for categorical/one-hot/fastest-level "
            "summaries. Numeric extraction still runs for higher-cardinality text."
        ),
    )
    return parser.parse_args()


def load_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        frames.append(pd.read_csv(csv_path).assign(csv_source=csv_path.name))
    return pd.concat(frames, ignore_index=True)


def prepare_rows(
    df: pd.DataFrame,
    include_time_limits: bool,
    benchmark_timeout_milp: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    prepared = df.copy()
    prepared[RAW_RUNTIME_COLUMN] = pd.to_numeric(prepared[RAW_RUNTIME_COLUMN], errors="coerce")

    counts = {
        "loaded_rows": len(prepared),
        "blank_runtime_rows": int(prepared[RAW_RUNTIME_COLUMN].isna().sum()),
    }

    prepared = prepared[prepared[RAW_RUNTIME_COLUMN].notna()].copy()

    if not include_time_limits:
        excluded_mask = prepared["status_name"].isin(DEFAULT_EXCLUDED_STATUSES)
        counts["excluded_time_limit_rows"] = int(excluded_mask.sum())
        prepared = prepared[~excluded_mask].copy()
    else:
        counts["excluded_time_limit_rows"] = 0

    prepared, ratio_counts = add_runtime_to_benchmark_ratio(
        prepared,
        benchmark_timeout_milp=benchmark_timeout_milp,
    )
    counts.update(ratio_counts)
    counts["analyzed_rows"] = len(prepared)
    return prepared, counts


def normalize_key_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def build_target_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(normalize_key_value(row[column]) for column in BENCHMARK_MATCH_COLUMNS)


def add_runtime_to_benchmark_ratio(
    df: pd.DataFrame,
    benchmark_timeout_milp: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_columns = [
        column
        for column in BENCHMARK_MATCH_COLUMNS
        + list(BENCHMARK_DEPTH_COLUMN_FILTERS)
        + ["timeout_milp_s", RAW_RUNTIME_COLUMN]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns needed for benchmark ratio: {missing_columns}")

    benchmark_mask = pd.Series(True, index=df.index)
    for column, expected_value in BENCHMARK_DEPTH_COLUMN_FILTERS.items():
        benchmark_mask &= pd.to_numeric(df[column], errors="coerce") == expected_value
    if benchmark_timeout_milp is not None:
        benchmark_mask &= (
            pd.to_numeric(df["timeout_milp_s"], errors="coerce") == benchmark_timeout_milp
        )

    benchmark_rows = df[benchmark_mask].copy()
    if benchmark_rows.empty:
        timeout_clause = (
            f"timeout_milp_s={benchmark_timeout_milp:g}, "
            if benchmark_timeout_milp is not None
            else ""
        )
        raise ValueError(
            "No benchmark rows found with "
            f"{timeout_clause}"
            "current_depth=0, parent_attempt_count=1, "
            "and parent_max_depth_reached=0."
        )

    benchmark_rows["_benchmark_key"] = benchmark_rows.apply(build_target_key, axis=1)
    benchmark_runtimes = benchmark_rows.groupby("_benchmark_key")[RAW_RUNTIME_COLUMN].median()

    with_ratios = df.copy()
    with_ratios["_benchmark_key"] = with_ratios.apply(build_target_key, axis=1)
    with_ratios["benchmark_iteration_runtime_seconds"] = with_ratios["_benchmark_key"].map(
        benchmark_runtimes
    )
    with_ratios[TARGET_COLUMN] = (
        with_ratios[RAW_RUNTIME_COLUMN]
        / with_ratios["benchmark_iteration_runtime_seconds"]
    )

    missing_benchmark_mask = with_ratios["benchmark_iteration_runtime_seconds"].isna()
    zero_benchmark_mask = with_ratios["benchmark_iteration_runtime_seconds"] == 0
    invalid_ratio_mask = (
        missing_benchmark_mask
        | zero_benchmark_mask
        | with_ratios[TARGET_COLUMN].isna()
        | ~np.isfinite(with_ratios[TARGET_COLUMN])
    )

    counts = {
        "benchmark_timeout_milp": benchmark_timeout_milp,
        "benchmark_rows": len(benchmark_rows),
        "benchmark_targets": int(benchmark_rows["_benchmark_key"].nunique()),
        "rows_without_matching_benchmark": int(missing_benchmark_mask.sum()),
        "rows_with_zero_benchmark_runtime": int(zero_benchmark_mask.sum()),
    }

    with_ratios = with_ratios[~invalid_ratio_mask].copy()
    with_ratios = with_ratios.drop(columns=["_benchmark_key"])
    return with_ratios, counts


def as_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def get_feature_columns(df: pd.DataFrame, start_column: str) -> list[str]:
    if start_column not in df.columns:
        raise ValueError(f"Start column {start_column!r} was not found in the CSV columns.")

    start_index = df.columns.get_loc(start_column)
    return [
        column
        for column in df.columns[start_index:]
        if column
        not in {
            RAW_RUNTIME_COLUMN,
            TARGET_COLUMN,
            "benchmark_iteration_runtime_seconds",
            "csv_source",
        }
    ]


def extract_numbers(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [float(match.group(0)) for match in NUMERIC_TOKEN_PATTERN.finditer(str(value))]


def build_embedded_numeric_features(series: pd.Series) -> dict[str, pd.Series]:
    numbers_by_row = series.map(extract_numbers)

    def summarize(numbers: list[float], summary_name: str) -> float:
        if not numbers:
            return np.nan
        if summary_name == "number_count":
            return float(len(numbers))
        if summary_name == "number_first":
            return numbers[0]
        if summary_name == "number_last":
            return numbers[-1]
        if summary_name == "number_min":
            return min(numbers)
        if summary_name == "number_max":
            return max(numbers)
        if summary_name == "number_mean":
            return float(np.mean(numbers))
        if summary_name == "number_span":
            return max(numbers) - min(numbers)
        raise ValueError(f"Unknown numeric summary: {summary_name}")

    features = {}
    for summary_name in [
        "number_count",
        "number_first",
        "number_last",
        "number_min",
        "number_max",
        "number_mean",
        "number_span",
    ]:
        features[summary_name] = numbers_by_row.map(
            lambda numbers, name=summary_name: summarize(numbers, name)
        )
    return features


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return np.nan
    x_values = frame.iloc[:, 0].to_numpy(dtype=float)
    y_values = frame.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return np.nan
    ranked_x = frame.iloc[:, 0].rank(method="average")
    ranked_y = frame.iloc[:, 1].rank(method="average")
    return pearson_corr(ranked_x, ranked_y)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame({"category": categories.astype("string"), "value": values})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["category"].nunique() < 2:
        return np.nan

    overall_mean = frame["value"].mean()
    total_ss = float(((frame["value"] - overall_mean) ** 2).sum())
    if total_ss == 0:
        return np.nan

    between_ss = 0.0
    for _, group in frame.groupby("category", dropna=True):
        between_ss += len(group) * float((group["value"].mean() - overall_mean) ** 2)
    eta_squared = between_ss / total_ss
    return float(np.sqrt(max(0.0, eta_squared)))


def summarize_numeric_feature(
    column: str,
    feature_name: str,
    numeric: pd.Series,
    target: pd.Series,
    source_kind: str,
) -> dict[str, object]:
    pearson = pearson_corr(numeric, target)
    spearman = spearman_corr(numeric, target)
    unique_values = int(numeric.dropna().nunique())
    strength_values = [value for value in [abs(pearson), abs(spearman)] if not np.isnan(value)]

    direction = ""
    if not np.isnan(spearman):
        direction = "higher runtime with larger values" if spearman > 0 else "lower runtime with larger values"

    return {
        "parameter": column,
        "feature": feature_name,
        "kind": source_kind,
        "unique_values": unique_values,
        "rows_used": int(pd.concat([numeric, target], axis=1).dropna().shape[0]),
        "pearson": pearson,
        "abs_pearson": abs(pearson) if not np.isnan(pearson) else np.nan,
        "spearman": spearman,
        "abs_spearman": abs(spearman) if not np.isnan(spearman) else np.nan,
        "eta": np.nan,
        "eta_squared": np.nan,
        "strength": max(strength_values) if strength_values else np.nan,
        "direction_or_best_level": direction,
    }


def summarize_categorical_column(
    column: str,
    series: pd.Series,
    target: pd.Series,
) -> dict[str, object]:
    frame = pd.DataFrame({"category": series.astype("string"), TARGET_COLUMN: target}).dropna()
    eta = correlation_ratio(frame["category"], frame[TARGET_COLUMN])
    unique_values = int(frame["category"].nunique())

    best_level = ""
    if unique_values > 0:
        level_summary = (
            frame.groupby("category", dropna=True)[TARGET_COLUMN]
            .agg(["count", "median", "mean"])
            .reset_index()
            .sort_values(["median", "mean", "category"], kind="mergesort")
        )
        best = level_summary.iloc[0]
        best_level = f"{best['category']} (median_ratio={best['median']:.6g}, n={int(best['count'])})"

    return {
        "parameter": column,
        "feature": "raw_category",
        "kind": "categorical",
        "unique_values": unique_values,
        "rows_used": len(frame),
        "pearson": np.nan,
        "abs_pearson": np.nan,
        "spearman": np.nan,
        "abs_spearman": np.nan,
        "eta": eta,
        "eta_squared": eta * eta if not np.isnan(eta) else np.nan,
        "strength": eta,
        "direction_or_best_level": best_level,
    }


def build_parameter_ranking(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_categorical_levels: int,
) -> pd.DataFrame:
    target = df[TARGET_COLUMN]
    rows = []

    for column in feature_columns:
        numeric = as_numeric(df[column])
        numeric_unique = int(numeric.dropna().nunique())
        raw_unique = int(df[column].dropna().nunique())

        if raw_unique < 2:
            continue

        non_null_count = int(df[column].notna().sum())
        is_plain_numeric = numeric.notna().sum() == non_null_count and numeric_unique >= 2

        if is_plain_numeric:
            rows.append(summarize_numeric_feature(column, "raw", numeric, target, "numeric"))
        else:
            for feature_name, feature_values in build_embedded_numeric_features(df[column]).items():
                if feature_values.dropna().nunique() >= 2:
                    rows.append(
                        summarize_numeric_feature(
                            column,
                            feature_name,
                            feature_values,
                            target,
                            "embedded_numeric",
                        )
                    )

            if raw_unique <= max_categorical_levels:
                rows.append(summarize_categorical_column(column, df[column], target))

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["strength", "parameter"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def build_one_hot_ranking(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_categorical_levels: int,
) -> pd.DataFrame:
    target = df[TARGET_COLUMN]
    rows = []

    for column in feature_columns:
        unique_count = int(df[column].dropna().nunique())
        if unique_count < 2 or unique_count > max_categorical_levels:
            continue

        values = df[column].astype("string").fillna("<NA>")
        for level in sorted(values.unique()):
            indicator = (values == level).astype(float)
            pearson = pearson_corr(indicator, target)
            if np.isnan(pearson):
                continue
            level_target = target[indicator == 1]
            other_target = target[indicator == 0]
            rows.append(
                {
                    "parameter": column,
                    "level": level,
                    "rows_with_level": int(indicator.sum()),
                    "pearson": pearson,
                    "abs_pearson": abs(pearson),
                    "median_ratio_for_level": float(level_target.median()),
                    "median_ratio_without_level": float(other_target.median()),
                    "mean_ratio_for_level": float(level_target.mean()),
                    "mean_ratio_without_level": float(other_target.mean()),
                }
            )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["abs_pearson", "parameter", "level"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_fastest_levels(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_categorical_levels: int,
) -> pd.DataFrame:
    rows = []

    for column in feature_columns:
        unique_count = int(df[column].dropna().nunique())
        if unique_count < 2 or unique_count > max_categorical_levels:
            continue

        summary = (
            df.groupby(column, dropna=False)[TARGET_COLUMN]
            .agg(["count", "median", "mean", "min", "max"])
            .reset_index()
            .rename(columns={column: "level"})
        )
        summary.insert(0, "parameter", column)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["median", "mean", "parameter", "level"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_fastest_combinations(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in COMBINATION_COLUMNS if column in df.columns]
    if not columns:
        return pd.DataFrame()

    return (
        df.groupby(columns, dropna=False)[TARGET_COLUMN]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
        .sort_values(["median", "mean"] + columns, kind="mergesort")
        .reset_index(drop=True)
    )


def print_table(title: str, df: pd.DataFrame, columns: list[str], top_n: int) -> None:
    print(f"\n{title}")
    if df.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df.loc[:, columns].head(top_n).to_string(index=False))


def write_reports(
    *,
    ranking: pd.DataFrame,
    one_hot_ranking: pd.DataFrame,
    fastest_levels: pd.DataFrame,
    fastest_combinations: pd.DataFrame,
    include_time_limits: bool,
    start_column: str,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_tag = "with_time_limits" if include_time_limits else "non_time_limit"
    start_tag = start_column.replace(" ", "_")
    report_prefix = OUTPUT_DIR / (
        f"runtime_ratio_from_{start_tag}_analysis_{status_tag}_{timestamp}"
    )

    outputs = [
        (report_prefix.with_name(report_prefix.name + "_correlations.csv"), ranking),
        (report_prefix.with_name(report_prefix.name + "_one_hot_levels.csv"), one_hot_ranking),
        (report_prefix.with_name(report_prefix.name + "_fastest_levels.csv"), fastest_levels),
        (report_prefix.with_name(report_prefix.name + "_fastest_combinations.csv"), fastest_combinations),
    ]

    written_paths = []
    for output_path, frame in outputs:
        frame.to_csv(output_path, index=False)
        written_paths.append(output_path)
    return written_paths


def main() -> None:
    args = parse_args()
    df = load_csvs(CSV_PATHS)
    analyzed, counts = prepare_rows(
        df,
        include_time_limits=args.include_time_limits,
        benchmark_timeout_milp=args.benchmark_timeout_milp,
    )

    if analyzed.empty:
        raise ValueError("No rows available after filtering.")

    feature_columns = get_feature_columns(analyzed, args.start_column)
    ranking = build_parameter_ranking(
        analyzed,
        feature_columns,
        args.max_categorical_levels,
    )
    one_hot_ranking = build_one_hot_ranking(
        analyzed,
        feature_columns,
        args.max_categorical_levels,
    )
    fastest_levels = build_fastest_levels(
        analyzed,
        feature_columns,
        args.max_categorical_levels,
    )
    fastest_combinations = build_fastest_combinations(analyzed)

    print("Runtime-to-benchmark-ratio parameter analysis")
    print(f"CSV files: {len(CSV_PATHS)}")
    print(f"Loaded rows: {counts['loaded_rows']}")
    print(f"Rows skipped for blank {RAW_RUNTIME_COLUMN}: {counts['blank_runtime_rows']}")
    print(f"TIME_LIMIT rows excluded: {counts['excluded_time_limit_rows']}")
    if counts["benchmark_timeout_milp"] is None:
        print("Benchmark timeout_milp_s filter: any")
    else:
        print(f"Benchmark timeout_milp_s filter: {counts['benchmark_timeout_milp']:g}")
    print(f"Benchmark rows: {counts['benchmark_rows']}")
    print(f"Benchmark targets: {counts['benchmark_targets']}")
    print(f"Rows without matching benchmark: {counts['rows_without_matching_benchmark']}")
    print(f"Rows with zero benchmark runtime: {counts['rows_with_zero_benchmark_runtime']}")
    print(f"Rows analyzed: {counts['analyzed_rows']}")
    print(f"Correlation target: {TARGET_COLUMN}")
    print(f"Feature columns analyzed: {len(feature_columns)} columns from {args.start_column!r} onward")

    print_table(
        "Strongest parameter associations",
        ranking,
        [
            "parameter",
            "feature",
            "kind",
            "unique_values",
            "rows_used",
            "strength",
            "pearson",
            "spearman",
            "eta",
            "direction_or_best_level",
        ],
        args.top_n,
    )
    print_table(
        "Strongest one-hot level correlations",
        one_hot_ranking,
        [
            "parameter",
            "level",
            "rows_with_level",
            "pearson",
            "median_ratio_for_level",
            "median_ratio_without_level",
        ],
        args.top_n,
    )
    print_table(
        "Lowest-ratio individual parameter levels",
        fastest_levels,
        ["parameter", "level", "count", "median", "mean", "min", "max"],
        args.top_n,
    )
    print_table(
        "Lowest-ratio parameter combinations",
        fastest_combinations,
        COMBINATION_COLUMNS + ["count", "median", "mean", "min", "max"],
        args.top_n,
    )

    if not args.no_save:
        output_paths = write_reports(
            ranking=ranking,
            one_hot_ranking=one_hot_ranking,
            fastest_levels=fastest_levels,
            fastest_combinations=fastest_combinations,
            include_time_limits=args.include_time_limits,
            start_column=args.start_column,
        )
        print("\nWrote reports:")
        for output_path in output_paths:
            print(f"- {output_path}")


if __name__ == "__main__":
    main()
