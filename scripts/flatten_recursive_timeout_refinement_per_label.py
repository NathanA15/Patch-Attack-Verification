#!/usr/bin/env python3
"""
Flatten a recursive timeout refinement summary CSV into per-label attempt rows.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_verifier import MILP_STATUS, format_milp_status, parse_log_details


OUTPUT_COLUMNS = [
    "source_csv_path",
    "source_row_number",
    "image_index",
    "true_label",
    "adv_label",
    "patch_x",
    "patch_y",
    "patch_size",
    "upper_bound",
    "timeout_milp_s",
    "add_bool_constraints",
    "use_refine_poly",
    "middle_bound",
    "run_scope",
    "global_attempt_index",
    "label_attempt_index",
    "current_depth",
    "status_code",
    "status_name",
    "final_outcome",
    "is_timeout",
    "is_adversarial_found",
    "iteration_runtime_seconds",
    "iteration_runtime_source",
    "attempt_runtime_seconds",
    "parent_total_run_time_seconds",
    "parent_status",
    "parent_attempt_count",
    "parent_max_depth_reached",
    "log_path",
]


COUNTER_PATTERN = re.compile(
    r"Counter is\s+\d+(?:\s+candidate label is\s+(\d+)\s+adv label is\s+(\d+)|\s+label is\s+(\d+))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten recursive timeout refinement summary CSV into per-label rows."
    )
    parser.add_argument("input_csv", type=Path, help="Path to recursive timeout refinement CSV.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <input>_per_label.csv.",
    )
    return parser.parse_args()


def parse_python_literal(value: str, field_name: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse {field_name}: {value!r}") from exc


def parse_timedelta_seconds(value: str) -> float:
    raw = value.strip()
    day_match = re.fullmatch(r"(?:(\d+)\s+day[s]?,\s+)?(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", raw)
    if not day_match:
        raise ValueError(f"Unsupported timedelta string: {value!r}")

    days = int(day_match.group(1) or 0)
    hours = int(day_match.group(2))
    minutes = int(day_match.group(3))
    seconds = float(day_match.group(4))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_optional_int(text: str, key: str) -> int | None:
    match = re.search(rf"'{re.escape(key)}':\s*(-?\d+)", text)
    return int(match.group(1)) if match else None


def parse_optional_float(text: str, key: str) -> float | None:
    match = re.search(rf"'{re.escape(key)}':\s*(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def parse_optional_bool(text: str, key: str) -> bool | None:
    match = re.search(rf"'{re.escape(key)}':\s*(True|False)", text)
    if not match:
        return None
    return match.group(1) == "True"


def parse_true_label(text: str) -> int:
    value = parse_optional_int(text, "label")
    if value is None:
        raise ValueError("Could not parse true label from log header.")
    return value


def parse_log_config(text: str) -> dict[str, object]:
    return {
        "add_bool_constraints": parse_optional_bool(text, "add_bool_constraints"),
        "use_refine_poly": parse_optional_bool(text, "use_refine_poly"),
        "middle_bound": parse_optional_float(text, "middle_bound"),
    }


def iter_label_blocks(text: str) -> Iterable[dict[str, object]]:
    matches = list(COUNTER_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        candidate_label = int(match.group(1)) if match.group(1) is not None else None
        adv_label = int(match.group(2) if match.group(2) is not None else match.group(3))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        yield {
            "candidate_label": candidate_label,
            "adv_label": adv_label,
            "block_text": text[start:end],
        }


def parse_block_local_runtime_seconds(block_text: str) -> tuple[float | None, str]:
    verified_match = re.search(r"Time taken to successfully verify:\s*([0-9]+(?:\.[0-9]+)?)", block_text)
    if verified_match:
        return float(verified_match.group(1)), "initial_label_local_verify_time"

    failure_match = re.search(
        r"Time taken and terminate on failure:\s*([0-9]+(?:\.[0-9]+)?)", block_text
    )
    if failure_match:
        return float(failure_match.group(1)), "initial_label_local_failure_time"

    return None, "initial_label_local_missing"


def classify_final_outcome(label_result: dict[str, object]) -> str:
    if label_result.get("model_found_adv"):
        return "adversarial"
    if label_result.get("verified"):
        return "verified"
    if label_result.get("timed_out"):
        return "timeout"
    if label_result.get("failed"):
        return "failed"
    return "failed"


def build_default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_per_label.csv")


def normalize_source_row(source_row: dict[str, str]) -> dict[str, object]:
    eram_run_times = parse_python_literal(source_row["ERAN Run Times"], "ERAN Run Times")
    log_paths = [path for path in source_row["Log paths"].split(";") if path]

    if len(eram_run_times) != len(log_paths):
        raise ValueError(
            "Mismatch between ERAN Run Times and Log paths counts "
            f"({len(eram_run_times)} != {len(log_paths)}) for row {source_row!r}"
        )

    return {
        "image_index": int(source_row["Image Index"]),
        "patch_x": int(source_row["X patch"]),
        "patch_y": int(source_row["Y patch"]),
        "patch_size": int(source_row["Patch Size"]),
        "upper_bound": float(source_row["Upper bound"]),
        "timeout_milp_s": float(source_row["Timeout MILP (s)"]),
        "attempt_runtimes_seconds": [parse_timedelta_seconds(str(value)) for value in eram_run_times],
        "total_run_time_seconds": float(source_row["Total Run Time (s)"]),
        "parent_status": source_row["Status"],
        "parent_attempt_count": int(source_row["Attempt Count"]),
        "parent_max_depth_reached": int(source_row["Max Depth Reached"]),
        "log_paths": log_paths,
    }


def build_rows_for_attempt(
    *,
    source_csv_path: Path,
    source_row_number: int,
    parent_row: dict[str, object],
    global_attempt_index: int,
    log_path: Path,
    attempt_runtime_seconds: float,
    label_attempt_counts: dict[int, int],
) -> list[dict[str, object]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    text = log_path.read_text()
    parsed = parse_log_details(text)
    true_label = parse_true_label(text)
    log_config = parse_log_config(text)

    label_results_by_adv = {
        int(result["adv_label"]): result for result in parsed["label_results"]
    }
    block_infos = list(iter_label_blocks(text))

    if global_attempt_index == 0:
        run_scope = "initial_all_labels"
    else:
        run_scope = "single_label_rerun"
        if len(block_infos) != 1:
            raise ValueError(
                f"Expected exactly one label block in rerun log {log_path}, found {len(block_infos)}."
            )

    rows = []
    for block_info in block_infos:
        adv_label = int(block_info["adv_label"])
        label_result = label_results_by_adv.get(adv_label)
        if label_result is None:
            raise ValueError(f"Missing parsed label result for adv_label={adv_label} in {log_path}.")

        label_attempt_index = label_attempt_counts.get(adv_label, 0) + 1
        label_attempt_counts[adv_label] = label_attempt_index
        current_depth = label_attempt_index - 1

        if global_attempt_index == 0:
            iteration_runtime_seconds, runtime_source = parse_block_local_runtime_seconds(
                str(block_info["block_text"])
            )
        else:
            iteration_runtime_seconds = attempt_runtime_seconds
            runtime_source = "rerun_attempt_runtime"

        final_outcome = classify_final_outcome(label_result)
        status_code = label_result.get("milp_status")
        is_timeout = final_outcome == "timeout" or status_code == MILP_STATUS["TIME_LIMIT"]
        is_adversarial_found = final_outcome == "adversarial"

        rows.append(
            {
                "source_csv_path": str(source_csv_path),
                "source_row_number": source_row_number,
                "image_index": parent_row["image_index"],
                "true_label": true_label,
                "adv_label": adv_label,
                "patch_x": parent_row["patch_x"],
                "patch_y": parent_row["patch_y"],
                "patch_size": parent_row["patch_size"],
                "upper_bound": parent_row["upper_bound"],
                "timeout_milp_s": parent_row["timeout_milp_s"],
                "add_bool_constraints": log_config["add_bool_constraints"],
                "use_refine_poly": log_config["use_refine_poly"],
                "middle_bound": log_config["middle_bound"],
                "run_scope": run_scope,
                "global_attempt_index": global_attempt_index,
                "label_attempt_index": label_attempt_index,
                "current_depth": current_depth,
                "status_code": status_code,
                "status_name": format_milp_status(status_code),
                "final_outcome": final_outcome,
                "is_timeout": is_timeout,
                "is_adversarial_found": is_adversarial_found,
                "iteration_runtime_seconds": iteration_runtime_seconds,
                "iteration_runtime_source": runtime_source,
                "attempt_runtime_seconds": attempt_runtime_seconds,
                "parent_total_run_time_seconds": parent_row["total_run_time_seconds"],
                "parent_status": parent_row["parent_status"],
                "parent_attempt_count": parent_row["parent_attempt_count"],
                "parent_max_depth_reached": parent_row["parent_max_depth_reached"],
                "log_path": str(log_path),
            }
        )

    return rows


def flatten_csv(input_csv: Path, output_csv: Path) -> int:
    total_rows = 0

    with input_csv.open(newline="") as infile, output_csv.open("w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for source_row_number, raw_row in enumerate(reader, start=1):
            parent_row = normalize_source_row(raw_row)
            label_attempt_counts: dict[int, int] = {}

            for global_attempt_index, (log_path_str, attempt_runtime_seconds) in enumerate(
                zip(parent_row["log_paths"], parent_row["attempt_runtimes_seconds"])
            ):
                attempt_rows = build_rows_for_attempt(
                    source_csv_path=input_csv,
                    source_row_number=source_row_number,
                    parent_row=parent_row,
                    global_attempt_index=global_attempt_index,
                    log_path=Path(log_path_str),
                    attempt_runtime_seconds=attempt_runtime_seconds,
                    label_attempt_counts=label_attempt_counts,
                )
                writer.writerows(attempt_rows)
                total_rows += len(attempt_rows)

    return total_rows


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = (args.output_csv or build_default_output_path(input_csv)).resolve()

    row_count = flatten_csv(input_csv, output_csv)
    print(f"Wrote {row_count} flattened rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
