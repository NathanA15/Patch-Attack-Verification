"""
Run the timeout example across multiple parameter settings and append results to runs.csv.
"""

import csv
from datetime import datetime
import os
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_verifier
from config import LOG_DIR, MNIST_DATA_PATH


RUNS_CSV_PATH = Path("/root/Projects/Shuey/Patch-Attack-Verification/runs.csv")

CSV_COLUMNS = [
    "Image Index",
    "X patch",
    "Y patch",
    " Patch Size",
    "Upper bound",
    "MIDDLE BOUND",
    " Run Time (s)",
    "Second Run Time (s)",
    "Status",
    "Used Only MILP",
    " Comment",
    "Log path",
    "CSV path",
]

PARAMETER_RUNS = [
    (0.65, False, None),
    (0.65, True, 0.3),
    (0.65, True, 0.6),
    (0.65, True, 0.05),
    (0.65, True, 0.64),
    (0.7, False, None),
    (0.7, True, 0.6),
    (0.7, True, 0.65),
    (0.7, True, 0.35),
    (0.75, False, None),
    (0.75, True, 0.7),
    (0.75, True, 0.55),
    (0.75, True, 0.35),
    (0.8, False, None),
    (0.8, True, 0.75),
    (0.8, True, 0.5),
]

DEFAULT_MIDDLE_BOUND = 0.5

TIMEOUT_MILP = 604800  # 7 days
USE_REFINE_POLY = False

IMAGE_INDEX = 2
PATCH_SIZE = 10
PATCH_X_Y = (0, 0)


def fresh_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_folder = LOG_DIR / datetime.now().strftime("%Y%m%d")
    daily_folder.mkdir(parents=True, exist_ok=True)
    return daily_folder / f"{timestamp}_eran_run.log"


def ensure_trailing_newline(path: Path):
    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            with path.open("ab") as out_handle:
                out_handle.write(b"\n")


def append_run_row(row):
    RUNS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RUNS_CSV_PATH.exists()
    if file_exists:
        ensure_trailing_newline(RUNS_CSV_PATH)
    with RUNS_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([row.get(col, "") for col in CSV_COLUMNS])


def main():
    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    total_runs = len(PARAMETER_RUNS)

    print(
        f"Starting verification for index {IMAGE_INDEX} with patch size {PATCH_SIZE} at {PATCH_X_Y}"
    )
    print(f"Timeout per run: {TIMEOUT_MILP} seconds")
    print(f"Results will be appended to: {RUNS_CSV_PATH}")
    print("-" * 80)

    for run_idx, (upper_bound, add_bool_constraints, middle_bound) in enumerate(
        PARAMETER_RUNS, start=1
    ):
        display_middle = "" if middle_bound is None else middle_bound
        run_middle = DEFAULT_MIDDLE_BOUND if middle_bound is None else middle_bound
        log_file = fresh_log_file()
        run_verifier.LOG_FILE = log_file

        print(
            f"[{run_idx}/{total_runs}] upper={upper_bound} add_bool={add_bool_constraints} "
            f"middle={display_middle or 'n/a'} log={log_file}"
        )

        try:
            elapsed_time, last_status, _failed_labels, _example, is_adversarial = (
                run_verifier.verify_image(
                    img_index=IMAGE_INDEX,
                    pixels=pixels,
                    labels=labels,
                    x_box=PATCH_X_Y[0],
                    y_box=PATCH_X_Y[1],
                    size_box=PATCH_SIZE,
                    timeout_milp=TIMEOUT_MILP,
                    with_plots=False,
                    ul=upper_bound,
                    add_bool_constraints=add_bool_constraints,
                    use_refine_poly=USE_REFINE_POLY,
                    middle_bound=run_middle,
                )
            )

            time_seconds = (
                elapsed_time.total_seconds()
                if hasattr(elapsed_time, "total_seconds")
                else float(elapsed_time)
            )
            status = "Adversarial" if is_adversarial else "Verified"

            row = {
                "Image Index": IMAGE_INDEX,
                "X patch": PATCH_X_Y[0],
                "Y patch": PATCH_X_Y[1],
                " Patch Size": PATCH_SIZE,
                "Upper bound": upper_bound,
                "MIDDLE BOUND": display_middle,
                " Run Time (s)": int(round(time_seconds)),
                "Second Run Time (s)": "",
                "Status": status,
                "Used Only MILP": True,
                " Comment": "Ran from run_multiple_till_sol.py",
                "Log path": str(log_file),
                "CSV path": "",
            }
            append_run_row(row)

            print(
                f"  {status} (Status: {last_status}, Time: {time_seconds:.2f}s)"
            )

        except Exception as exc:
            row = {
                "Image Index": IMAGE_INDEX,
                "X patch": PATCH_X_Y[0],
                "Y patch": PATCH_X_Y[1],
                " Patch Size": PATCH_SIZE,
                "Upper bound": upper_bound,
                "MIDDLE BOUND": display_middle,
                " Run Time (s)": "",
                "Second Run Time (s)": "",
                "Status": "Error",
                "Used Only MILP": True,
                " Comment": f"error: {exc}",
                "Log path": str(log_file),
                "CSV path": "",
            }
            append_run_row(row)
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 80)
    print("All runs complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
