"""
Batch runner for recursive MNIST patch verification across upper bounds
and timeout retries.
"""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_verifier
from config import MNIST_DATA_PATH
from utils import resolve_runs_csv_path


# Edit these globals to define what the script will run.
IMAGE_INDEX = 2
PATCH_X = 0
PATCH_Y = 0
PATCH_SIZE = 10
MAX_DEPTH = 4
TOP_K = 30
ADD_BOOL_CONSTRAINTS = True
USE_REFINE_POLY = False
SAVE_CSV = True
CSV_PATH = None

# One entry per upper bound. For each upper bound, the timeout list is tried
# in order until that bound resolves or the list is exhausted.
RUN_SCHEDULE = [
    {"upper_bound": 0.70, "timeout_milps": [15, 30, 60, 120, 200, 270, 450]},
    {"upper_bound": 0.75, "timeout_milps": [200, 600, 1200, 3000, 6500]},
    {"upper_bound": 0.80, "timeout_milps": [600,1000,5000,10000,15000,30000]},
]


def dedupe_preserving_order(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def resolve_run_schedule(run_schedule):
    resolved_schedule = []

    for index, item in enumerate(run_schedule, start=1):
        if "upper_bound" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'upper_bound'.")
        if "timeout_milps" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'timeout_milps'.")

        upper_bound = float(item["upper_bound"])
        timeout_milps = dedupe_preserving_order(float(value) for value in item["timeout_milps"])

        if not timeout_milps:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has no timeout values."
            )

        resolved_schedule.append(
            {
                "upper_bound": upper_bound,
                "timeout_milps": timeout_milps,
            }
        )

    if not resolved_schedule:
        raise ValueError("RUN_SCHEDULE is empty.")

    return resolved_schedule


def build_run_id(batch_run_id, upper_bound, timeout_milp, try_index):
    def normalize(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return (
        f"{batch_run_id}_"
        f"ub_{normalize(upper_bound)}_"
        f"timeout_{normalize(timeout_milp)}_"
        f"try_{try_index:02d}"
    )


def is_resolved(result):
    return not result["unresolved_labels"]


def main():
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_schedule = resolve_run_schedule(RUN_SCHEDULE)
    runs_csv_path = resolve_runs_csv_path(CSV_PATH, batch_run_id)

    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    print("Batch recursive verification settings")
    print(f"image_index={IMAGE_INDEX} patch=({PATCH_X}, {PATCH_Y}) size={PATCH_SIZE}")
    print(f"max_depth={MAX_DEPTH} top_k={TOP_K}")
    print(f"run_schedule={run_schedule}")
    if SAVE_CSV:
        print(f"summary_csv={runs_csv_path}")

    batch_results = []

    for schedule_item in run_schedule:
        upper_bound = schedule_item["upper_bound"]
        timeout_milps = schedule_item["timeout_milps"]
        resolved_for_upper_bound = False

        print("\n" + "#" * 80)
        print(f"Starting upper_bound={upper_bound} with timeout schedule {timeout_milps}")
        print("#" * 80)

        for try_index, timeout_milp in enumerate(timeout_milps, start=1):
            print(
                f"\nRunning upper_bound={upper_bound} "
                f"timeout_milp={timeout_milp} try={try_index}/{len(timeout_milps)}"
            )

            result = run_verifier.verify_image_with_recursive_timeout_refinement(
                img_index=IMAGE_INDEX,
                pixels=pixels,
                labels=labels,
                x_box=PATCH_X,
                y_box=PATCH_Y,
                size_box=PATCH_SIZE,
                timeout_milp=timeout_milp,
                max_depth=MAX_DEPTH,
                top_k=TOP_K,
                ul=upper_bound,
                add_bool_constraints=ADD_BOOL_CONSTRAINTS,
                use_refine_poly=USE_REFINE_POLY,
                run_id=build_run_id(batch_run_id, upper_bound, timeout_milp, try_index),
                runs_csv_path=runs_csv_path,
                save_csv=SAVE_CSV,
            )

            batch_results.append(
                {
                    "upper_bound": upper_bound,
                    "timeout_milp": timeout_milp,
                    "try_index": try_index,
                    "result": result,
                }
            )

            if is_resolved(result):
                resolved_for_upper_bound = True
                print(
                    f"Resolved upper_bound={upper_bound} with timeout_milp={timeout_milp} "
                    f"status={result['status']} total_runtime={result['total_runtime_seconds']:.2f}s"
                )
                break

            print(
                f"upper_bound={upper_bound} remained unresolved with timeout_milp={timeout_milp}; "
                "trying the next timeout."
            )

        if not resolved_for_upper_bound:
            print(
                f"upper_bound={upper_bound} stayed unresolved after timeout schedule {timeout_milps}"
            )

    print("\n" + "=" * 80)
    print("Batch summary")
    print(f"CSV path: {runs_csv_path}")
    for item in batch_results:
        result = item["result"]
        print(
            f"upper_bound={item['upper_bound']} timeout_milp={item['timeout_milp']} "
            f"try={item['try_index']} status={result['status']} "
            f"total_runtime={result['total_runtime_seconds']:.2f}s "
            f"eran_runs={result['attempt_count']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
