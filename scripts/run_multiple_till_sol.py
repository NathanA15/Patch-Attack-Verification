"""
Batch runner for recursive MNIST patch verification across upper bounds
and per-depth timeout schedules.
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
IMAGE_INDEX = 12
PATCH_X = 0
PATCH_Y = 0
PATCH_SIZE = 10
MAX_DEPTH = 4
TOP_K = 30
ADD_BOOL_CONSTRAINTS = True
USE_REFINE_POLY = False
SAVE_CSV = True
CSV_PATH = None

# One entry per run. `timeout_milps[0]` is used for the initial all-labels
# attempt, `timeout_milps[1]` for the first split depth, and so on. When the
# recursion reaches a deeper depth than the list provides, the last timeout is
# reused.
RUN_SCHEDULE = [
    # {"upper_bound": 0.70, "timeout_milps": [720000]},
    # {"upper_bound": 0.70, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.70, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.70, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 0.75, "timeout_milps": [720000]},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 0.80, "timeout_milps": [720000]},
    # {"upper_bound": 0.80, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.80, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.80, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 0.85, "timeout_milps": [720000]},
    # {"upper_bound": 0.85, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.85, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.85, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 0.90, "timeout_milps": [720000]},
    # {"upper_bound": 0.90, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.90, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.90, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 0.95, "timeout_milps": [720000]},
    # {"upper_bound": 0.95, "timeout_milps": [10, 720000]},
    # {"upper_bound": 0.95, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 0.95, "timeout_milps": [10, 10, 10, 720000]},

    # {"upper_bound": 1.0, "timeout_milps": [720000]},
    # {"upper_bound": 1.0, "timeout_milps": [10, 720000]},
    # {"upper_bound": 1.0, "timeout_milps": [10, 10, 720000]},
    # {"upper_bound": 1.0, "timeout_milps": [10, 10, 10, 720000]},
    
    # {"upper_bound": 1.0, "timeout_milps": [10]},
    # {"upper_bound": 1.0, "timeout_milps": [10, 10]},
    # {"upper_bound": 1.0, "timeout_milps": [10, 10, 10]},
    {"upper_bound": 1.0, "timeout_milps": [10, 10, 10, 10]},

]
# RUN_SCHEDULE = [
#     {"upper_bound": 0.70, "timeout_milps": [72000]},
#     {"upper_bound": 0.75, "timeout_milps": [72000]},
#     {"upper_bound": 0.80, "timeout_milps": [72000]},
# ]

def resolve_run_schedule(run_schedule):
    resolved_schedule = []

    for index, item in enumerate(run_schedule, start=1):
        if "upper_bound" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'upper_bound'.")
        if "timeout_milps" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'timeout_milps'.")

        upper_bound = float(item["upper_bound"])
        timeout_milps = [float(value) for value in item["timeout_milps"]]

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


def build_run_id(batch_run_id, upper_bound, timeout_milps, schedule_index):
    def normalize(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")

    timeout_tag = "_".join(normalize(value) for value in timeout_milps)
    return (
        f"{batch_run_id}_"
        f"run_{schedule_index:02d}_"
        f"ub_{normalize(upper_bound)}_"
        f"timeouts_{timeout_tag}"
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
        print(f"details_csv={runs_csv_path}")

    batch_results = []

    for schedule_index, schedule_item in enumerate(run_schedule, start=1):
        upper_bound = schedule_item["upper_bound"]
        timeout_milps = schedule_item["timeout_milps"]

        print("\n" + "#" * 80)
        print(f"Starting upper_bound={upper_bound} with timeout schedule {timeout_milps}")
        print("#" * 80)

        result = run_verifier.verify_image_with_recursive_timeout_refinement(
            img_index=IMAGE_INDEX,
            pixels=pixels,
            labels=labels,
            x_box=PATCH_X,
            y_box=PATCH_Y,
            size_box=PATCH_SIZE,
            timeout_milp=timeout_milps,
            max_depth=MAX_DEPTH,
            top_k=TOP_K,
            ul=upper_bound,
            add_bool_constraints=ADD_BOOL_CONSTRAINTS,
            use_refine_poly=USE_REFINE_POLY,
            run_id=build_run_id(batch_run_id, upper_bound, timeout_milps, schedule_index),
            runs_csv_path=runs_csv_path,
            save_csv=SAVE_CSV,
        )

        batch_results.append(
            {
                "upper_bound": upper_bound,
                "timeout_schedule": timeout_milps,
                "result": result,
            }
        )

        if is_resolved(result):
            print(
                f"Resolved upper_bound={upper_bound} with timeout_schedule={timeout_milps} "
                f"status={result['status']} total_runtime={result['total_runtime_seconds']:.2f}s"
            )
        else:
            print(
                f"upper_bound={upper_bound} stayed unresolved with timeout_schedule={timeout_milps}"
            )

    print("\n" + "=" * 80)
    print("Batch summary")
    print(f"CSV path: {runs_csv_path}")
    for item in batch_results:
        result = item["result"]
        print(
            f"upper_bound={item['upper_bound']} timeout_schedule={item['timeout_schedule']} "
            f"status={result['status']} "
            f"total_runtime={result['total_runtime_seconds']:.2f}s "
            f"eran_runs={result['attempt_count']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
