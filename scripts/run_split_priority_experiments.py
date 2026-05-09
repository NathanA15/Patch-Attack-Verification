"""
Controlled experiments for singleton-selector skipping and split-bit branching.

This uses the same image/patch/upper-bound setup as the 20260424_134749
investigation and runs each formulation variant against the original timeout
schedules.
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


IMAGE_INDEX = 2
PATCH_X = 0
PATCH_Y = 0
PATCH_SIZE = 10
UPPER_BOUND = 0.75
MAX_DEPTH = 4
TOP_K = 30
ADD_BOOL_CONSTRAINTS = True
USE_REFINE_POLY = False
SAVE_CSV = True
CSV_PATH = None

SPLIT_BIT_BRANCH_PRIORITY = 1000
CORE_24_SPLIT_INDICES = [
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    254,
    256,
    257,
    258,
    260,
    261,
]

TIMEOUT_SCHEDULES = [
    [720000],
    [10, 720000],
    [10, 10, 720000],
    [10, 10, 10, 720000],
]

FORMULATION_VARIANTS = [
    {
        "name": "current_formulation",
        "skip_singleton_bounds": False,
        "enable_split_bit_branch_priority": False,
        "split_bit_branch_priority": SPLIT_BIT_BRANCH_PRIORITY,
        "fixed_split_indices": None,
    },
    {
        "name": "skip_singleton_only",
        "skip_singleton_bounds": True,
        "enable_split_bit_branch_priority": False,
        "split_bit_branch_priority": SPLIT_BIT_BRANCH_PRIORITY,
        "fixed_split_indices": None,
    },
    {
        "name": "skip_singleton_branch_priority",
        "skip_singleton_bounds": True,
        "enable_split_bit_branch_priority": True,
        "split_bit_branch_priority": SPLIT_BIT_BRANCH_PRIORITY,
        "fixed_split_indices": None,
    },
    {
        "name": "skip_singleton_branch_priority_core24",
        "skip_singleton_bounds": True,
        "enable_split_bit_branch_priority": True,
        "split_bit_branch_priority": SPLIT_BIT_BRANCH_PRIORITY,
        "fixed_split_indices": CORE_24_SPLIT_INDICES,
    },
]


def normalize(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_run_id(batch_run_id, variant_name, timeout_milps, schedule_index):
    timeout_tag = "_".join(normalize(value) for value in timeout_milps)
    return (
        f"{batch_run_id}_"
        f"{variant_name}_"
        f"schedule_{schedule_index:02d}_"
        f"ub_{normalize(UPPER_BOUND)}_"
        f"timeouts_{timeout_tag}"
    )


def main():
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_split_priority")
    runs_csv_path = resolve_runs_csv_path(CSV_PATH, batch_run_id)

    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    print("Split priority controlled experiment settings")
    print(f"image_index={IMAGE_INDEX} patch=({PATCH_X}, {PATCH_Y}) size={PATCH_SIZE}")
    print(f"upper_bound={UPPER_BOUND} max_depth={MAX_DEPTH} top_k={TOP_K}")
    print(f"timeout_schedules={TIMEOUT_SCHEDULES}")
    print(f"variants={[variant['name'] for variant in FORMULATION_VARIANTS]}")
    if SAVE_CSV:
        print(f"details_csv={runs_csv_path}")

    results = []
    for variant in FORMULATION_VARIANTS:
        for schedule_index, timeout_milps in enumerate(TIMEOUT_SCHEDULES, start=1):
            print("\n" + "#" * 80)
            print(
                f"Starting variant={variant['name']} "
                f"timeout_schedule={timeout_milps}"
            )
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
                ul=UPPER_BOUND,
                add_bool_constraints=ADD_BOOL_CONSTRAINTS,
                use_refine_poly=USE_REFINE_POLY,
                skip_singleton_bounds=variant["skip_singleton_bounds"],
                enable_split_bit_branch_priority=variant["enable_split_bit_branch_priority"],
                split_bit_branch_priority=variant["split_bit_branch_priority"],
                fixed_split_indices=variant["fixed_split_indices"],
                run_id=build_run_id(
                    batch_run_id,
                    variant["name"],
                    timeout_milps,
                    schedule_index,
                ),
                runs_csv_path=runs_csv_path,
                save_csv=SAVE_CSV,
            )
            results.append(
                {
                    "variant": variant["name"],
                    "timeout_schedule": timeout_milps,
                    "result": result,
                }
            )

    print("\n" + "=" * 80)
    print("Controlled experiment summary")
    print(f"CSV path: {runs_csv_path}")
    for item in results:
        result = item["result"]
        print(
            f"variant={item['variant']} "
            f"timeout_schedule={item['timeout_schedule']} "
            f"status={result['status']} "
            f"total_runtime={result['total_runtime_seconds']:.2f}s "
            f"eran_runs={result['attempt_count']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
