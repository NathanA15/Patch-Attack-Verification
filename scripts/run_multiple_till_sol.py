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


# Edit these globals to define the default settings for scheduled runs.
IMAGE_INDEX = 2
PATCH_X = 0
PATCH_Y = 0
PATCH_SIZE = 10
MAX_DEPTH = 4
TOP_K = 30
SPLIT_SELECTION_MODE = "max"
SPLIT_RANDOM_SEED = 42
ADD_BOOL_CONSTRAINTS = True
USE_REFINE_POLY = False
SKIP_SINGLETON_BOUNDS = False
ENABLE_SPLIT_BIT_BRANCH_PRIORITY = False # important param !!
SPLIT_BIT_BRANCH_PRIORITY = 1000
FIXED_SPLIT_INDICES = None
SAVE_CSV = True
CSV_PATH = None

# One entry per run.
# Required: `upper_bound` and `timeout_milps`.
# `timeout_milps[0]` is used for the initial attempt, `timeout_milps[1]` for
# the first split depth, and so on. Deeper recursion reuses the last timeout.
# Optional `image_index`, `patch_x`, `patch_y`, `patch_size`, and `max_depth`:
# per-run overrides. Omit them to use the matching globals above.
# Optional `adv_label`: int adversarial label to target. Omit it, or set it to
# -1, to run the current full all-label pass.
# Optional `top_k`: non-negative int number of patch pixels to split on.
# Omit it to use global `TOP_K`.
# Optional `split_selection_mode`: "max", "min", or "random".
# - "max": highest raw relaxed-model gradients first.
# - "min": lowest raw relaxed-model gradients first, meaning most negative.
# - "random": random unique pixels from inside the patch.
# Omit it to use global `SPLIT_SELECTION_MODE`.
# Global-only `SPLIT_RANDOM_SEED`: None or int seed for reproducible random
# split selection. If `FIXED_SPLIT_INDICES` is set, it overrides these split
# choices and the CSV records the effective mode as "fixed".
RUN_SCHEDULE = [
    # Previous comparison runs, kept here for reference.
    # {"upper_bound": 0.75, "timeout_milps": [720000], "adv_label": 7},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "max"},
    # {"upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "max"},

    # {"upper_bound": 0.7, "timeout_milps": [720000], "adv_label": 2},
    # {"upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},
    # {"upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},

    # Previous active random-mode runs, kept here for reference.
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.75, "timeout_milps": [10, 10, 10, 720000], "adv_label": 7, "top_k": 10, "split_selection_mode": "random"},

    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},
    # {"image_index": 2, "patch_x": 0, "patch_y": 0, "patch_size": 10, "max_depth": 4, "upper_bound": 0.7, "timeout_milps": [10, 10, 10, 720000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},

    # Active image 7, patch 0,0 size 9 runs for adv_label=2 and upper_bound=0.8.
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [72000], "adv_label": 2},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "random"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "max"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 10, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 20, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
    {"image_index": 7, "patch_x": 0, "patch_y": 0, "patch_size": 9, "max_depth": 4, "upper_bound": 0.8, "timeout_milps": [10, 10, 10, 72000], "adv_label": 2, "top_k": 30, "split_selection_mode": "min"},
]


def resolve_run_schedule(run_schedule):
    resolved_schedule = []

    for index, item in enumerate(run_schedule, start=1):
        if "upper_bound" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'upper_bound'.")
        if "timeout_milps" not in item:
            raise ValueError(f"RUN_SCHEDULE entry #{index} is missing 'timeout_milps'.")

        upper_bound = float(item["upper_bound"])
        timeout_milps = [float(value) for value in item["timeout_milps"]]
        image_index = int(item.get("image_index", IMAGE_INDEX))
        patch_x = int(item.get("patch_x", PATCH_X))
        patch_y = int(item.get("patch_y", PATCH_Y))
        patch_size = int(item.get("patch_size", PATCH_SIZE))
        max_depth = int(item.get("max_depth", MAX_DEPTH))
        adv_label = int(item.get("adv_label", -1))
        top_k = int(item.get("top_k", TOP_K))
        split_selection_mode = run_verifier.normalize_split_selection_mode(
            item.get("split_selection_mode", SPLIT_SELECTION_MODE)
        )

        if not timeout_milps:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has no timeout values."
            )
        if image_index < 0:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has "
                f"image_index={image_index}; image_index must be non-negative."
            )
        if patch_x < 0 or patch_y < 0:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has "
                f"patch=({patch_x}, {patch_y}); patch coordinates must be non-negative."
            )
        if patch_size <= 0:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has "
                f"patch_size={patch_size}; patch_size must be positive."
            )
        if max_depth < 0:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has "
                f"max_depth={max_depth}; max_depth must be non-negative."
            )
        if top_k < 0:
            raise ValueError(
                f"RUN_SCHEDULE entry #{index} for upper_bound={upper_bound} has top_k={top_k}; "
                "top_k must be non-negative."
            )

        resolved_schedule.append(
            {
                "image_index": image_index,
                "patch_x": patch_x,
                "patch_y": patch_y,
                "patch_size": patch_size,
                "max_depth": max_depth,
                "upper_bound": upper_bound,
                "timeout_milps": timeout_milps,
                "adv_label": adv_label,
                "top_k": top_k,
                "split_selection_mode": split_selection_mode,
            }
        )

    if not resolved_schedule:
        raise ValueError("RUN_SCHEDULE is empty.")

    return resolved_schedule


def build_run_id(
    batch_run_id,
    image_index,
    patch_x,
    patch_y,
    patch_size,
    max_depth,
    upper_bound,
    timeout_milps,
    schedule_index,
    adv_label=-1,
):
    def normalize(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")

    timeout_tag = "_".join(normalize(value) for value in timeout_milps)
    label_tag = "" if adv_label == -1 else f"adv_label_{adv_label}_"
    return (
        f"{batch_run_id}_"
        f"run_{schedule_index:02d}_"
        f"img_{image_index}_"
        f"patch_{patch_x}_{patch_y}_size_{patch_size}_"
        f"max_depth_{max_depth}_"
        f"ub_{normalize(upper_bound)}_"
        f"{label_tag}"
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
    print(f"default image_index={IMAGE_INDEX} patch=({PATCH_X}, {PATCH_Y}) size={PATCH_SIZE}")
    print(
        f"default max_depth={MAX_DEPTH} top_k={TOP_K} "
        f"split_selection_mode={SPLIT_SELECTION_MODE} "
        f"split_random_seed={SPLIT_RANDOM_SEED}"
    )
    print(
        f"skip_singleton_bounds={SKIP_SINGLETON_BOUNDS} "
        f"enable_split_bit_branch_priority={ENABLE_SPLIT_BIT_BRANCH_PRIORITY} "
        f"split_bit_branch_priority={SPLIT_BIT_BRANCH_PRIORITY}"
    )
    if FIXED_SPLIT_INDICES is not None:
        print(f"fixed_split_indices={FIXED_SPLIT_INDICES}")
    print(f"run_schedule={run_schedule}")
    if SAVE_CSV:
        print(f"details_csv={runs_csv_path} (includes parsed Gurobi statistics)")

    batch_results = []

    for schedule_index, schedule_item in enumerate(run_schedule, start=1):
        image_index = schedule_item["image_index"]
        patch_x = schedule_item["patch_x"]
        patch_y = schedule_item["patch_y"]
        patch_size = schedule_item["patch_size"]
        max_depth = schedule_item["max_depth"]
        upper_bound = schedule_item["upper_bound"]
        timeout_milps = schedule_item["timeout_milps"]
        adv_label = schedule_item["adv_label"]
        top_k = schedule_item["top_k"]
        split_selection_mode = schedule_item["split_selection_mode"]

        print("\n" + "#" * 80)
        label_scope = "all labels" if adv_label == -1 else f"adv_label={adv_label}"
        print(
            f"Starting image_index={image_index} patch=({patch_x}, {patch_y}) "
            f"size={patch_size} max_depth={max_depth} "
            f"upper_bound={upper_bound} {label_scope} "
            f"with timeout schedule {timeout_milps} "
            f"top_k={top_k} split_selection_mode={split_selection_mode}"
        )
        print("#" * 80)

        result = run_verifier.verify_image_with_recursive_timeout_refinement(
            img_index=image_index,
            pixels=pixels,
            labels=labels,
            x_box=patch_x,
            y_box=patch_y,
            size_box=patch_size,
            timeout_milp=timeout_milps,
            max_depth=max_depth,
            top_k=top_k,
            split_selection_mode=split_selection_mode,
            split_random_seed=SPLIT_RANDOM_SEED,
            ul=upper_bound,
            add_bool_constraints=ADD_BOOL_CONSTRAINTS,
            use_refine_poly=USE_REFINE_POLY,
            skip_singleton_bounds=SKIP_SINGLETON_BOUNDS,
            enable_split_bit_branch_priority=ENABLE_SPLIT_BIT_BRANCH_PRIORITY,
            split_bit_branch_priority=SPLIT_BIT_BRANCH_PRIORITY,
            fixed_split_indices=FIXED_SPLIT_INDICES,
            adv_label=adv_label,
            run_id=build_run_id(
                batch_run_id,
                image_index,
                patch_x,
                patch_y,
                patch_size,
                max_depth,
                upper_bound,
                timeout_milps,
                schedule_index,
                adv_label,
            ),
            runs_csv_path=runs_csv_path,
            save_csv=SAVE_CSV,
        )

        batch_results.append(
            {
                "image_index": image_index,
                "patch_x": patch_x,
                "patch_y": patch_y,
                "patch_size": patch_size,
                "max_depth": max_depth,
                "upper_bound": upper_bound,
                "timeout_schedule": timeout_milps,
                "adv_label": adv_label,
                "top_k": top_k,
                "split_selection_mode": split_selection_mode,
                "result": result,
            }
        )

        if is_resolved(result):
            print(
                f"Resolved image_index={image_index} patch=({patch_x}, {patch_y}) "
                f"size={patch_size} max_depth={max_depth} "
                f"upper_bound={upper_bound} adv_label={adv_label} "
                f"with timeout_schedule={timeout_milps} "
                f"top_k={top_k} split_selection_mode={split_selection_mode} "
                f"status={result['status']} total_runtime={result['total_runtime_seconds']:.2f}s"
            )
        else:
            print(
                f"image_index={image_index} patch=({patch_x}, {patch_y}) "
                f"size={patch_size} max_depth={max_depth} "
                f"upper_bound={upper_bound} adv_label={adv_label} "
                f"stayed unresolved with timeout_schedule={timeout_milps} "
                f"top_k={top_k} split_selection_mode={split_selection_mode}"
            )

    print("\n" + "=" * 80)
    print("Batch summary")
    print(f"CSV path: {runs_csv_path}")
    for item in batch_results:
        result = item["result"]
        print(
            f"image_index={item['image_index']} "
            f"patch=({item['patch_x']}, {item['patch_y']}) "
            f"size={item['patch_size']} max_depth={item['max_depth']} "
            f"upper_bound={item['upper_bound']} timeout_schedule={item['timeout_schedule']} "
            f"adv_label={item['adv_label']} "
            f"top_k={item['top_k']} split_selection_mode={item['split_selection_mode']} "
            f"status={result['status']} "
            f"total_runtime={result['total_runtime_seconds']:.2f}s "
            f"eran_runs={result['attempt_count']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
