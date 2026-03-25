"""
Recursive per-label timeout refinement for MNIST patch verification.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_verifier
from config import MNIST_DATA_PATH

DEFAULT_IMAGE_INDEX = 2
DEFAULT_PATCH_SIZE = 10
DEFAULT_PATCH_X = 0
DEFAULT_PATCH_Y = 0
DEFAULT_UPPER_BOUND = 0.65
DEFAULT_TIMEOUT_MILP = 27
DEFAULT_MAX_DEPTH = 2
DEFAULT_TOP_K = 30
DEFAULT_ADD_BOOL_CONSTRAINTS = True
DEFAULT_USE_REFINE_POLY = False
SAVE_CSV = True


def parse_args():
    parser = argparse.ArgumentParser(description="Recursive per-label timeout refinement for MNIST.")
    parser.add_argument("--image-index", type=int, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--patch-x", type=int, default=DEFAULT_PATCH_X)
    parser.add_argument("--patch-y", type=int, default=DEFAULT_PATCH_Y)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--upper-bound", type=float, default=DEFAULT_UPPER_BOUND)
    parser.add_argument("--timeout-milp", type=float, default=DEFAULT_TIMEOUT_MILP)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--add-bool-constraints",
        dest="add_bool_constraints",
        action="store_true",
        default=DEFAULT_ADD_BOOL_CONSTRAINTS,
    )
    parser.add_argument(
        "--no-add-bool-constraints",
        dest="add_bool_constraints",
        action="store_false",
    )
    parser.add_argument(
        "--use-refine-poly",
        dest="use_refine_poly",
        action="store_true",
        default=DEFAULT_USE_REFINE_POLY,
    )
    parser.add_argument(
        "--no-use-refine-poly",
        dest="use_refine_poly",
        action="store_false",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    run_verifier.verify_image_with_recursive_timeout_refinement(
        img_index=args.image_index,
        pixels=pixels,
        labels=labels,
        x_box=args.patch_x,
        y_box=args.patch_y,
        size_box=args.patch_size,
        timeout_milp=args.timeout_milp,
        max_depth=args.max_depth,
        top_k=args.top_k,
        ul=args.upper_bound,
        add_bool_constraints=args.add_bool_constraints,
        use_refine_poly=args.use_refine_poly,
        run_id=run_id,
        save_csv=SAVE_CSV,
    )


if __name__ == "__main__":
    main()
