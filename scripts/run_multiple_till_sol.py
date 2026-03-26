"""
Batch runner for recursive MNIST patch verification across upper bounds
and timeout retries.
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
from utils import resolve_runs_csv_path

DEFAULT_IMAGE_INDEX = 2
DEFAULT_PATCH_SIZE = 10
DEFAULT_PATCH_X = 0
DEFAULT_PATCH_Y = 0
DEFAULT_UPPER_BOUND = 0.65
DEFAULT_UPPER_BOUNDS = [0.65, 0.70, 0.75, 0.80]
DEFAULT_TIMEOUT_MILP = 27
DEFAULT_TIMEOUT_MILPS = [DEFAULT_TIMEOUT_MILP]
DEFAULT_MAX_DEPTH = 4
DEFAULT_TOP_K = 30
DEFAULT_ADD_BOOL_CONSTRAINTS = True
DEFAULT_USE_REFINE_POLY = False
SAVE_CSV = True
DEFAULT_CSV_PATH = None


def parse_float_sequence(values):
    parsed_values = []
    for value in values or []:
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                parsed_values.append(float(chunk))
    return parsed_values


def dedupe_preserving_order(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def resolve_upper_bounds(args):
    if args.upper_bounds:
        parsed_values = dedupe_preserving_order(parse_float_sequence(args.upper_bounds))
        if not parsed_values:
            raise ValueError("No valid upper bounds were provided.")
        return parsed_values
    if args.upper_bound is not None:
        return [float(args.upper_bound)]
    return list(DEFAULT_UPPER_BOUNDS)


def resolve_timeout_milps(args):
    if args.timeout_milps:
        parsed_values = dedupe_preserving_order(parse_float_sequence(args.timeout_milps))
        if not parsed_values:
            raise ValueError("No valid timeout values were provided.")
        return parsed_values
    if args.timeout_milp is not None:
        return [float(args.timeout_milp)]
    return list(DEFAULT_TIMEOUT_MILPS)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch recursive timeout refinement for MNIST.")
    parser.add_argument("--image-index", type=int, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--patch-x", type=int, default=DEFAULT_PATCH_X)
    parser.add_argument("--patch-y", type=int, default=DEFAULT_PATCH_Y)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=None,
        help="Run a single upper bound. Used only when --upper-bounds is not provided.",
    )
    parser.add_argument(
        "--upper-bounds",
        nargs="+",
        default=None,
        help=(
            "Upper bounds to iterate through. Accepts space-separated values, "
            "comma-separated values, or a mix of both. "
            f"Default: {' '.join(str(value) for value in DEFAULT_UPPER_BOUNDS)}"
        ),
    )
    parser.add_argument(
        "--timeout-milp",
        type=float,
        default=None,
        help="Run a single timeout. Used only when --timeout-milps is not provided.",
    )
    parser.add_argument(
        "--timeout-milps",
        nargs="+",
        default=None,
        help=(
            "Timeouts to try for each upper bound until the run resolves. "
            "Accepts space-separated values, comma-separated values, or a mix of both."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=(
            "Optional summary CSV path. If the file already exists, this run is "
            "appended to it; otherwise it is created."
        ),
    )
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
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    upper_bounds = resolve_upper_bounds(args)
    timeout_milps = resolve_timeout_milps(args)
    runs_csv_path = resolve_runs_csv_path(args.csv_path, batch_run_id)

    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    print("Batch recursive verification settings")
    print(f"image_index={args.image_index} patch=({args.patch_x}, {args.patch_y}) size={args.patch_size}")
    print(f"upper_bounds={upper_bounds}")
    print(f"timeout_milps={timeout_milps}")
    if SAVE_CSV:
        print(f"summary_csv={runs_csv_path}")

    batch_results = []

    for upper_bound in upper_bounds:
        resolved_for_upper_bound = False

        print("\n" + "#" * 80)
        print(f"Starting upper_bound={upper_bound}")
        print("#" * 80)

        for try_index, timeout_milp in enumerate(timeout_milps, start=1):
            print(
                f"\nRunning upper_bound={upper_bound} "
                f"timeout_milp={timeout_milp} try={try_index}/{len(timeout_milps)}"
            )

            result = run_verifier.verify_image_with_recursive_timeout_refinement(
                img_index=args.image_index,
                pixels=pixels,
                labels=labels,
                x_box=args.patch_x,
                y_box=args.patch_y,
                size_box=args.patch_size,
                timeout_milp=timeout_milp,
                max_depth=args.max_depth,
                top_k=args.top_k,
                ul=upper_bound,
                add_bool_constraints=args.add_bool_constraints,
                use_refine_poly=args.use_refine_poly,
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
