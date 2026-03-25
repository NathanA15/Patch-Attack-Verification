"""
Recursive per-label timeout refinement for MNIST patch verification.
"""

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import pandas as pd
import torch
from onnx2pytorch import ConvertModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_verifier
from config import CSV_DIR, LOG_DIR, MNIST_DATA_PATH, NETNAME
from utils import generate_initial_bounds, subdivide_bounds_at_indices

N_PIXELS = 28 * 28
DEFAULT_IMAGE_INDEX = 2
DEFAULT_PATCH_SIZE = 10
DEFAULT_PATCH_X = 0
DEFAULT_PATCH_Y = 0
DEFAULT_UPPER_BOUND = 0.7
DEFAULT_TIMEOUT_MILP = 100
DEFAULT_MAX_DEPTH = 10
DEFAULT_TOP_K = 30
DEFAULT_ADD_BOOL_CONSTRAINTS = True
DEFAULT_USE_REFINE_POLY = False
SAVE_CSV = True

CSV_COLUMNS = [
    "Image Index",
    "X patch",
    "Y patch",
    "Patch Size",
    "Upper bound",
    "Initial Timeout Labels",
    "Finally Verified Labels",
    "Adversarial Labels",
    "Unresolved Labels",
    "Total Run Time (s)",
    "Max Depth Reached",
    "Attempt Count",
    "Status",
    "Comment",
    "Log paths",
    "CSV path",
]

_PYTORCH_MODEL = None


def build_runs_csv_path(run_id=None):
    timestamp = run_id or time.strftime("%Y_%m_%d_%H_%M_%S")
    return Path(CSV_DIR) / f"recursive_timeout_refinement_{timestamp}.csv"


def build_run_log_dir(run_id=None):
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_folder = LOG_DIR / datetime.now().strftime("%Y%m%d")
    run_log_dir = daily_folder / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return run_log_dir


def fresh_log_file(run_log_dir, attempt_idx, adv_label):
    label_tag = "all_labels" if adv_label == -1 else f"adv_label_{adv_label}"
    return run_log_dir / f"{attempt_idx:03d}_{label_tag}_eran_run.log"


def ensure_trailing_newline(path):
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


def append_run_row(csv_path, row):
    if not SAVE_CSV:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    if file_exists:
        ensure_trailing_newline(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([row.get(col, "") for col in CSV_COLUMNS])


def elapsed_seconds(elapsed_time):
    if hasattr(elapsed_time, "total_seconds"):
        return float(elapsed_time.total_seconds())
    return float(elapsed_time)


def get_pytorch_model():
    global _PYTORCH_MODEL
    if _PYTORCH_MODEL is None:
        onnx_model = onnx.load(str(NETNAME))
        _PYTORCH_MODEL = ConvertModel(onnx_model)
        _PYTORCH_MODEL.eval()
    return _PYTORCH_MODEL


def get_pixel_influence(model_source, input_image, correct_class_idx, target_class_idx):
    if not torch.is_tensor(input_image):
        input_tensor = torch.from_numpy(np.asarray(input_image)).float()
    else:
        input_tensor = input_image.clone().float()

    if input_tensor.numel() == 784:
        input_tensor = input_tensor.view(1, 1, 28, 28)
    elif tuple(input_tensor.shape) == (28, 28):
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif tuple(input_tensor.shape) == (1, 28, 28):
        input_tensor = input_tensor.unsqueeze(0)

    if tuple(input_tensor.shape) != (1, 1, 28, 28):
        raise ValueError(f"Input shape mismatch: {tuple(input_tensor.shape)}")

    input_tensor.requires_grad_(True)
    pytorch_model = model_source
    pytorch_model.zero_grad()
    outputs = pytorch_model(input_tensor)
    margin = outputs[0, correct_class_idx] - outputs[0, target_class_idx]
    margin.backward()
    return input_tensor.grad.data.abs()


def get_top_k_pixels_in_patch(
    influence_map,
    k=30,
    patch_x=0,
    patch_y=0,
    patch_size=10,
    img_w=28,
    img_h=28,
):
    grad_2d = influence_map.view(img_h, img_w)
    candidates = []

    y_start = max(0, patch_y)
    y_end = min(img_h, patch_y + patch_size)
    x_start = max(0, patch_x)
    x_end = min(img_w, patch_x + patch_size)

    for row in range(y_start, y_end):
        for col in range(x_start, x_end):
            candidates.append((grad_2d[row, col].item(), row * img_w + col))

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = candidates[:k]
    return [item[1] for item in top_k]


def choose_split_pixels(relaxed_example, label_img, adv_label, patch_x, patch_y, patch_size, top_k):
    if relaxed_example is None:
        raise ValueError(f"Cannot refine label {adv_label}: relaxed example missing from log")

    grads = get_pixel_influence(
        model_source=get_pytorch_model(),
        input_image=np.array(relaxed_example, dtype=np.float32).reshape(1, 1, 28, 28),
        correct_class_idx=label_img,
        target_class_idx=adv_label,
    )
    return get_top_k_pixels_in_patch(
        grads,
        k=top_k,
        patch_x=patch_x,
        patch_y=patch_y,
        patch_size=patch_size,
    )


def get_single_label_result(run_result, adv_label):
    for label_result in run_result["label_results"]:
        if int(label_result["adv_label"]) == int(adv_label):
            return label_result
    raise ValueError(f"Did not find label result for adv_label={adv_label}")


def summarize_status(label_outcomes):
    adversarial_labels = sorted(
        label for label, outcome in label_outcomes.items() if outcome["final_outcome"] == "adversarial"
    )
    unresolved_labels = sorted(
        label
        for label, outcome in label_outcomes.items()
        if outcome["final_outcome"] in {"max_depth_timeout", "error", "failed"}
    )
    if adversarial_labels:
        return "Adversarial"
    if unresolved_labels:
        return "MaxDepthTimeout"
    return "Verified"


def resolve_timed_out_labels(initial_results, base_bounds, max_depth, choose_pixels_fn, rerun_label_fn):
    label_outcomes = {}
    rerun_attempts = 0
    rerun_runtime = 0.0
    max_depth_reached = 0
    log_paths = []

    for idx, initial_result in enumerate(initial_results):
        adv_label = int(initial_result["adv_label"])
        if not initial_result["timed_out"]:
            label_outcomes[adv_label] = {
                "final_outcome": initial_result["final_outcome"],
                "depth": 0,
                "runtime_seconds": 0.0,
                "log_paths": [],
                "last_status": initial_result["milp_status"],
            }
            if initial_result["final_outcome"] == "adversarial":
                skipped_labels = [
                    int(result["adv_label"])
                    for result in initial_results[idx + 1 :]
                    if result["timed_out"]
                ]
                return {
                    "label_outcomes": label_outcomes,
                    "rerun_attempts": rerun_attempts,
                    "rerun_runtime": rerun_runtime,
                    "max_depth_reached": max_depth_reached,
                    "log_paths": log_paths,
                    "stopped_early": True,
                    "adversarial_label": adv_label,
                    "skipped_labels": skipped_labels,
                }
            continue

        state_bounds = deepcopy(base_bounds)
        current_result = dict(initial_result)
        label_runtime = 0.0
        label_logs = []
        depth = 0

        while current_result["timed_out"] and depth < max_depth:
            split_indices = choose_pixels_fn(current_result["relaxed_example"], adv_label)
            state_bounds = subdivide_bounds_at_indices(state_bounds, split_indices)
            depth += 1
            rerun_attempts += 1

            rerun_result = rerun_label_fn(adv_label, state_bounds)
            rerun_seconds = elapsed_seconds(rerun_result["elapsed_time"])
            rerun_runtime += rerun_seconds
            label_runtime += rerun_seconds
            label_logs.append(str(rerun_result["log_path"]))
            log_paths.append(str(rerun_result["log_path"]))

            current_result = get_single_label_result(rerun_result, adv_label)
            max_depth_reached = max(max_depth_reached, depth)

            if current_result["final_outcome"] == "adversarial":
                break
            if not current_result["timed_out"]:
                break

        final_outcome = current_result["final_outcome"]
        if current_result["timed_out"] and depth >= max_depth:
            final_outcome = "max_depth_timeout"

        label_outcomes[adv_label] = {
            "final_outcome": final_outcome,
            "depth": depth,
            "runtime_seconds": label_runtime,
            "log_paths": label_logs,
            "last_status": current_result["milp_status"],
        }

        if final_outcome == "adversarial":
            skipped_labels = [
                int(result["adv_label"])
                for result in initial_results[idx + 1 :]
                if result["timed_out"]
            ]
            return {
                "label_outcomes": label_outcomes,
                "rerun_attempts": rerun_attempts,
                "rerun_runtime": rerun_runtime,
                "max_depth_reached": max_depth_reached,
                "log_paths": log_paths,
                "stopped_early": True,
                "adversarial_label": adv_label,
                "skipped_labels": skipped_labels,
            }

    return {
        "label_outcomes": label_outcomes,
        "rerun_attempts": rerun_attempts,
        "rerun_runtime": rerun_runtime,
        "max_depth_reached": max_depth_reached,
        "log_paths": log_paths,
        "stopped_early": False,
        "adversarial_label": None,
        "skipped_labels": [],
    }


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
    runs_csv_path = build_runs_csv_path(run_id)
    run_log_dir = build_run_log_dir(run_id)

    df = pd.read_csv(MNIST_DATA_PATH, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    label_img = int(labels[args.image_index])
    base_bounds = generate_initial_bounds(N_PIXELS, 0, args.upper_bound)

    print(
        f"Starting recursive verification for image {args.image_index} "
        f"label={label_img} patch=({args.patch_x}, {args.patch_y}) size={args.patch_size}"
    )
    print(
        f"upper_bound={args.upper_bound} timeout_milp={args.timeout_milp} "
        f"max_depth={args.max_depth} top_k={args.top_k}"
    )
    if SAVE_CSV:
        print(f"Summary CSV: {runs_csv_path}")
    print(f"Run log directory: {run_log_dir}")
    print("-" * 80)

    total_runtime = 0.0
    attempt_count = 0
    all_log_paths = []
    max_depth_reached = 0

    def run_for_label(adv_label, bounds):
        nonlocal total_runtime, attempt_count, all_log_paths
        log_file = fresh_log_file(run_log_dir, attempt_count, adv_label)
        run_verifier.LOG_FILE = log_file
        attempt_count += 1
        result = run_verifier.verify_image(
            img_index=args.image_index,
            pixels=pixels,
            labels=labels,
            x_box=args.patch_x,
            y_box=args.patch_y,
            size_box=args.patch_size,
            timeout_milp=args.timeout_milp,
            with_plots=False,
            ul=args.upper_bound,
            add_bool_constraints=args.add_bool_constraints,
            use_refine_poly=args.use_refine_poly,
            bounds=bounds,
            adv_label=adv_label,
            return_details=True,
        )
        run_seconds = elapsed_seconds(result["elapsed_time"])
        total_runtime += run_seconds
        all_log_paths.append(str(result["log_path"]))
        return result

    initial_result = run_for_label(-1, base_bounds)
    initial_label_results = initial_result["label_results"]
    initial_timeout_labels = sorted(
        int(label_result["adv_label"])
        for label_result in initial_label_results
        if label_result["timed_out"]
    )

    print(f"Initial timeout labels: {initial_timeout_labels}")

    resolved = resolve_timed_out_labels(
        initial_results=initial_label_results,
        base_bounds=base_bounds,
        max_depth=args.max_depth,
        choose_pixels_fn=lambda relaxed_example, adv_label: choose_split_pixels(
            relaxed_example,
            label_img,
            adv_label,
            args.patch_x,
            args.patch_y,
            args.patch_size,
            args.top_k,
        ),
        rerun_label_fn=run_for_label,
    )

    max_depth_reached = max(max_depth_reached, resolved["max_depth_reached"])
    all_log_paths.extend(resolved["log_paths"])
    label_outcomes = resolved["label_outcomes"]
    skipped_labels = resolved["skipped_labels"]

    finally_verified_labels = sorted(
        label for label, outcome in label_outcomes.items() if outcome["final_outcome"] == "verified"
    )
    adversarial_labels = sorted(
        label for label, outcome in label_outcomes.items() if outcome["final_outcome"] == "adversarial"
    )
    unresolved_labels = sorted(
        label
        for label, outcome in label_outcomes.items()
        if outcome["final_outcome"] in {"max_depth_timeout", "error", "failed"}
    )

    overall_status = summarize_status(label_outcomes)
    comment = (
        f"Recursive timeout refinement finished. "
        f"timed_out_initial={initial_timeout_labels} "
        f"rerun_attempts={resolved['rerun_attempts']} "
        f"stopped_early={resolved['stopped_early']} "
        f"adversarial_label={resolved['adversarial_label']} "
        f"skipped_labels={skipped_labels}"
    )

    row = {
        "Image Index": args.image_index,
        "X patch": args.patch_x,
        "Y patch": args.patch_y,
        "Patch Size": args.patch_size,
        "Upper bound": args.upper_bound,
        "Initial Timeout Labels": str(initial_timeout_labels),
        "Finally Verified Labels": str(finally_verified_labels),
        "Adversarial Labels": str(adversarial_labels),
        "Unresolved Labels": str(unresolved_labels),
        "Total Run Time (s)": int(round(total_runtime)),
        "Max Depth Reached": max_depth_reached,
        "Attempt Count": attempt_count,
        "Status": overall_status,
        "Comment": comment,
        "Log paths": ";".join(all_log_paths),
        "CSV path": "" if not SAVE_CSV else str(runs_csv_path),
    }
    append_run_row(runs_csv_path, row)

    print("\n" + "=" * 80)
    print(f"Status: {overall_status}")
    print(f"Initial timeout labels: {initial_timeout_labels}")
    print(f"Finally verified labels: {finally_verified_labels}")
    print(f"Adversarial labels: {adversarial_labels}")
    print(f"Unresolved labels: {unresolved_labels}")
    if resolved["stopped_early"]:
        print(
            f"Stopped early after proving not robust against label {resolved['adversarial_label']} "
            f"(skipped timed-out labels: {skipped_labels})"
        )
    print(f"Total runtime: {total_runtime:.2f}s across {attempt_count} ERAN runs")
    print("=" * 80)


if __name__ == "__main__":
    main()
