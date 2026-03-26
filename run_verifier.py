import ast
import random
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from config import *
from patch_input_box import *
from utils import *


# =========================
# GLOBAL PARAMETERS
# =========================


# MILP status codes (Gurobi)
MILP_STATUS = {
	"LOADED": 1,  # Model is loaded, but no solution information is available.
	"OPTIMAL": 2,  # Model was solved to optimality (subject to tolerances), and an optimal solution is available.
	"INFEASIBLE": 3,  # Model was proven to be infeasible.
	"INF_OR_UNBD": 4,  # Model was proven to be either infeasible or unbounded.
	"UNBOUNDED": 5,  # Model was proven to be unbounded; says nothing about feasibility.
	"CUTOFF": 6,  # Optimal objective was proven worse than the Cutoff parameter value.
	"ITERATION_LIMIT": 7,  # Terminated: simplex/barrier iterations exceeded IterationLimit/BarIterLimit.
	"NODE_LIMIT": 8,  # Terminated: branch-and-cut nodes exceeded NodeLimit.
	"TIME_LIMIT": 9,  # Terminated: time expended exceeded TimeLimit.
	"SOLUTION_LIMIT": 10,  # Terminated: number of solutions found reached SolutionLimit.
	"INTERRUPTED": 11,  # Optimization was terminated by the user.
	"NUMERIC": 12,  # Terminated due to unrecoverable numerical difficulties.
	"SUBOPTIMAL": 13,  # Suboptimal solution available; optimality tolerances not satisfied.
	"INPROGRESS": 14,  # Asynchronous optimization run not yet complete.
	"USER_OBJ_LIMIT": 15,  # User-specified objective limit reached.
	"WORK_LIMIT": 16,  # Terminated: work expended exceeded WorkLimit.
	"MEM_LIMIT": 17,  # Terminated: memory allocated exceeded SoftMemLimit.
	"LOCALLY_OPTIMAL": 18,  # Solved to local optimality (preview feature).
	"LOCALLY_INFEASIBLE": 19,  # Appears locally infeasible (preview feature).
}

MILP_STATUS_NAME = {code: name for name, code in MILP_STATUS.items()}

_PYTORCH_MODEL = None


# =========================
# CORE FUNCTIONS
# =========================

def run_onnx(model_path, input_data):
    """Run ONNX inference on a single input batch."""

    # Create inference session.
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Get input name.
    input_name = session.get_inputs()[0].name

    # Ensure input is numpy float32.
    input_array = np.asarray(input_data, dtype=np.float32)

    # Run inference.
    return session.run(None, {input_name: input_array})


def parse_failed_labels(text):
    """Parse the failed labels list from ERAN log output."""

    match = re.search(r"failed labels:\s*(.*)", text)
    if not match:
        return []

    raw = match.group(1).strip()
    if raw in {"", "[]", "None"}:
        return []

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [int(x) for x in re.findall(r"-?\d+", raw)]

    if parsed is None:
        return []
    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, (list, tuple, set)):
        return [int(x) for x in parsed]
    return []


def parse_adversarial_examples(text):
    """Parse the adversarial example payload from ERAN log output."""

    pattern = r"possible adversarial examples is:\s*(\[[^\]]*\])"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


def parse_relaxed_example(text):
    """Parse the best relaxed fractional example from ERAN log output."""

    # Escape the parentheses and allow matches across multiple lines.
    pattern = r"Best RELAXED \(fractional\) values:\s*(\[.*?\])"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        # Safely convert the string representation of the list into a real list.
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


def parse_milp_statuses(text):
    """Extract all MILP status codes emitted in the log."""

    return [int(x) for x in re.findall(r"Adv_label:\s*\d+\s*,\s*Status:\s+(\d+)", text)]


def format_milp_status(status_code):
    """Map a numeric MILP status code to a human-readable label."""

    if status_code is None:
        return "UNKNOWN"
    return MILP_STATUS_NAME.get(status_code, f"UNKNOWN_{status_code}")


def parse_return_value(text):
    """Extract the final RETURN value from ERAN log output."""

    match = re.search(r"RETURN\s+(-?\d+)", text)
    return int(match.group(1)) if match else None


def parse_logged_status_code(text):
    """Extract a status code from the ERAN per-adversarial-label log format."""

    match = re.search(r"Adv_label:\s*\d+\s*,\s*Status:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def parse_label_results(text, failed_labels=None):
    """Parse per-label verification outcomes from a detailed ERAN log."""

    failed_set = set(failed_labels or [])
    counter_pattern = re.compile(
        r"Counter is\s+\d+(?:\s+candidate label is\s+(\d+)\s+adv label is\s+(\d+)|\s+label is\s+(\d+))"
    )
    counter_matches = list(counter_pattern.finditer(text))
    results = []

    for idx, match in enumerate(counter_matches):
        candidate_label = int(match.group(1)) if match.group(1) is not None else None
        adv_label = int(match.group(2) if match.group(2) is not None else match.group(3))
        start = match.start()
        end = counter_matches[idx + 1].start() if idx + 1 < len(counter_matches) else len(text)
        block = text[start:end]

        status_code = parse_logged_status_code(block)
        relaxed_example = parse_relaxed_example(block)
        model_verified = bool(
            re.search(r"MILP VERIFIED SUCCESSFULLY AGAINST LABEL\s+%d\b" % adv_label, block)
        )
        model_found_adv = bool(re.search(r"adv found against adv_label\s+%d\b" % adv_label, block))

        results.append(
            {
                "candidate_label": candidate_label,
                "adv_label": adv_label,
                "milp_status": status_code,
                "status_name": format_milp_status(status_code),
                "timed_out": status_code == MILP_STATUS["TIME_LIMIT"],
                "failed": adv_label in failed_set,
                "model_found_adv": model_found_adv,
                "verified": model_verified,
                "relaxed_example": relaxed_example,
            }
        )

    return results


def build_candidate_last_statuses(label_results):
    """Build a map from candidate label to its last recorded MILP status."""

    candidate_last_statuses = {}
    for result in label_results:
        candidate_label = result.get("candidate_label")
        if candidate_label is None:
            continue
        candidate_last_statuses[int(candidate_label)] = result.get("milp_status")
    return candidate_last_statuses


def build_adv_label_last_statuses(label_results):
    """Build a map from adversarial label to its last recorded MILP status."""

    adv_label_last_statuses = {}
    for result in label_results:
        adv_label = result.get("adv_label")
        if adv_label is None:
            continue
        adv_label_last_statuses[int(adv_label)] = result.get("milp_status")
    return adv_label_last_statuses


def parse_log_details(text):
    """Collect structured verification details from a raw ERAN log string."""

    failed_labels = parse_failed_labels(text)
    label_results = parse_label_results(text, failed_labels)
    statuses = parse_milp_statuses(text)

    return {
        "return_value": parse_return_value(text),
        "example": parse_adversarial_examples(text),
        "failed_labels": failed_labels,
        "label_results": label_results,
        "candidate_last_statuses": build_candidate_last_statuses(label_results),
        "adv_label_last_statuses": build_adv_label_last_statuses(label_results),
        "statuses": statuses,
        "last_status": statuses[-1] if statuses else None,
        "relaxed_example": parse_relaxed_example(text),
    }


def plot_adv(example):
    """Render a 28x28 adversarial example in grayscale."""

    adv_img = np.array(example).reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.imshow(adv_img, cmap="gray")
    plt.axis("off")


def verify_adv_example(example, label_img, failed_labels):
    """Verify an adversarial example using the ONNX model.

    Args:
        example (list or np.ndarray): The adversarial example to verify.
        label_img (int): The original label of the image.
        failed_labels (list): Labels that failed during verification.

    Returns:
        bool: True if it is a real adversarial example, False otherwise.
    """

    # Reshape to the expected 1x1x28x28 input tensor.
    ex = np.array(example, dtype=np.float32).reshape(1, 1, 28, 28)
    predicted_class = np.argmax(run_onnx(str(NETNAME), ex)[0])
    print("Predicted class for adversarial example:", predicted_class)
    print("Failed labels during verification using network:", failed_labels)
    print("Original label:", label_img)
    return label_img != predicted_class


def run_eran(
    input_box_path,
    domain,
    complete=False,
    timeout_complete=60,
    use_milp=False,
    label=-1,
    timeout_final_milp=30,
    adv_label=-1,
    add_bool_constraints=True,
    use_refine_poly=True,
    middle_bound=0.5,
    bounds=None,
):
    """Run ERAN and return structured verification details from the log.

    Args:
        input_box_path (str): Path to the input box file with pixel values in [0, 1].
        domain (str): Verification domain such as `deepzono`, `refinezono`,
            `deeppoly`, or `refinepoly`.
        complete (bool): Whether to enable the complete verification flow.
        timeout_complete (int): Timeout for complete verification.
        use_milp (bool): Whether MILP refinement is enabled.
        label (int): True label of the image being verified.
        timeout_final_milp (int): Timeout for the final MILP stage.
        adv_label (int): Specific adversarial label to target, or `-1` for all.
        add_bool_constraints (bool): Whether to add boolean constraints.
        use_refine_poly (bool): Whether to enable refinepoly-specific refinements.
        middle_bound (float): Split point used by the underlying bounds logic.
        bounds (list | None): Optional per-pixel bounds overrides.

    Raises:
        ValueError: If the log does not contain a RETURN value.

    Returns:
        dict: Structured details extracted from the ERAN run log.
    """

    cmd = [
        PYTHON_BIN,
        ".",
        "--dataset",
        DATASET,
        "--netname",
        NETNAME,
        "--domain",
        domain,
        "--input_box",
        input_box_path,
        "--logdir",
        str(LOG_DIR),
        "--complete",
        str(complete),
        "--timeout_complete",
        str(timeout_complete),
        "--use_milp",
        str(use_milp),
        "--timeout_milp",
        str(180),
        "--timeout_final_milp",
        str(timeout_final_milp),
        "--label",
        str(label),
        "--adv_label",
        str(adv_label),
        "--add_bool_constraints",
        str(add_bool_constraints),
        "--use_refine_poly",
        str(use_refine_poly),
        "--middle_bound",
        str(middle_bound),
        "--bounds",
        str(bounds if bounds is not None else []),
    ]

    start_time = datetime.now()
    print(f"\nLog file: {LOG_FILE}")
    with open(LOG_FILE, "w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=ERAN_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    elapsed_time = datetime.now() - start_time
    print(f"ERAN run completed in {elapsed_time}")

    log_text = LOG_FILE.read_text()
    details = parse_log_details(log_text)
    if details["return_value"] is None:
        raise ValueError("Could not find RETURN value in log output")

    details["elapsed_time"] = elapsed_time
    details["log_path"] = LOG_FILE
    print(
        "Last MILP status =",
        details["last_status"],
        f"({format_milp_status(details['last_status']) if details['last_status'] is not None else 'UNKNOWN'})",
    )
    if details["adv_label_last_statuses"]:
        print("Per-adversarial-label statuses:")
        for current_adv_label, status_code in sorted(details["adv_label_last_statuses"].items()):
            print(f"Adv label: {current_adv_label}, Status: {status_code} {format_milp_status(status_code)}")
    return details


def get_num_classes_for_image(img):
    """Infer the number of output classes for a single image."""

    sample = np.array(img, dtype=np.float32).reshape(1, 1, 28, 28)
    return int(run_onnx(str(NETNAME), sample)[0].shape[-1])


def build_candidate_adv_labels(label_img, adv_label, num_classes):
    """List adversarial labels to evaluate for the current image."""

    if adv_label != -1:
        return [adv_label]
    return [idx for idx in range(num_classes) if idx != int(label_img)]


def finalize_label_results(label_results, label_img, adv_label, num_classes, failed_labels, is_adversarial):
    """Normalize per-label results and assign final verification outcomes."""

    normalized_results = []
    for result in label_results:
        normalized = dict(result)
        if normalized.get("candidate_label") is None:
            normalized["candidate_label"] = int(label_img)
        normalized_results.append(normalized)

    results_by_label = {
        result["adv_label"]: dict(result)
        for result in normalized_results
    }
    candidate_labels = build_candidate_adv_labels(label_img, adv_label, num_classes)
    adversarial_set = set()
    if is_adversarial:
        if adv_label != -1:
            adversarial_set.add(int(adv_label))
        elif len(failed_labels) == 1:
            adversarial_set.add(int(failed_labels[0]))

    for candidate in candidate_labels:
        result = results_by_label.setdefault(
            candidate,
            {
                "candidate_label": int(label_img),
                "adv_label": candidate,
                "milp_status": None,
                "status_name": "PRECHECK_VERIFIED",
                "timed_out": False,
                "failed": False,
                "model_found_adv": False,
                "verified": True,
                "relaxed_example": None,
            },
        )

        if candidate in adversarial_set:
            result["final_outcome"] = "adversarial"
        elif result["timed_out"]:
            result["final_outcome"] = "timeout"
        elif result["verified"] or not result["failed"]:
            result["verified"] = True
            if result["status_name"] == "UNKNOWN":
                result["status_name"] = "PRECHECK_VERIFIED"
            result["final_outcome"] = "verified"
        else:
            result["final_outcome"] = "failed"

    return [results_by_label[label] for label in candidate_labels]


def verify_image(
    img_index,
    pixels,
    labels,
    x_box,
    y_box,
    size_box,
    timeout_milp=30,
    with_plots=False,
    ul=1.0,
    add_bool_constraints=True,
    use_refine_poly=True,
    middle_bound=0.5,
    bounds=None,
    adv_label=-1,
    return_details=False,
):
    """Verify one image against the configured patch attack region.

    This runs ERAN, optionally plots a returned adversarial example, and
    validates any candidate example with the ONNX model.

    Args:
        img_index (int): Index of the image to verify.
        pixels (np.ndarray): Array of image pixels.
        labels (np.ndarray): Array of image labels.
        timeout_milp (int, optional): Timeout for MILP verification.
        with_plots (bool, optional): Whether to plot the adversarial example.
        ul (float, optional): Upper limit for the patch pixel range.
        add_bool_constraints (bool, optional): Whether to add boolean constraints.
        use_refine_poly (bool, optional): Whether to use refinepoly refinement.
        middle_bound (float, optional): Split point used in bound refinement.
        bounds (list | None, optional): Optional per-pixel bounds overrides.
        adv_label (int, optional): Specific adversarial label to target.
        return_details (bool, optional): Whether to return the structured result.

    Returns:
        dict | tuple: Structured result when `return_details=True`, otherwise a
        tuple containing elapsed time, status, failed labels, example, and the
        adversarial flag.
    """

    img = pixels[img_index].reshape(28, 28)
    label_img = int(labels[img_index])

    # Patch size 11 finds an adversarial example for index 7.
    input_box_path = create_patch_input_config_file(img, x_box, y_box, size_box, label=label_img, ul=ul)
    run_details = run_eran(
        input_box_path=input_box_path,
        label=label_img,
        domain="refinepoly",
        complete=True,
        timeout_final_milp=timeout_milp,
        use_milp=True,
        adv_label=adv_label,
        add_bool_constraints=add_bool_constraints,
        use_refine_poly=use_refine_poly,
        middle_bound=middle_bound,
        bounds=bounds,
    )

    example = run_details["example"]
    failed_labels = run_details["failed_labels"]
    is_adversarial = False

    if example is None or len(example) == 0:
        print("No adversarial example returned")
    else:
        if with_plots:
            plot_adv(example)
        is_adversarial = verify_adv_example(example, label_img, failed_labels)

    num_classes = get_num_classes_for_image(img)
    final_label_results = finalize_label_results(
        run_details["label_results"],
        label_img,
        adv_label,
        num_classes,
        failed_labels,
        is_adversarial,
    )

    result = {
        "elapsed_time": run_details["elapsed_time"],
        "last_status": run_details["last_status"],
        "failed_labels": failed_labels,
        "example": example,
        "is_adversarial": is_adversarial,
        "log_path": run_details["log_path"],
        "return_value": run_details["return_value"],
        "adv_label": adv_label,
        "label_results": final_label_results,
        "candidate_last_statuses": run_details["candidate_last_statuses"],
        "adv_label_last_statuses": run_details["adv_label_last_statuses"],
        "relaxed_example": run_details["relaxed_example"],
    }

    if return_details:
        return result

    return (
        result["elapsed_time"],
        result["last_status"],
        result["failed_labels"],
        result["example"],
        result["is_adversarial"],
    )


def get_pytorch_model():
    """Load and cache the verification network as a PyTorch model."""

    global _PYTORCH_MODEL

    if _PYTORCH_MODEL is None:
        import onnx
        from onnx2pytorch import ConvertModel

        onnx_model = onnx.load(str(NETNAME))
        _PYTORCH_MODEL = ConvertModel(onnx_model)
        _PYTORCH_MODEL.eval()

    return _PYTORCH_MODEL


def get_pixel_influence(model_source, input_image, correct_class_idx, target_class_idx):
    """Compute absolute input gradients for the class margin."""

    import torch

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
    # Compute the margin between the target class and the correct class.
    # we want to maximize the target - meaning reinforcing the adversarial class, therefore the most important pixels will be the pixels whose gradients maximize the loss, therefore the pixels with biggest derivative.
    margin = outputs[0, target_class_idx] - outputs[0, correct_class_idx]
    margin.backward()
    return input_tensor.grad.detach()


def get_top_k_pixels_in_patch(
    influence_map,
    k=30,
    patch_x=0,
    patch_y=0,
    patch_size=10,
    img_w=28,
    img_h=28,
):
    """Select the most influential flattened pixel indices within the patch."""

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
    """Choose the patch pixels to split next using gradient magnitude."""

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
    """Return the per-label result for one adversarial label."""

    for label_result in run_result["label_results"]:
        if int(label_result["adv_label"]) == int(adv_label):
            return label_result
    raise ValueError(f"Did not find label result for adv_label={adv_label}")


def summarize_status(label_outcomes):
    """Summarize recursive timeout refinement outcomes into one status label."""

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
    """Rerun timed-out labels with progressively refined bounds."""

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


def verify_image_with_recursive_timeout_refinement(
    img_index,
    pixels,
    labels,
    x_box,
    y_box,
    size_box,
    timeout_milp=27,
    max_depth=2,
    top_k=30,
    ul=0.65,
    add_bool_constraints=True,
    use_refine_poly=False,
    middle_bound=0.5,
    with_plots=False,
    run_id=None,
    runs_csv_path=None,
    run_log_dir=None,
    save_csv=True,
):
    """Verify one image and recursively refine timed-out labels with gradient splits.

    Args:
        img_index (int): Index of the image to verify.
        pixels (np.ndarray): Array of image pixels.
        labels (np.ndarray): Array of image labels.
        x_box (int): Patch top-left x (column).
        y_box (int): Patch top-left y (row).
        size_box (int): Patch size.
        timeout_milp (int | float, optional): MILP timeout per ERAN call.
        max_depth (int, optional): Maximum number of recursive split reruns.
        top_k (int, optional): Number of top-gradient patch pixels to split each rerun.
        ul (float, optional): Upper bound for patch pixels.
        add_bool_constraints (bool, optional): Whether to add boolean constraints.
        use_refine_poly (bool, optional): Whether to enable refinepoly refinements.
        middle_bound (float, optional): Split point used by ERAN's bounds logic.
        with_plots (bool, optional): Whether to plot returned adversarial examples.
        run_id (str | None, optional): Run identifier used for logs and CSV naming.
        runs_csv_path (str | Path | None, optional): Optional summary CSV path override.
        run_log_dir (str | Path | None, optional): Optional per-run log directory override.
        save_csv (bool, optional): Whether to append the summary row to CSV.

    Returns:
        dict: Summary of the recursive timeout refinement run.
    """

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv_path = resolve_runs_csv_path(runs_csv_path, run_id)
    if run_log_dir is None:
        run_log_dir = build_run_log_dir(run_id)
    else:
        run_log_dir = Path(run_log_dir)
        run_log_dir.mkdir(parents=True, exist_ok=True)

    label_img = int(labels[img_index])
    base_bounds = generate_initial_bounds(int(pixels[img_index].size), 0, ul)

    print(
        f"Starting recursive verification for image {img_index} "
        f"label={label_img} patch=({x_box}, {y_box}) size={size_box}"
    )
    print(
        f"upper_bound={ul} timeout_milp={timeout_milp} "
        f"max_depth={max_depth} top_k={top_k}"
    )
    if save_csv:
        print(f"Summary CSV: {runs_csv_path}")
    print(f"Run log directory: {run_log_dir}")
    print("-" * 80)

    total_runtime = 0.0
    attempt_count = 0
    all_log_paths = []
    max_depth_reached = 0

    def run_for_label(adv_label, bounds):
        nonlocal total_runtime, attempt_count, all_log_paths
        global LOG_FILE

        log_file = fresh_log_file(run_log_dir, attempt_count, adv_label)
        previous_log_file = LOG_FILE
        LOG_FILE = log_file
        attempt_count += 1

        try:
            result = verify_image(
                img_index=img_index,
                pixels=pixels,
                labels=labels,
                x_box=x_box,
                y_box=y_box,
                size_box=size_box,
                timeout_milp=timeout_milp,
                with_plots=with_plots,
                ul=ul,
                add_bool_constraints=add_bool_constraints,
                use_refine_poly=use_refine_poly,
                middle_bound=middle_bound,
                bounds=bounds,
                adv_label=adv_label,
                return_details=True,
            )
        finally:
            LOG_FILE = previous_log_file

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
        max_depth=max_depth,
        choose_pixels_fn=lambda relaxed_example, adv_label: choose_split_pixels(
            relaxed_example,
            label_img,
            adv_label,
            x_box,
            y_box,
            size_box,
            top_k,
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
        "Image Index": img_index,
        "X patch": x_box,
        "Y patch": y_box,
        "Patch Size": size_box,
        "Upper bound": ul,
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
        "CSV path": "" if not save_csv else str(runs_csv_path),
    }
    append_run_row(runs_csv_path, row, save_csv=save_csv)

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

    return {
        "image_index": img_index,
        "label": label_img,
        "initial_timeout_labels": initial_timeout_labels,
        "finally_verified_labels": finally_verified_labels,
        "adversarial_labels": adversarial_labels,
        "unresolved_labels": unresolved_labels,
        "total_runtime_seconds": total_runtime,
        "max_depth_reached": max_depth_reached,
        "attempt_count": attempt_count,
        "status": overall_status,
        "comment": comment,
        "log_paths": all_log_paths,
        "csv_path": "" if not save_csv else str(runs_csv_path),
        "run_log_dir": str(run_log_dir),
        "label_outcomes": label_outcomes,
        "stopped_early": resolved["stopped_early"],
        "adversarial_label": resolved["adversarial_label"],
        "skipped_labels": skipped_labels,
        "initial_result": initial_result,
    }


def verify_image_with_sub_splits(
    img_index,
    pixels,
    labels,
    x_box,
    y_box,
    size_box,
    timeout_milp=30,
    split_pixels_count=4,
    is_random=False,
    split_pixels_list=None,
    split_value=0.5,
    split_amounts=1,
    ul=1.0,
    add_bool_constraints=True,
    use_refine_poly=True,
    middle_bound=0.5,
):
    """Verify one image by splitting selected patch pixels into sub-ranges.

    Args:
        img_index (int): Index of the image to verify.
        pixels (np.ndarray): Array of image pixels.
        labels (np.ndarray): Array of image labels.
        timeout_milp (int, optional): Timeout for MILP verification.
        split_pixels_count (int, optional): Number of pixels to split.
        is_random (bool, optional): Whether to randomly select split pixels.
        split_pixels_list (list, optional): Specific patch-local pixel indices.
        split_value (float, optional): Value used to split each selected pixel.
        split_amounts (int, optional): Number of sub-splits per selected pixel.
        ul (float, optional): Upper limit for the patch pixel range.
        add_bool_constraints (bool, optional): Whether to add boolean constraints.
        use_refine_poly (bool, optional): Whether to use refinepoly refinement.
        middle_bound (float, optional): Split point used in bound refinement.

    Returns:
        tuple: Elapsed time, last MILP status, failed labels, example, and
        whether the example is adversarial.
    """

    img = pixels[img_index].reshape(28, 28)
    label_img = labels[img_index]

    if is_random:
        # Select random pixels to split as indexes within the patch.
        split_pixels_list = random.sample(range(size_box * size_box), split_pixels_count)
    elif split_pixels_list is None:
        raise ValueError("split_pixels_list should not be defined if is_random is False")

    # Convert patch-local indexes to image coordinates.
    split_pixels_list = [
        convert_index_of_patch_pixel_to_coordinates(index, x_box, y_box, size_box)
        for index in split_pixels_list
    ]
    print(f"Splitting on pixels: {split_pixels_list}")

    # Range of values for each split pixel.
    split_pixels_ranges = [[[0, split_value], [split_value, 1]]] * split_pixels_count
    input_box_path = create_patch_input_config_file(
        img,
        x_box,
        y_box,
        size_box,
        label=label_img,
        split_pixels_list=split_pixels_list,
        split_pixel_range=split_pixels_ranges,
        split_amounts=split_amounts,
        ul=ul,
    )

    run_details = run_eran(
        input_box_path=input_box_path,
        label=label_img,
        domain="refinepoly",
        complete=True,
        timeout_final_milp=timeout_milp,
        use_milp=True,
        add_bool_constraints=add_bool_constraints,
        use_refine_poly=use_refine_poly,
        middle_bound=middle_bound,
    )

    example = run_details["example"]
    is_adversarial = False
    if example is None or len(example) == 0:
        print("No adversarial example returned")
    else:
        is_adversarial = verify_adv_example(example, label_img, run_details["failed_labels"])

    print(
        "Sub-verification result: "
        f"Elapsed time: {run_details['elapsed_time']}, "
        f"Last MILP status: {run_details['last_status']} "
        f"({format_milp_status(run_details['last_status']) if run_details['last_status'] is not None else 'UNKNOWN'}), "
        f"Is adversarial: {is_adversarial}"
    )

    return (
        run_details["elapsed_time"],
        run_details["last_status"],
        run_details["failed_labels"],
        example,
        is_adversarial,
    )
