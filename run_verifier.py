import ast
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from config import *
from patch_input_box import *
from utils import *


MILP_STATUS = {
    "LOADED": 1,
    "OPTIMAL": 2,
    "INFEASIBLE": 3,
    "INF_OR_UNBD": 4,
    "UNBOUNDED": 5,
    "CUTOFF": 6,
    "ITERATION_LIMIT": 7,
    "NODE_LIMIT": 8,
    "TIME_LIMIT": 9,
    "SOLUTION_LIMIT": 10,
    "INTERRUPTED": 11,
    "NUMERIC": 12,
    "SUBOPTIMAL": 13,
    "INPROGRESS": 14,
    "USER_OBJ_LIMIT": 15,
    "WORK_LIMIT": 16,
    "MEM_LIMIT": 17,
    "LOCALLY_OPTIMAL": 18,
    "LOCALLY_INFEASIBLE": 19,
}
MILP_STATUS_NAME = {code: name for name, code in MILP_STATUS.items()}


def run_onnx(model_path, input_data):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_array = np.asarray(input_data, dtype=np.float32)
    return session.run(None, {input_name: input_array})


def parse_failed_labels(text):
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
    pattern = r"possible adversarial examples is:\s*(\[[^\]]*\])"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


def parse_relaxed_example(text):
    pattern = r"Best RELAXED \(fractional\) values:\s*(\[.*?\])"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


def parse_milp_statuses(text):
    return [int(x) for x in re.findall(r"MILP status is\s+(\d+)", text)]


def format_milp_status(status_code):
    if status_code is None:
        return "UNKNOWN"
    return MILP_STATUS_NAME.get(status_code, f"UNKNOWN_{status_code}")


def parse_return_value(text):
    match = re.search(r"RETURN\s+(-?\d+)", text)
    return int(match.group(1)) if match else None


def parse_label_results(text, failed_labels=None):
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

        status_match = re.search(r"MILP status is\s+(\d+)", block)
        status_code = int(status_match.group(1)) if status_match else None
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
    candidate_last_statuses = {}
    for result in label_results:
        candidate_label = result.get("candidate_label")
        if candidate_label is None:
            continue
        candidate_last_statuses[int(candidate_label)] = result.get("milp_status")
    return candidate_last_statuses


def parse_log_details(text):
    failed_labels = parse_failed_labels(text)
    label_results = parse_label_results(text, failed_labels)
    statuses = parse_milp_statuses(text)

    return {
        "return_value": parse_return_value(text),
        "example": parse_adversarial_examples(text),
        "failed_labels": failed_labels,
        "label_results": label_results,
        "candidate_last_statuses": build_candidate_last_statuses(label_results),
        "statuses": statuses,
        "last_status": statuses[-1] if statuses else None,
        "relaxed_example": parse_relaxed_example(text),
    }


def plot_adv(example):
    adv_img = np.array(example).reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.imshow(adv_img, cmap="gray")
    plt.axis("off")


def verify_adv_example(example, label_img, failed_labels):
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
    if details["candidate_last_statuses"]:
        formatted_statuses = {
            candidate_label: format_milp_status(status_code)
            for candidate_label, status_code in details["candidate_last_statuses"].items()
        }
        print("Per-candidate last MILP statuses =", formatted_statuses)
    return details


def get_num_classes_for_image(img):
    sample = np.array(img, dtype=np.float32).reshape(1, 1, 28, 28)
    return int(run_onnx(str(NETNAME), sample)[0].shape[-1])


def build_candidate_adv_labels(label_img, adv_label, num_classes):
    if adv_label != -1:
        return [adv_label]
    return [idx for idx in range(num_classes) if idx != int(label_img)]


def finalize_label_results(label_results, label_img, adv_label, num_classes, failed_labels, is_adversarial):
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
    img = pixels[img_index].reshape(28, 28)
    label_img = int(labels[img_index])
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
    img = pixels[img_index].reshape(28, 28)
    label_img = labels[img_index]

    if is_random:
        split_pixels_list = random.sample(range(size_box * size_box), split_pixels_count)
    elif split_pixels_list is None:
        raise ValueError("split_pixels_list should not be defined if is_random is False")

    split_pixels_list = [
        convert_index_of_patch_pixel_to_coordinates(index, x_box, y_box, size_box)
        for index in split_pixels_list
    ]
    print(f"Splitting on pixels: {split_pixels_list}")

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
