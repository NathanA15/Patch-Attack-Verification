import random
import subprocess
import re
from datetime import datetime
from pathlib import Path
import argparse
import ast
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from patch_input_box import *
from utils import *
from config import *

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

# =========================
# CORE FUNCTION
# =========================

def run_onnx(model_path, input_data):
	# Create inference session
	session = ort.InferenceSession(
		model_path,
		providers=["CPUExecutionProvider"]
	)

	# Get input name
	input_name = session.get_inputs()[0].name

	# Ensure input is numpy float32
	input_array = np.asarray(input_data, dtype=np.float32)

	# Run inference
	outputs = session.run(None, {input_name: input_array})

	return outputs


def parse_failed_labels(text):
	match = re.search(r"failed labels:\s*(.*)", text)
	if not match:
		return ""
	return match.group(1).strip()

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
    # We must escape the parentheses '(' and ')' with backslashes
    # We use re.DOTALL so the dot (.) matches newlines, in case the list is long
    pattern = r"Best RELAXED \(fractional\) values:\s*(\[.*?\])"
    
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    try:
        # safely convert the string representation of the list into a real list
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


# def parse_milp_status(text):
# 	match = re.search(r"MILP status is\s+(\d+)", text)
# 	if not match:
# 		return None
# 	return int(match.group(1))

def parse_milp_statuses(text):
	return [int(x) for x in re.findall(r"MILP status is\s+(\d+)", text)]

def format_milp_status(status_code):
	if status_code is None:
		return "UNKNOWN"
	return MILP_STATUS_NAME.get(status_code, f"UNKNOWN_{status_code}")


def plot_adv(example):
    adv_img = np.array(example).reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.imshow(adv_img, cmap="gray")
    plt.axis("off")
    

def verify_adv_example(example, label_img, failed_labels):
	"""Verify adversarial example using ONNX model

	Args:
		example (list or np.ndarray): The adversarial example to verify.
		label_img (int): The original label of the image.
		failed_labels (list): List of labels that failed during verification.

	Returns:
		bool: True if its a real adversarial example, False otherwise
	"""

	ex = np.array(example, dtype=np.float32).reshape(1, 1, 28, 28) # dimension 1x1x28x28
	predicted_class = np.argmax(run_onnx(str(NETNAME), ex)[0])
	print("Predicted class for adversarial example:", predicted_class)
	print("Failed labels during verification using network:", failed_labels)
	print("Original label:", label_img)

	return label_img != predicted_class



def run_eran(input_box_path: str, domain: str, complete: bool = False, timeout_complete: int = 60, use_milp: bool = False, 
			 label: int = -1, timeout_final_milp: int = 30, adv_label: int = -1, add_bool_constraints=True, use_refine_poly=True,
			 middle_bound=0.5,
			 bounds: list=[],
			 ) -> int:
	""" 
	Run eran and extract dominant class or -1 if hasn't succeeded
	Dominant class is the class given to all permutated images in the range.
	
	Args:
		input_box_path (str): path to input_box with values between [0,1] for each pixels
		domain (str): 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly'
		complete (bool): if True, activate MILP Verifier if it failed proving robustness with current domain
		timeout_complete (int): timeout for MILP Verifer

	Raises:
		ValueError: ...

	Returns:
		int: dominant class or -1 if failed
	"""



	cmd = [
		PYTHON_BIN, ".",
		"--dataset", DATASET,
		"--netname", NETNAME,
		"--domain", domain,
		"--input_box", input_box_path,
		"--logdir", str(LOG_DIR),
		"--complete", str(complete),
		"--timeout_complete", str(timeout_complete),
		"--use_milp", str(use_milp),
		"--timeout_milp", str(180),
		"--timeout_final_milp", str(timeout_final_milp), # this is the right one to set
		"--label", str(label),
		"--adv_label", str(adv_label),
		"--add_bool_constraints", str(add_bool_constraints),
		"--use_refine_poly", str(use_refine_poly),
		"--middle_bound", str(middle_bound),
		"--bounds", str(bounds)
	]
	start_time = datetime.now()
	print(f"\nLog file: {LOG_FILE}")
	with open(LOG_FILE, "w") as f:
		subprocess.run(
			cmd,
			cwd=ERAN_DIR,
			stdout=f,
			stderr=subprocess.STDOUT,
			text=True,
			check=True
		)
	
	end_time = datetime.now()
	elapsed_time = end_time - start_time
	print(f"ERAN run completed in {elapsed_time}")

	log_text = LOG_FILE.read_text()

	match = re.search(r"RETURN\s+(-?\d+)", log_text)
	if not match:
		raise ValueError("Could not find RETURN value in log output")

	hold = int(match.group(1))

	example = parse_adversarial_examples(log_text)

	failed_labels = parse_failed_labels(log_text)

	statuses = parse_milp_statuses(log_text)

	relaxed_example = parse_relaxed_example(log_text)

	last_status = statuses[-1] if statuses else None # last one because break_on_failure is true

	print("Last MILP status =", last_status, f"({format_milp_status(last_status) if last_status is not None else 'UNKNOWN'})")

	return failed_labels, elapsed_time, last_status, example


def verify_image(img_index, pixels, labels, x_box, y_box, size_box, timeout_milp=30, with_plots=False, ul=1.0, add_bool_constraints=True, use_refine_poly=True, middle_bound=0.5, bounds=[]):
	"""
	This function verifies a specific image from the dataset using ERAN with patch attack verification.
	And plots the adversarial example if found.
	Including prints of relevant information.

	Args:
		img_index (int): index of image to verify
		pixels (np.ndarray): array of pixel values
		labels (np.ndarray): array of labels
		timeout_milp (int, optional): timeout for MILP verification. Defaults to 30.
	"""

	img = pixels[img_index].reshape(28, 28)
	label_img = labels[img_index]

	# patch size 11 finds adversarial example for index 7
	input_box_path = create_patch_input_config_file(img, x_box, y_box, size_box, label=label_img, ul=ul)
	failed_labels, elapsed_time, last_status, example = run_eran(input_box_path=input_box_path, label=label_img, domain="refinepoly", complete=True, 
															  timeout_final_milp=timeout_milp, use_milp=True, add_bool_constraints=add_bool_constraints, use_refine_poly=use_refine_poly,
															  middle_bound=middle_bound,
															  bounds=bounds) #, adv_labels=[2]
	
	is_adversarial = False

	if example is None or len(example) == 0:
		print("No adversarial example returned")
	else:
		if with_plots:
			plot_adv(example)

		is_adversarial = verify_adv_example(example, label_img, failed_labels)

	return elapsed_time, last_status, failed_labels, example, is_adversarial 



def verify_image_with_sub_splits(img_index, pixels, labels, x_box, y_box, size_box, timeout_milp=30, split_pixels_count=4, is_random=False, split_pixels_list=None, 
								 split_value=0.5, split_amounts=1, ul=1.0, add_bool_constraints=True, use_refine_poly=True, middle_bound=0.5):
	"""
	This function verifies a specific image from the dataset using ERAN with patch attack verification.
	It splits the verification into multiple sub-verifications by splitting pixel ranges.
	Including prints of relevant information.

	Args:
		img_index (int): index of image to verify
		pixels (np.ndarray): array of pixel values
		labels (np.ndarray): array of labels
		timeout_milp (int, optional): timeout for MILP verification. Defaults to 30.
		split_pixels_count (int, optional): number of pixels to split. Defaults to 4.
		is_random (bool, optional): whether to randomly select pixels to split. Defaults to False.
		split_pixels_list (list, optional): list of pixel indices to split. Defaults to None.
		split_value (float, optional): value to split pixels at. Defaults to 0.5.

	Returns:
		dict: results of each sub-verification
	"""

	img = pixels[img_index].reshape(28, 28)
	label_img = labels[img_index]

	results = {}

	if is_random:
		# select random pixels to split as indexes within the patch
		split_pixels_list = random.sample(range(size_box * size_box), split_pixels_count)

	elif not is_random and split_pixels_list is None:
		raise ValueError("split_pixels_list should not be defined if is_random is False")

	# convert to coordinates in the image
	split_pixels_list = [convert_index_of_patch_pixel_to_coordinates(index, x_box, y_box, size_box) for index in split_pixels_list]

	print(f"Splitting on pixels: {split_pixels_list}")

	# range of values for each split pixel.
	split_pixels_ranges = [[[0,split_value], [split_value,1]]] * split_pixels_count

	input_box_path = create_patch_input_config_file(img, x_box, y_box, size_box, label=label_img, split_pixels_list=split_pixels_list, split_pixel_range=split_pixels_ranges, split_amounts=split_amounts, ul=ul)

	failed_labels, elapsed_time, last_status, example = run_eran(input_box_path=input_box_path, label=label_img, domain="refinepoly", complete=True, 
															  timeout_final_milp=timeout_milp, use_milp=True, add_bool_constraints=add_bool_constraints, use_refine_poly=use_refine_poly, middle_bound=middle_bound) #, adv_labels=[2]

	is_adversarial = False

	if example is None or len(example) == 0:
		print("No adversarial example returned")
	else:

		is_adversarial = verify_adv_example(example, label_img, failed_labels)
	
	print(f"Sub-verification result: Elapsed time: {elapsed_time}, Last MILP status: {last_status} ({format_milp_status(last_status) if last_status is not None else 'UNKNOWN'}), Is adversarial: {is_adversarial}")

	return elapsed_time, last_status, failed_labels, example, is_adversarial 


# # =========================
# # MAIN ENTRY POINT
# # =========================

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description="Run ERAN and extract RETURN value")
# 	parser.add_argument("--input_box_path", required=True, help="Path to input box file")
# 	parser.add_argument("--domain", required=True, help="Verification domain (e.g. deeppoly)")

# 	args = parser.parse_args()

# 	run_eran(
# 		input_box_path=args.input_box_path,
# 		domain=args.domain
# 	)





