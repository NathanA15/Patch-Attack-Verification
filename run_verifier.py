import subprocess
import re
from datetime import datetime
from pathlib import Path
import argparse
import ast
import onnxruntime as ort
import numpy as np

# =========================
# GLOBAL PARAMETERS
# =========================

ERAN_DIR = "/root/ERAN/tf_verify/"
PYTHON_BIN = "python3"
DATASET = "mnist"
NETNAME = "/root/ERAN/tf_verify/models/mnist_convSmallRELU__PGDK.onnx"
LOG_DIR = Path("/root/Projects/logs")

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
		return []

	try:
		return ast.literal_eval(match.group(1))
	except (ValueError, SyntaxError):
		return []
	
def parse_milp_status(text):
	match = re.search(r"MILP status is\s+(\d+)", text)
	if not match:
		return None
	return int(match.group(1))

def format_milp_status(status_code):
	if status_code is None:
		return "UNKNOWN"
	return MILP_STATUS_NAME.get(status_code, f"UNKNOWN_{status_code}")

def run_eran(input_box_path: str, domain: str, complete: bool = False, timeout_complete: int = 60, use_milp: bool = False, label: int = -1, timeout_final_milp: int = 30) -> int:
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
	LOG_DIR.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file = LOG_DIR / f"{timestamp}_eran_run.log"

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
		"--label", str(label)
	]

	with open(log_file, "w") as f:
		subprocess.run(
			cmd,
			cwd=ERAN_DIR,
			stdout=f,
			stderr=subprocess.STDOUT,
			text=True,
			check=True
		)

	log_text = log_file.read_text()

	match = re.search(r"RETURN\s+(-?\d+)", log_text)
	if not match:
		raise ValueError("Could not find RETURN value in log output")

	hold = int(match.group(1))

	example = parse_adversarial_examples(log_text)

	failed_labels = parse_failed_labels(log_text)

	status = parse_milp_status(log_text)

	print("MILP status =", status, f"({format_milp_status(status)})")
	# print("hold =", hold)
	# print("log file =", log_file)

	return hold, example, failed_labels

# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run ERAN and extract RETURN value")
	parser.add_argument("--input_box_path", required=True, help="Path to input box file")
	parser.add_argument("--domain", required=True, help="Verification domain (e.g. deeppoly)")

	args = parser.parse_args()

	run_eran(
		input_box_path=args.input_box_path,
		domain=args.domain
	)
