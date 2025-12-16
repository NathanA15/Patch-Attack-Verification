import subprocess
import re
from datetime import datetime
from pathlib import Path
import argparse

# =========================
# GLOBAL PARAMETERS
# =========================

ERAN_DIR = "/root/ERAN/tf_verify/"
PYTHON_BIN = "python3"
DATASET = "mnist"
NETNAME = "/root/ERAN/tf_verify/models/mnist_convSmallRELU__PGDK.onnx"
LOG_DIR = Path("/root/Projects/logs")

# =========================
# CORE FUNCTION
# =========================

def run_eran(input_box_path: str, domain: str, complete: bool = False, timeout_complete: int = 60) -> int:
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

	# print("hold =", hold)
	# print("log file =", log_file)

	return hold

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