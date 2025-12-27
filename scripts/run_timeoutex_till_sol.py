"""
This script runs the example that got timeout till a solution is found to know our starting point.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from patch_input_box import *
from run_verifier import *

GRB_LICENSE_FILE="/root/ERAN/gurobi912/linux64/gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = GRB_LICENSE_FILE
print(os.environ.get("GRB_LICENSE_FILE"))

model_path = "/root/ERAN/tf_verify/models/mnist_convSmallRELU__PGDK.onnx"
test_data_path = "/root/ERAN/data/mnist_test.csv"

images_dir = "/root/Projects/Nathan/Patch-Attack-Verification/images"

# Load the CSV
df = pd.read_csv(test_data_path, header=None)

# Extract labels and image pixels
labels = df.iloc[:, 0].values
pixels = df.iloc[:, 1:].values



output_log_path = "/root/Projects/Nathan/Patch-Attack-Verification/timeout_example_max_time_img_2_patch_10_place_0_0.csv"

image_index = 2
patch_size = 10
patch_x_y = (0,0)

TIMEOUT_MILP = 259200  # Timeout in seconds for MILP = (72 hours - 3 days)

# Initialize results list
results = []

print(f"Starting verification for index {image_index} with patch size {patch_size} at position {patch_x_y} for timeout {TIMEOUT_MILP} seconds =  {(TIMEOUT_MILP/3600)/24} days")
print(f"Results will be saved to: {output_log_path}")
print("-" * 80)

label = labels[image_index]
print(f"  Testing image {image_index} (Label: {label})...", end=" ")

try:
	elapsed_time, last_status, failed_labels, example, is_adversarial = verify_image(
		img_index=image_index,
		pixels=pixels,
		labels=labels,
		x_box=patch_x_y[0],
		y_box=patch_x_y[1],
		size_box=patch_size,
		timeout_milp=TIMEOUT_MILP,
		with_plots=False
	)
	
	# Log the results
	result = {
		'image_index': image_index,
		'true_label': label,
		'patch_size': patch_size,
		'patch_x': patch_x_y[0],
		'patch_y': patch_x_y[1],
		'elapsed_time': elapsed_time,
		'status_code': last_status,
		'failed_labels': str(failed_labels) if failed_labels else '',
		'is_adversarial': is_adversarial,
		'has_counterexample': example is not None and len(example) > 0
	}
	results.append(result)
	
	# Save results to CSV after each verification
	results_df = pd.DataFrame(results)
	results_df.to_csv(output_log_path, index=False)
	
	status_str = "ADVERSARIAL" if is_adversarial else "VERIFIED"
	time_seconds = elapsed_time.total_seconds() if hasattr(elapsed_time, 'total_seconds') else elapsed_time
	print(f"{status_str} (Status: {last_status}, Time: {time_seconds:.2f}s)")
	
except Exception as e:
	print(f"ERROR: {str(e)}")
	result = {
		'image_index': image_index,
		'true_label': label,
		'patch_size': patch_size,
		'patch_x': patch_x_y[0],
		'patch_y': patch_x_y[1],
		'elapsed_time': None,
		'status_code': None,
		'failed_labels': '',
		'is_adversarial': None,
		'has_counterexample': False,
		'error': str(e)
	}
	results.append(result)
	
	# Save results to CSV after each verification (even on error)
	results_df = pd.DataFrame(results)
	results_df.to_csv(output_log_path, index=False)

print("\n" + "=" * 80)
print(f"Longest example verification complete!")
print(f"Results saved to: {output_log_path}")
print("=" * 80)
print(f"  Total adversarial examples found: {sum(1 for r in results if r.get('is_adversarial') == True)}")
print(f"  Total verified (no adversarial): {sum(1 for r in results if r.get('is_adversarial') == False)}")
print(f"  Total errors encountered: {sum(1 for r in results if 'error' in r)}")
