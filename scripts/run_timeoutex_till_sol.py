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
from config import *



# Load the CSV
df = pd.read_csv(MNIST_DATA_PATH, header=None)

# Extract labels and image pixels
labels = df.iloc[:, 0].values
pixels = df.iloc[:, 1:].values


# add timestamp to output log path
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

image_index = 2
patch_size = 10 # need to do 10
patch_x_y = (0,0)

# tests ---------------------------

# with 10 it verified in 23.83s without the bool constraints 
# with bool constraint at 0.3 did 
# upper bound at 0.6

# test upper bound at 0.7
# no bool constraints - 477.32s
# bool constraints 


# endtests ---------------------------


out_csv_path = CSV_DIR / f"timeout_example_with_bool_constraints_max_time_img_{image_index}_patch_{patch_size}_place_{patch_x_y[0]}_{patch_x_y[1]}_{timestamp}.csv"
# out_csv_path = CSV_DIR / f"timeout_trash_max_time_img_{image_index}_patch_{patch_size}_place_{patch_x_y[0]}_{patch_x_y[1]}_{timestamp}.csv"

# TIMEOUT_MILP = 604800  # Timeout in seconds for MILP = 7 days
TIMEOUT_MILP = 60*60*3 # 3 hours for testing
upper_bound = 0.7
# Initialize results list
results = []
print(f"This is to test adding conditions and split the input box, and all sizes to either be lower bound to mid or mid to upper bound")
print(f"Starting verification for index {image_index} with patch size {patch_size} at position {patch_x_y} for timeout {TIMEOUT_MILP} seconds =  {(TIMEOUT_MILP/3600)/24} days")
print(f"example with range of 0->0.05 for patch pixels")
print(f"Results will be saved to: {out_csv_path}")
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
		with_plots=False,
		ul=upper_bound,
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
	results_df.to_csv(out_csv_path, index=False)
	
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
	results_df.to_csv(out_csv_path, index=False)

print("\n" + "=" * 80)
print(f"Longest example verification complete!")
print(f"Results saved to: {out_csv_path}")
print("=" * 80)
print(f"  Total adversarial examples found: {sum(1 for r in results if r.get('is_adversarial') == True)}")
print(f"  Total verified (no adversarial): {sum(1 for r in results if r.get('is_adversarial') == False)}")
print(f"  Total errors encountered: {sum(1 for r in results if 'error' in r)}")
