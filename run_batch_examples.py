"""
This script runs the verify image, on N first images with increasing patch sizes and logs the results to a CSV file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


from patch_input_box import *
from run_verifier import *
import os

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



output_log_path = "/root/Projects/Nathan/Patch-Attack-Verification/patch_attack_verification_log.csv"

# Configuration parameters
N_IMAGES = 10  # Number of images to test
PATCH_SIZES = [3, 5, 6, 7, 8, 9,10, 11, 12]  # Patch sizes to test
# PATCH_SIZES = [9, 10, 11, 13]  # Patch sizes to test

PATCH_X = 0  # X coordinate for patch (center of 28x28 image)
PATCH_Y = 0  # Y coordinate for patch
TIMEOUT_MILP = 1800  # Timeout in seconds for MILP

# Initialize results list
results = []

print(f"Starting batch verification: {N_IMAGES} images with patch sizes {PATCH_SIZES}")
print(f"Results will be saved to: {output_log_path}")
print("-" * 80)

# Loop through different patch sizes first
for patch_size in PATCH_SIZES:
    print(f"\nProcessing Patch Size: {patch_size}")
    
    # Loop through N first images for each patch size
    for img_idx in range(N_IMAGES):
        label = labels[img_idx]
        print(f"  Testing image {img_idx} (Label: {label})...", end=" ")
        
        try:
            elapsed_time, last_status, failed_labels, example, is_adversarial = verify_image(
                img_index=img_idx,
                pixels=pixels,
                labels=labels,
                x_box=PATCH_X,
                y_box=PATCH_Y,
                size_box=patch_size,
                timeout_milp=TIMEOUT_MILP,
                with_plots=False
            )
            
            # Log the results
            result = {
                'image_index': img_idx,
                'true_label': label,
                'patch_size': patch_size,
                'patch_x': PATCH_X,
                'patch_y': PATCH_Y,
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
                'image_index': img_idx,
                'true_label': label,
                'patch_size': patch_size,
                'patch_x': PATCH_X,
                'patch_y': PATCH_Y,
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
print(f"Batch verification complete!")
print(f"Total runs: {len(results)}")
print(f"Results saved to: {output_log_path}")
print("=" * 80)

# Print summary statistics
print("\nSummary:")
print(f"  Total images tested: {N_IMAGES}")
print(f"  Patch sizes tested: {PATCH_SIZES}")
adversarial_count = sum(1 for r in results if r.get('is_adversarial') == True)
verified_count = sum(1 for r in results if r.get('is_adversarial') == False)
error_count = sum(1 for r in results if 'error' in r)
print(f"  Adversarial examples found: {adversarial_count}")
print(f"  Verified (no adversarial): {verified_count}")
print(f"  Errors encountered: {error_count}")