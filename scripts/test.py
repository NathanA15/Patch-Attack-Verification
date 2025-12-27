import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import patch_input_box
from patch_input_box import create_patch_input_config_file


def test_create_patch_input_config_file_with_dataset_image():
    # Same dataset loading pattern as run_batch_examples.py
    test_data_path = "/root/ERAN/data/mnist_test.csv"

    df = pd.read_csv(test_data_path, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    img_idx = 0
    img = pixels[img_idx].reshape(28, 28)
    label = labels[img_idx]

    output_dir = Path(__file__).resolve().parent / "input_box_configs"
    patch_input_box.input_box_configs_dir = str(output_dir)

    output_path = create_patch_input_config_file(img, 0, 0, 6, label=label)

    assert Path(output_path).exists()
    assert f"_label{label}_patch_box_0_0_6.txt" in output_path
    lines = Path(output_path).read_text().splitlines()
    assert len(lines) == 28 * 28

if __name__ == "__main__":
    test_create_patch_input_config_file_with_dataset_image()
