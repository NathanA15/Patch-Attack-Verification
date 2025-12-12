import numpy as np
from datetime import datetime
import os

input_box_configs_dir = "input_box_configs"

def stringify_box(box_config):
    """
    Convert a (h, w, 2) box_config array into a text format where
    EACH pixel is on its own line, like:
    [lb,ub]
    [lb,ub]
    ...
    """
    h, w, _ = box_config.shape
    lines = []

    for row in range(h):
        for col in range(w):
            lb, ub = box_config[row, col]
            lines.append(f"[{lb},{ub}]")

    return "\n".join(lines)


def create_patch_input_box(image, i, j, c):
    """
    Creates a patch input box.

    Args:
        image: The image.
        i (int): The row index of the patch.
        j (int): The column index of the patch.
        c: patch size.
    Returns:
        
    """
    if c == 0:
        h, w = image.shape[:2]
        # put zeros everywhere 
        # Start by copying each pixel twice: shape (h, w, 2)
        box_config = np.stack([image, image], axis=2).astype(float)

        # Replace those positions with [0, 255]
        box_config[:, :] = [0, 0]

        return box_config
        
    image = np.asarray(image, dtype=np.float32) / 255.0  # now guaranteed in [0, 1]

    h, w = image.shape[:2]

    # Start by copying each pixel twice: shape (h, w, 2)
    box_config = np.stack([image, image], axis=2).astype(float)

    # Patch mask selecting rows i..i+c-1 and cols j..j+c-1
    mask = np.zeros((h, w), dtype=bool)
    mask[i:i+c, j:j+c] = True

    # Replace those positions with [0, 1]
    box_config[mask] = [0, 1]

    return box_config

def create_patch_input_config_file(image, i, j, c, label):
    """
    Creates a patch input config file.

    Args:
        image: The image.
        i (int): The row index of the patch.
        j (int): The column index of the patch.
        c: patch size.
    Returns:
        None
    """
    box_config = create_patch_input_box(image, i, j, c)
    text = stringify_box(box_config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(
        input_box_configs_dir,
        f"{timestamp}_label{label}_patch_box_{i}_{j}_{c}.txt"
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(text)
    
    return os.path.abspath(filename)
