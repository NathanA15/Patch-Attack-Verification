import numpy as np
from datetime import datetime
import os
from config import *


# def stringify_box(box_config):
#     """
#     Convert a (h, w) object array of ranges into text with one pixel per line.
#     """
#     h, w = box_config.shape
#     lines = []

#     for row in range(h):
#         for col in range(w):
#             ranges_for_pixel = box_config[row, col]
#             if isinstance(ranges_for_pixel, np.ndarray) and ranges_for_pixel.ndim == 1:
#                 ranges_for_pixel = [ranges_for_pixel.tolist()]
#             lb, ub = ranges_for_pixel[0]
#             lines.append(f"[{lb},{ub}]")

#     return "\n".join(lines)


def stringify_box_new(box_config):
    """
    Convert a (h, w) object array of ranges into text with one pixel per line.
    Supports multiple ranges per pixel (concatenated on the same line).
    """
    h, w = box_config.shape
    lines = []

    for row in range(h):
        for col in range(w):
            ranges_for_pixel = box_config[row, col]
            if isinstance(ranges_for_pixel, np.ndarray) and ranges_for_pixel.ndim == 1:
                ranges_for_pixel = [ranges_for_pixel.tolist()]
            text = ""
            for pixel_range in ranges_for_pixel:
                text += f"[{pixel_range[0]},{pixel_range[1]}] "
            lines.append(text.strip())

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
        Array of shape (h, w, 2) representing the input box.
    """
    if c == 0:
        raise ValueError("Patch size c must be greater than 0.")
        
    image = np.asarray(image, dtype=np.float32) / 255.0  # now guaranteed in [0, 1]

    h, w = image.shape[:2]

    box_config = np.empty((h, w), dtype=object)
    for row in range(h):
        for col in range(w):
            val = float(image[row, col])
            box_config[row, col] = [[val, val]]

    # Patch mask selecting rows i..i+c-1 and cols j..j+c-1
    mask = np.zeros((h, w), dtype=bool)
    mask[i:i+c, j:j+c] = True

    # Replace those positions with [0, 1]
    for row, col in np.argwhere(mask):
        box_config[row, col] = [[0,0.65]]

    return box_config

def create_patch_input_config_file(image, i, j, c, label, split_pixels_list=None, split_pixel_range=None, split_amounts=1):
    """
    Creates a patch input config file.

    Args:
        image: The image.
        i (int): The row index of the patch.
        j (int): The column index of the patch.
        c: patch size.
        label: The true label of the image.
        split_pixels_list: List of (x, y) coordinates of pixels to split.
        split_pixel_range: List of (lb, ub) ranges for the split pixels. example: [[[0, 0.5], [0.5, 1]], [[0, 0.5], [0.5, 1]], ...]
    Returns:
        None
    """
    box_config = create_patch_input_box(image, i, j, c)

    if split_amounts is not None and split_amounts > 1:
        for idx, pixel in enumerate(split_pixels_list):
            ranges = []
            step = 1.0 / split_amounts
            for s in range(split_amounts):
                lb = s * step
                ub = (s + 1) * step
                ranges.append([lb, ub])
            print(pixel)
            print(ranges)
            box_config[pixel[1], pixel[0]] = ranges
        pass
    elif split_pixels_list is not None and split_pixel_range is not None:
        for idx, pixel in enumerate(split_pixels_list):
            ranges = split_pixel_range[idx]
            if isinstance(ranges, np.ndarray):
                ranges = ranges.tolist()
            # Ensure we store a list of ranges (each range is [lb, ub])
            if len(ranges) > 0 and isinstance(ranges[0], (float, int)):
                ranges = [ranges]
            print(pixel)
            print(ranges)
            box_config[pixel[1], pixel[0]] = ranges

    # text = stringify_box(box_config)
    text = stringify_box_new(box_config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    date_stamp = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(
        BOX_DIR,
        date_stamp,
        f"{timestamp}_label{label}_patch_box_{i}_{j}_{c}.txt"
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(text)
    
    return os.path.abspath(filename)
