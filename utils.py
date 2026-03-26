import csv
import time
from datetime import datetime
from pathlib import Path

from gurobipy import GRB

from config import CSV_DIR, LOG_DIR


RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS = [
    "Image Index",
    "X patch",
    "Y patch",
    "Patch Size",
    "Upper bound",
    "Initial Timeout Labels",
    "Finally Verified Labels",
    "Adversarial Labels",
    "Unresolved Labels",
    "Total Run Time (s)",
    "Max Depth Reached",
    "Attempt Count",
    "Status",
    "Comment",
    "Log paths",
    "CSV path",
]


def dump_callback_codes():
	pairs = []
	for name in dir(GRB.Callback):
		if name.startswith("_"):
			continue
		try:
			val = getattr(GRB.Callback, name)
		except Exception:
			continue
		if isinstance(val, int):
			pairs.append((val, name))
	pairs.sort()
	for val, name in pairs:
		print(val, name)

def convert_index_of_patch_pixel_to_coordinates(index, x_patch, y_patch, patch_size):
	"""
	Convert an index of a pixel within the patch to its (x, y) coordinates in the image.

	Args:
		index (int): The index of the pixel within the patch (0 to patch_size*patch_size - 1).
		x_patch (int): The x-coordinate of the top-left corner of the patch in the image.
		y_patch (int): The y-coordinate of the top-left corner of the patch in the image.
		patch_size (int): The size of the patch (assumed square).

	Returns:
		(int, int): The (x, y) coordinates of the pixel in the image.
	"""
	row_in_patch = index // patch_size
	col_in_patch = index % patch_size
	x_coord = x_patch + col_in_patch
	y_coord = y_patch + row_in_patch
	return (x_coord, y_coord)


def generate_initial_bounds(num_pixels, lb=0, ub=1):
    bounds = [] 
    for i in range(num_pixels):
        bounds.append([[lb, ub]])
    return bounds


def build_runs_csv_path(run_id=None):
    """
    Build the summary CSV path for recursive timeout refinement runs.
    """
    timestamp = run_id or time.strftime("%Y_%m_%d_%H_%M_%S")
    return Path(CSV_DIR) / f"recursive_timeout_refinement_{timestamp}.csv"


def resolve_runs_csv_path(csv_path=None, run_id=None):
    """
    Resolve the summary CSV path for recursive timeout refinement runs.

    If ``csv_path`` is provided, it is reused as-is so multiple runs can append
    to the same CSV. Otherwise a timestamped per-run CSV path is generated.
    """
    if csv_path is not None:
        return Path(csv_path)
    return build_runs_csv_path(run_id)


def build_run_log_dir(run_id=None):
    """
    Build the per-run log directory for recursive timeout refinement.
    """
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_folder = LOG_DIR / datetime.now().strftime("%Y%m%d")
    run_log_dir = daily_folder / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return run_log_dir


def fresh_log_file(run_log_dir, attempt_idx, adv_label):
    """
    Build a unique log file path for a single ERAN attempt.
    """
    label_tag = "all_labels" if adv_label == -1 else f"adv_label_{adv_label}"
    return Path(run_log_dir) / f"{attempt_idx:03d}_{label_tag}_eran_run.log"


def ensure_trailing_newline(path):
    """
    Ensure an appended CSV continues on a new line.
    """
    path = Path(path)
    if not path.exists():
        return

    with path.open("rb") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            return
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            with path.open("ab") as out_handle:
                out_handle.write(b"\n")


def append_run_row(csv_path, row, save_csv=True):
    """
    Append one summary row for a recursive timeout refinement run.
    """
    if not save_csv:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    if file_exists:
        ensure_trailing_newline(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS)
        writer.writerow([row.get(col, "") for col in RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS])


def elapsed_seconds(elapsed_time):
    """
    Return elapsed time as seconds from either timedelta or numeric input.
    """
    if hasattr(elapsed_time, "total_seconds"):
        return float(elapsed_time.total_seconds())
    return float(elapsed_time)


def subdivide_bounds_at_indices(bounds_list_per_pixel, indices_to_split):
    """
    Subdivide the bounds of specific pixels by splitting each interval in half.

    Args:
        bounds_list_per_pixel (list): A 1D list where each element is a list of
            [lb, ub] intervals for that pixel.
            Example: [[[0, 1]], [[0, 1]]]
        indices_to_split (list): List of pixel indices whose bounds should be subdivided.

    Returns:
        list: A new bounds_list_per_pixel with the specified pixels' intervals split in half.

    Example:
        >>> bounds = [[[0, 1]], [[0, 1]]]
        >>> subdivide_bounds_at_indices(bounds, [1])
        [[[0, 1]], [[0, 0.5], [0.5, 1]]]
    """
    result = [list(intervals) for intervals in bounds_list_per_pixel]

    for idx in indices_to_split:
        current_intervals = result[idx]
        new_intervals = []
        for lb, ub in current_intervals:
            mid = (lb + ub) / 2.0
            new_intervals.append([lb, mid])
            new_intervals.append([mid, ub])
        result[idx] = new_intervals

    return result
