import csv
import time
from datetime import datetime
from pathlib import Path

from gurobipy import GRB

from config import CSV_DIR, LOG_DIR


BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS = [
    "image_index",
    "true_label",
    "adv_label",
    "patch_x",
    "patch_y",
    "patch_size",
    "upper_bound",
    "timeout_milp_s",
    "add_bool_constraints",
    "use_refine_poly",
    "middle_bound",
    "skip_singleton_bounds",
    "enable_split_bit_branch_priority",
    "split_bit_branch_priority",
    "split_indices",
    "run_scope",
    "global_attempt_index",
    "label_attempt_index",
    "current_depth",
    "status_code",
    "status_name",
    "final_outcome",
    "is_timeout",
    "is_adversarial_found",
    "iteration_runtime_seconds",
    "iteration_runtime_source",
    "attempt_runtime_seconds",
    "parent_total_run_time_seconds",
    "parent_status",
    "parent_attempt_count",
    "parent_max_depth_reached",
    "log_path",
]


RUN_CONFIG_COLUMNS = [
    "top_k",
    "split_selection_mode",
    "split_random_seed",
]


GUROBI_STATS_COLUMNS = [
    "linear_constraint_matrix_constrs",
    "linear_constraint_matrix_vars",
    "linear_constraint_matrix_nzs",
    "variable_types_continuous",
    "variable_types_integer",
    "variable_types_binary",
    "general_constraints",
    "matrix_coefficient_range",
    "objective_coefficient_range",
    "variable_bound_range",
    "rhs_coefficient_range",
    "linear_constraint_matrix_constrs_presolved",
    "linear_constraint_matrix_vars_presolved",
    "linear_constraint_matrix_nzs_presolved",
    "variable_types_continuous_presolved",
    "variable_types_integer_presolved",
    "variable_types_binary_presolved",
    "general_constraints_presolved",
    "matrix_coefficient_range_presolved",
    "objective_coefficient_range_presolved",
    "variable_bound_range_presolved",
    "rhs_coefficient_range_presolved",
    "optimizer_presolve_removed_rows",
    "optimizer_presolve_removed_columns",
    "optimizer_presolve_removed_steps",
    "optimizer_presolve_time_seconds",
    "optimizer_presolved_rows",
    "optimizer_presolved_columns",
    "optimizer_presolved_nonzeros",
    "optimizer_presolved_variable_types_continuous",
    "optimizer_presolved_variable_types_integer",
    "optimizer_presolved_variable_types_binary",
    "extra_simplex_iterations_after_uncrush",
    "extra_simplex_iterations_dual_to_original",
    "root_relaxation_result",
    "root_relaxation_objective",
    "root_relaxation_iterations",
    "root_relaxation_seconds",
    "explored_nodes",
    "explored_simplex_iterations",
    "explored_time_seconds",
    "mip_final_status",
    "mip_final_runtime_seconds",
    "mip_final_nodes",
    "mip_final_objbound",
    "mip_final_solcnt",
    "relaxed_presolve_removed_rows",
    "relaxed_presolve_removed_columns",
    "relaxed_presolve_removed_steps",
    "relaxed_presolve_time_seconds",
    "relaxed_presolved_rows",
    "relaxed_presolved_columns",
    "relaxed_presolved_nonzeros",
    "relaxed_presolved_variable_types_continuous",
    "relaxed_presolved_variable_types_integer",
    "relaxed_presolved_variable_types_binary",
    "relaxed_extra_simplex_iterations_after_uncrush",
    "relaxed_extra_simplex_iterations_dual_to_original",
    "relaxed_barrier_solved_iterations",
    "relaxed_barrier_solved_seconds",
    "relaxed_solved_iterations",
    "relaxed_solved_seconds",
    "relaxed_optimal_objective",
]


RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS = (
    BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS + RUN_CONFIG_COLUMNS + GUROBI_STATS_COLUMNS
)


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
    Build the detailed CSV path for recursive timeout refinement runs.
    """
    timestamp = run_id or time.strftime("%Y_%m_%d_%H_%M_%S")
    return Path(CSV_DIR) / f"recursive_timeout_refinement_{timestamp}.csv"


def resolve_runs_csv_path(csv_path=None, run_id=None):
    """
    Resolve the detailed CSV path for recursive timeout refinement runs.

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


def validate_recursive_timeout_header(header, csv_path):
    """
    Ensure an existing CSV uses the recursive timeout refinement base schema.
    """
    if not header:
        return

    base_len = len(BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS)
    if header[:base_len] != BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS:
        raise ValueError(
            f"Existing CSV header at {csv_path} does not match the recursive timeout "
            "refinement base schema. Use a new --csv-path or migrate the existing CSV first."
        )


def build_dynamic_csv_header(existing_header, row):
    """
    Preserve base columns first, then append known and newly discovered stats.
    """
    if not existing_header:
        header = list(RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS)
    else:
        header = list(BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS)
        for column in RUN_CONFIG_COLUMNS:
            if column not in header:
                header.append(column)
        for column in existing_header[len(BASE_RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS) :]:
            if column not in header:
                header.append(column)

    for column in RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS:
        if column not in header:
            header.append(column)

    for column in sorted(row.keys()):
        if column not in header:
            header.append(column)

    return header


def rewrite_csv_with_header(csv_path, header, existing_rows):
    """
    Rewrite a CSV after adding new dynamic columns.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({column: existing_row.get(column, "") for column in header})


def append_run_row(csv_path, row, save_csv=True):
    """
    Append one detailed row for a recursive timeout refinement run.
    """
    if not save_csv:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = csv_path.exists() and csv_path.stat().st_size > 0
    existing_rows = []
    existing_header = []

    if has_content:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        validate_recursive_timeout_header(existing_header, csv_path)

    header = build_dynamic_csv_header(existing_header, row)
    if has_content and header != existing_header:
        rewrite_csv_with_header(csv_path, header, existing_rows)
    elif has_content:
        ensure_trailing_newline(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not has_content:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in header})


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
