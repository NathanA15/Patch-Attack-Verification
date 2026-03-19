from gurobipy import GRB


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