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