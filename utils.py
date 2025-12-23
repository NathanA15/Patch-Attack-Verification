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

