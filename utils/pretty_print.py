import numpy as np

def pprint(a: np.ndarray):
	for i in range(a.shape[0]):
		row = ""
		for j in range(a.shape[1]):
			row += str(a[i, j])
		print(row)
