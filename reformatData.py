import numpy as np
from joblib import Parallel, delayed
import torch

def do(dsPath, i, name):
	if (i % 100) == 0:
		print(i)
	ds = np.load(dsPath)
	x = ds["x"][i,:,:,0]
	y = ds["y"][i,:,:,0]
	torch.save(x, "./data/individualizedData/{}/features/{}".format(name, str(i)))
	torch.save(y, "./data/individualizedData/{}/labels/{}".format(name, str(i)))


def main():
	datasets = []
	datasets.append(("train","./data/train.npz"))
	datasets.append(("val","./data/val.npz"))
	datasets.append(("test","./data/test.npz"))

	for name, dsPath in datasets:
		print(name)
		ds = np.load(dsPath)
		setLength = ds["x"].shape[0]
		ids = range(setLength)
		torch.save(ids, "./data/individualizedData/{}/idList".format(name))
		Parallel(n_jobs=-1)(delayed(do)(dsPath, i, name) for i in range(setLength))

if __name__ == '__main__':
	main()