from train import *
from joblib import Parallel, delayed
import os
import numpy as np
import argparse

params = {"h_dim": (250, 1024),
	"z_dim": (250, 1024),
	"batch_size": (16, 257),
	"n_layers": (1,4),
	"initial_lr": (-5, -3),
	"weight_decay": (-8, -2),
	"scheduling_start": (.3, .8),
	"scheduling_end": (0.0, 0.3)}

def getParams(args):
	p = {}
	p["model"] = args.model
	for key in ["h_dim", "z_dim", "batch_size", "n_layers"]:
		vals = params[key]
		p[key] = np.random.randint(vals[0], vals[1])

	for key in ["weight_decay", "initial_lr"]:
		vals = params[key]
		possib = np.logspace(vals[0], vals[1], num=5)
		p[key] = np.random.choice(possib)

	for key in ["scheduling_end", "scheduling_start"]:
		vals = params[key]
		possib = np.linspace(vals[0], vals[1], num=5)
		p[key] = np.random.choice(possib)
	return p

def runExperiment(args):
	p = getParams(args)
	return trainF(p)

def getSaveFile():
	saveFile = './save/gridSearch/gridSearch_1.txt'
	if not os.path.isdir("./save/"):
		os.mkdir("./save/")
	if not os.path.isdir("./save/gridSearch/"):
		os.mkdir("./save/gridSearch/")
	while os.path.isfile(saveFile):
		numStart = saveFile.rfind("_")+1
		numEnd = saveFile.rfind(".")
		saveFile = saveFile[:numStart] + str(int(saveFile[numStart:numEnd])+1) + ".txt"
	return saveFile

def main():
	args = parser.parse_args()
	tries = args.tries
	saveFile = getSaveFile()
	results = Parallel(n_jobs=3)(delayed(runExperiment)(args) for i in range(tries)) # train, val, saveDir
	results = sorted(results, key=lambda x: x[1])
	with open(saveFile, "w+") as f:
		f.write("Save Directory\t\tTrain Loss\t\tValidation Loss\n")
		for tup in results:
			f.write("{}\t\t{:.3f}\t\t{:.3f}\n".format(tup[2], tup[0], tup[1]))



if __name__ == '__main__':
	main()
