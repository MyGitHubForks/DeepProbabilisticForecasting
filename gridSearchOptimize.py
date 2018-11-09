from train import *
from joblib import Parallel, delayed
import os
import numpy as np
import argparse

params = {"h_dim": (4, 9, 2),
	"z_dim": (4, 9, 2),
	"batch_size": (1, 10, 5),
	"n_layers": (1, 3, 1),
	"initial_lr": (-4, -2, 10)}

def getSaveDir():
    saveDir = './save/models/model0/'
    while os.path.isdir(saveDir):
        numStart = saveDir.rfind("model")+5
        numEnd = saveDir.rfind("/")
        saveDir = saveDir[:numStart] + str(int(saveDir[numStart:numEnd])+1) + "/"
    os.mkdir(saveDir)
    return saveDir

def getParams(args, saveDir):
	p = {}
	p["save_dir"] = saveDir
	p["model"] = args.model
	for key in ["batch_size", "n_layers"]:
		vals = params[key]
		p[key] = np.random.randint(vals[0], vals[1]) * vals[2]
		print(key, p[key])

	for key in ["h_dim", "z_dim", "initial_lr"]:
		vals = params[key]
		possib = np.logspace(vals[0], vals[1], base=vals[2], num=vals[1]-vals[0]+1)
		p[key] = np.random.choice(possib)
		print(key, p[key])
	return p

def runExperiment(args, saveDir):
	p = getParams(args, saveDir)
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
	saveDirs = [getSaveDir() for i in range(tries)]
	results = Parallel(n_jobs=4)(delayed(runExperiment)(args, saveDirs[i]) for i in range(tries))
	# trainReconLosses, trainKLDLosses, valReconLosses, valKLDLosses, args.save_dir
	results = sorted(results, key=lambda x: x[2])
	saveFile = getSaveFile()
	if args.model == "rnn":
		with open(saveFile, "w+") as f:
			f.write("Save Directory\t\tTrain Loss\t\tValidation Loss\n")
			for tup in results:
				f.write("{}\t\t{:.3f}\t\t{:.3f}\n".format(tup[4], tup[0], tup[2]))
	elif args.model == "vrnn":
		with open(saveFile, "w+") as f:
			f.write("Save Directory\t\tTrain Recon Loss\tTrain KLD Loss\tValidation Recon Loss\tValidation KLD Loss\n")
			for tup in results:
				f.write("{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t\t{:.3f}\n".format(tup[4], tup[0],tup[1],tup[2], tup[3]))
	else:
		assert False, "bad model"



if __name__ == '__main__':
	main()
