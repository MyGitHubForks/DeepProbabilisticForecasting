from train import *
from joblib import Parallel, delayed
import os
import numpy as np
import argparse
import utils

log_params = {"h_dim": (4, 9, 2),
	"initial_lr": (-5, -2, 10),
	"batch_size": (4, 7, 2),
	"lambda_l1" : (-7, -4, 2),
	"lambda_l2" : (-6, -2, 5)
}

lin_params = {
	"n_layers": (2,5,1),
	"encoder_input_dropout" : (0.1, 0.9, 0.2),
	"encoder_layer_dropout" : (0.1, 0.9, 0.2),
	"decoder_input_dropout" : (0.1, 0.9, 0.2),
	"decoder_layer_dropout" : (0.1, 0.9, 0.2)
}

def getSaveDir():
   	saveDir = '../save/models/model0/'
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
	for key, vals in log_params.items():
		possib = np.logspace(vals[0], vals[1], base=vals[2], num=vals[1]-vals[0]+1)
		p[key] = np.random.choice(possib)

	for key, vals in lin_params.items():
		possib = np.arange(vals[0], vals[1], vals[2])
		p[key] = np.random.choice(possib)
	return p

def runExperiment(args, saveDir, data):
	p = getParams(args, saveDir)
	return p, trainF(data=data, suggestions=p)

def getSaveFile():
	saveFile = '../save/gridSearch/gridSearch_1.tsv'
	if not os.path.isdir("../save/"):
		os.mkdir("../save/")
	if not os.path.isdir("../save/gridSearch/"):
		os.mkdir("../save/gridSearch/")
	while os.path.isfile(saveFile):
		numStart = saveFile.rfind("_")+1
		numEnd = saveFile.rfind(".")
		saveFile = saveFile[:numStart] + str(int(saveFile[numStart:numEnd])+1) + ".tsv"
	return saveFile

def loadData(args):
	print("loading data")
        if args.dataset == "traffic":
            dataDir = "/home/dan/data/traffic/trafficWithTime/"
            data = utils.load_traffic_dataset(dataDir, args.batch_size, down_sample=args.down_sample, load_test=args.predictOnTest)
        elif args.dataset == "human":
            dataDir = "/home/dan/data/human/Processed/"
            data = utils.load_human_dataset(dataDir, args.batch_size, down_sample=args.down_sample, load_test=args.predictOnTest)

def main():
	args = parser.parse_args()
	data = loadData(args)
	tries = args.tries
	saveDirs = [getSaveDir() for i in range(tries)]
	results = Parallel(n_jobs=4)(delayed(runExperiment)(args, saveDirs[i], data) for i in range(tries))
	# trainReconLosses, trainKLDLosses, valReconLosses, valKLDLosses, args.save_dir
	results = sorted(results, key=lambda x: x[1][2])
	saveFile = getSaveFile()
	if args.model == "rnn":
		with open(saveFile, "w+") as f:
			f.write("Dataset: {}\tModel: {}\n".format(args.dataset, args.model))
			col = "Save Directory\tTrain Loss\tValidation Loss"
			sortedKeys = sorted(results[0][0].keys())
			for k in sortedKeys:
				if k not in ["model", "save_dir"]:
					col += "\t{}".format(k)
			col += "\n"
			f.write(col)
			for res in results:
				tup = res[1]
				row = "{}\t{:.3f}\t{:.3f}".format(tup[4], tup[0], tup[2])
				for k in sortedKeys:
					if k not in ["model", "save_dir"]:
						v = res[0][k]
						row+="\t{}".format(v)
				row += "\n"
				f.write(row)
	elif args.model == "vrnn" or args.model=="sketch-rnn":
		with open(saveFile, "w+") as f:
			f.write("Dataset: {}, Model: {}".format(args.dataset, args.model))
			f.write("Save Directory\t\tTrain Recon Loss\tTrain KLD Loss\tValidation Recon Loss\tValidation KLD Loss\n")
			for res in results:
				tup = res[1]
				f.write("{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t\t{:.3f}\n".format(tup[4], tup[0],tup[1],tup[2], tup[3]))
	else:
		assert False, "bad model"



if __name__ == '__main__':
	main()
