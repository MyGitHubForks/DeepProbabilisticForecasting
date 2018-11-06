from train import train
from joblib import Parallel, delayed
import os

params = {"h_dim": (16, 513),
	"z_dim": (16, 513),
	"batch_size": (1, 257)
	"n_layers": (1,4),
	"initial_lr": (-5, -1),
	"weight_decay": (-6, -1),
	"scheduling_start": (.3, .8),
	"scheduling_end": (0.0, 0.3)}

def getParams():
	p = {}
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

def runExperiment():
	p = getParams()
	return train(p)

def getSaveFile():
	saveFile = './save/gridSearch/gridSearch_1.txt'
	if not os.path.isdir("./save/"):
		os.mkdir("./save/")
	if not os.path.isdir("./save/gridSearch/"):
		os.mkdir("./save/gridSearch/")
	while os.path.isfile(saveFile):
		numStart = saveDir.rfind("_")+1
		numEnd = saveDir.rfind(".")
		saveFile = saveFile[:numStart] + str(int(saveFile[numStart:numEnd])+1) + ".txt"
	return saveFile

def main():
	saveFile = getSaveFile()
	results = Parallel(n_jobs=5)(delayed(runExperiment)() for i in range(10)) # train, val, saveDir
	results = sorted(results, key=lambda x: x[1])
	with open(saveFile) as f:
		f.write("Save Directory\tTrain Loss\tValidation Loss\n")
		for tup in results:
			f.write("{}\t{:.3f}\t{:.3f}\n".format(tup[2], tup[0], tup[1]))



if __name__ == '__main__':
	main()
