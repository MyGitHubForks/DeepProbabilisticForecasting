from train import train
from joblib import Parallel, delayed
import os

params = {"h_dim": np.linspace(16, 512, 20),
	"z_dim": np.linspace(16, 512,20),
	"batch_size": np.linspace(1, 256,20),
	"n_layers": [1,2,3],
	"initial_lr": np.linspace(.00001, .01, 10),
	"weight_decay": np.linspace(.000005, .5, 10),
	"scheduling_start": np.linspace(.3, .8, 10),
	"scheduling_end": np.linspace(0.0, 0.3, 10)}

def getParams():
	p = {}
	for key, vals in params.items():
		p[key] = np.random.choice(vals)
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
