import numpy as np
import cProfile
import model.Data as DF
import model.EncoderRNN as ERF
import model.AttnDecoder as ADRF
import utils
import pstats
import torch
from torch.utils import data
from model.RoseSeq2Seq import Seq2Seq
from model.vrnn.model import VRNN
from model.SketchyRNN import SketchyRNN
import os
import argparse
import json
from shutil import copy2, copyfile, copytree

parser = argparse.ArgumentParser(description='Batched Sequence to Sequence')
parser.add_argument('--h_dim', type=int, default=512)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument('--no_cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument("--no_attn", action="store_true", default=True, help="Do not use AttnDecoder")
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default= 10)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--initial_lr", type=float, default=1e-3)
parser.add_argument("--print_every", type=int, default = 20)
parser.add_argument("--plot_every", type=int, default = 1)
parser.add_argument("--criterion", type=str, default="RMSE")
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--down_sample", type=float, default=0.1, help="Keep this fraction of the training data")
parser.add_argument("--data_dir", type=str, default="./data/reformattedTraffic/")
parser.add_argument("--model", type=str, default="sketch-rnn")
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument("--no_schedule_sampling", action="store_true", default=False)
parser.add_argument("--scheduling_start", type=float, default=1.0)
parser.add_argument("--scheduling_end", type=float, default=0.0)
parser.add_argument("--tries", type=int, default=10)
parser.add_argument("--kld_warmup_until", type=int, default=5)
parser.add_argument("--kld_weight_max", type=float, default=0.10)
parser.add_argument("--no_shuffle_after_epoch", action="store_true", default=False)
parser.add_argument("--clip", type=int, default=10)

def savePredData(experimentData):
    # Save predictions based on model output
    for f in ["targetsT", "datasT", "targetsV", "datasV", "learningRates", "dataTimesArrTrain", "targetTimesArrTrain", "dataTimesArrVal", "targetTimesArrVal"]:
        torch.save(experimentData[f], args.save_dir+f)
    if args.model == "rnn":
        torch.save(experimentData["predsT"], experimentData["args"].save_dir+"train_preds")
        torch.save(experimentData["predsV"], experimentData["args"].save_dir+"validation_preds")
    elif args.model == "vrnn" or args.model == "sketch-rnn":
        # Save train prediction data
        torch.save(experimentData["meansT"], experimentData["args"].save_dir+"train_means")
        torch.save(experimentData["stdsT"], experimentData["args"].save_dir+"train_stds")
        torch.save(experimentData["data"]["train_mean"], experimentData["args"].save_dir+"train_mean")
        torch.save(experimentData["data"]["train_std"], experimentData["args"].save_dir+"train_std")
        torch.save(experimentData["meanKLDLossesT"], experimentData["args"].save_dir+"mean_train_kld_losses_per_timestep")
        # Validation prediction data
        torch.save(experimentData["meansV"], experimentData["args"].save_dir+"validation_means")
        torch.save(experimentData["stdsV"], experimentData["args"].save_dir+"validation_stds")
        torch.save(experimentData["data"]["val_mean"], experimentData["args"].save_dir+"val_mean")
        torch.save(experimentData["data"]["val_std"], experimentData["args"].save_dir+"val_std")
        torch.save(experimentData["meanKLDLossesV"], experimentData["args"].save_dir+"mean_validation_kld_losses_per_timestep")
    if args.model == "sketch-rnn":
        torch.save(experimentData["trainingZs"], experimentData["args"].save_dir+"train_Zs")
        torch.save(experimentData["validationZs"], experimentData["args"].save_dir+"validation_Zs")

def trainF(suggestions=None):
    experimentData = {}
    args = parser.parse_args()
    if not suggestions:
        saveDir = './save/models/model0/'
        while os.path.isdir(saveDir):
            numStart = saveDir.rfind("model")+5
            numEnd = saveDir.rfind("/")
            saveDir = saveDir[:numStart] + str(int(saveDir[numStart:numEnd])+1) + "/"
        os.mkdir(saveDir)
        args.save_dir = saveDir
    if suggestions:
        args.model = suggestions["model"]
        args.h_dim = int(suggestions["h_dim"])
        args.z_dim = int(suggestions["z_dim"])
        args.batch_size = suggestions["batch_size"]
        args.n_layers = suggestions["n_layers"]
        args.initial_lr = suggestions["initial_lr"]
        args.save_dir = suggestions["save_dir"]

    print("loading data")
    data = utils.load_dataset(args.data_dir, args.batch_size, down_sample=args.down_sample)
    experimentData["data"] = data
    print("setting additional params")
    # Set additional arguments
    assert args.kld_warmup_until <= args.n_epochs, "KLD Warm up stop > n_epochs"
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args._device = "cuda" if args.cuda else "cpu"
    args.use_attn = not args.no_attn
    args.x_dim = data['x_dim']
    args.sequence_len = data['sequence_len']
    args.use_schedule_sampling = not args.no_schedule_sampling
    argsFile = args.save_dir + "args.txt"
    with open(argsFile, "w") as f:
        f.write(json.dumps(vars(args)))
    copy2("./train.py", args.save_dir+"train.py")
    copy2("./utils.py", args.save_dir+"utils.py")
    copy2("./gridSearchOptimize.py", args.save_dir+"gridsearchOptimize.py")
    copytree("./model", args.save_dir+"model/")
    print("saved args to "+argsFile)
    
    print("generating model")
    if args.model == "sketch-rnn":
        model=SketchyRNN(args)
    elif args.model == "vrnn":
        print("using vrnn")
        model = VRNN(args)
    elif args.model == "rnn":
        print("using Seq2Seq RNN")
        model = Seq2Seq(args)
    else:
        assert False, "Model incorrectly specified"
    if args.cuda:
        model = model.cuda()
    experimentData["args"] = args
    experimentData["trainReconLosses"] = []
    experimentData["valReconLosses"] = []
    experimentData["trainKLDLosses"] = []
    experimentData["valKLDLosses"] = []
    experimentData["learningRates"] = []
    lr = args.initial_lr
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    print("beginning training")
    for epoch in range(1, args.n_epochs + 1):
        print("epoch {}".format(epoch))
        kldLossWeight = args.kld_weight_max * (epoch / (args.kld_warmup_until))
        avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss = \
                    utils.train(data['train_loader'].get_iterator(), data['val_loader'].get_iterator(),\
                                model, lr, args, data, epoch, optimizer, kldLossWeight)
        if (epoch % args.plot_every) == 0:
            experimentData["trainReconLosses"].append(avgTrainReconLoss)
            experimentData["valReconLosses"].append(avgValReconLoss)
            experimentData["trainKLDLosses"].append(avgTrainKLDLoss)
            experimentData["valKLDLosses"].append(avgValKLDLoss)
            lrs = []
            for param_group in optimizer.param_groups:
                lrs.append(param_group["lr"])
            experimentData["learningRates"].append(lrs)
        #saving model
        if (epoch % args.save_freq) == 0:
            fn = args.save_dir+'{}_state_dict_'.format(args.model)+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
        if not args.no_shuffle_after_epoch:
            # Shuffle training examples for next epoch
            data['train_loader'].shuffle()
    model_fn = args.save_dir + '{}_full_model'.format(args.model) +".pth"
    torch.save(model, model_fn)
    if args.model == "vrnn" or args.model == "sketch-rnn":
        utils.plotTrainValCurve(experimentData["trainReconLosses"], experimentData["valReconLosses"],\
            args.model, args.criterion, args, trainKLDLosses=experimentData["trainKLDLosses"],\
            valKLDLosses=experimentData["valKLDLosses"])
    else:
        utils.plotTrainValCurve(experimentData["trainReconLosses"], experimentData["valReconLosses"],\
            args.model, args.criterion, args)

    experimentData["predsV"], experimentData["targetsV"], experimentData["datasV"], experimentData["meansV"],\
        experimentData["stdsV"], experimentData["meanKLDLossesV"], experimentData["dataTimesArrVal"],\
        experimentData["targetTimesArrVal"], experimentData["validationZs"] = utils.getPredictions(args,\
        data['val_loader'].get_iterator(), model, data["val_mean"], data["val_std"])
    
    experimentData["predsT"], experimentData["targetsT"], experimentData["datasT"], experimentData["meansT"],\
        experimentData["stdsT"], experimentData["meanKLDLossesT"], experimentData["dataTimesArrTrain"],\
        experimentData["targetTimesArrTrain"], experimentData["trainingZs"] = utils.getPredictions(args, \
        data['train_loader'].get_iterator(), model, data["train_mean"], data["train_std"])

    savePredData(experimentData)
    return trainReconLosses[-1], trainKLDLosses[-1], valReconLosses[-1], valKLDLosses[-1], args.save_dir

if __name__ == '__main__':
        cProfile.run("trainF()", "restats")
        p = pstats.Stats('restats')
        p.sort_stats("tottime").print_stats(10)
