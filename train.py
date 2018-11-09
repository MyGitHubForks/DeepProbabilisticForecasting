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
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Batched Sequence to Sequence')
parser.add_argument('--h_dim', type=int, default=512)
parser.add_argument("--z_dim", type=int, default=512)
parser.add_argument('--no_cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument("--no_attn", action="store_true", default=True, help="Do not use AttnDecoder")
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default= 1)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--initial_lr", type=float, default=1e-4)
parser.add_argument("--no_lr_decay", action="store_true", default=False)
parser.add_argument("--lr_decay_ratio", type=float, default=0.10)
parser.add_argument("--lr_decay_beginning", type=int, default=20)
parser.add_argument("--lr_decay_every", type=int, default=10)
parser.add_argument("--print_every", type=int, default = 20)
parser.add_argument("--plot_every", type=int, default = 1)
parser.add_argument("--criterion", type=str, default="L1Loss")
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--down_sample", type=float, default=0, help="Keep this fraction of the training data")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--model", type=str, default="rnn")
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument("--no_schedule_sampling", action="store_true", default=False)
parser.add_argument("--scheduling_start", type=float, default=1.0)
parser.add_argument("--scheduling_end", type=float, default=0.0)
parser.add_argument("--tries", type=int, default=10)

def trainF(suggestions=None):
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
    print("setting additional params")
    # Set additional arguments
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args._device = "cuda" if args.cuda else "cpu"
    args.use_attn = not args.no_attn
    args.x_dim = data['x_dim']
    args.sequence_len = data['sequence_len']
    args.use_schedule_sampling = not args.no_schedule_sampling
    print("use schedule sampling", args.use_schedule_sampling)

    print("generating model")
    if args.model == "vrnn":
        print("using vrnn")
        model = VRNN(args)
    elif args.model == "rnn":
        print("using Seq2Seq RNN")
        model = Seq2Seq(args)
    else:
        assert False, "Model incorrectly specified"
    if args.cuda:
        model = model.cuda()

    trainReconLosses = []
    valReconLosses = []
    trainKLDLosses = []
    valKLDLosses = []
    lr = args.initial_lr
    argsFile = args.save_dir + "args.txt"
    with open(argsFile, "w") as f:
        f.write(json.dumps(vars(args)))
    print("saved args to "+argsFile)
    print("beginning training")
    for epoch in range(1, args.n_epochs + 1):
        print("epoch {}".format(epoch))
        if not args.no_lr_decay and epoch > args.lr_decay_beginning and epoch % args.lr_decay_every:
            lr = lr * (1 - args.lr_decay_ratio)
        avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss = utils.train(data['train_loader'].get_iterator(), data['val_loader'].get_iterator(), model, lr, args, data, epoch)
        if (epoch % args.plot_every) == 0:
            trainReconLosses.append(avgTrainReconLoss)
            valReconLosses.append(avgValReconLoss)
            trainKLDLosses.append(avgTrainKLDLoss)
            valKLDLosses.append(avgValKLDLoss)
        #saving model
        if (epoch % args.save_freq) == 0:
            fn = args.save_dir+'{}_state_dict_'.format(args.model)+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
    model_fn = args.save_dir + '{}_full_model'.format(args.model) +".pth"
    torch.save(model, model_fn)
    if args.model == "VRNN":
        utils.plotTrainValCurve(trainReconLosses, valReconLosses, args.model, args.criterion, args, trainKLDLosses=trainKLDLosses, valKLDLosses=valKLDLosses)
    else:
        utils.plotTrainValCurve(trainReconLosses, valReconLosses, args.model, args.criterion, args)
    predsV, targetsV, datasV, meansV, stdsV = utils.getPredictions(args,\
        data['val_loader'].get_iterator(), model, data["val_mean"], data["val_std"])
    
    predsT, targetsT, datasT, meansT, stdsT = utils.getPredictions(args, \
        data['train_loader'].get_iterator(), model, data["train_mean"], data["train_std"])

    # Save predictions based on model output
    if args.model == "rnn":
        torch.save(predsT, args.save_dir+"train_preds")
        torch.save(targetsT, args.save_dir+"train_targets")
        torch.save(datasT, args.save_dir+"train_datas")
        torch.save(predsV, args.save_dir+"validation_preds")
        torch.save(targetsV, args.save_dir+"validation_targets")
        torch.save(datasV, args.save_dir+"validation_datas")
    elif args.model == "vrnn":
        # Save train prediction data
        torch.save(meansT, args.save_dir+"train_means")
        torch.save(stdsT, args.save_dir+"train_stds")
        torch.save(targetsT, args.save_dir+"train_targets")
        torch.save(datasT, args.save_dir+"train_datas")
        torch.save(data["train_mean"], args.save_dir+"train_mean")
        torch.save(data["train_std"], args.save_dir+"train_std")
        # Validation prediction data
        torch.save(meansV, args.save_dir+"validation_means")
        torch.save(stdsV, args.save_dir+"validation_stds")
        torch.save(targetsV, args.save_dir+"validation_targets")
        torch.save(datasV, args.save_dir+"validation_datas")
        torch.save(data["val_mean"], args.save_dir+"val_mean")
        torch.save(data["val_std"], args.save_dir+"val_std")
    return trainReconLosses[-1], trainKLDLosses[-1], valReconLosses[-1], valKLDLosses[-1], args.save_dir

if __name__ == '__main__':
        cProfile.run("trainF()", "restats")
        p = pstats.Stats('restats')
        p.sort_stats("tottime").print_stats(10)
