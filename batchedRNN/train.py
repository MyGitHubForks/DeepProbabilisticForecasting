import numpy as np
import cProfile
import model.Data as DF
import utils
import pstats
import torch
from torch.utils import data
from model.RoseSeq2Seq import Seq2Seq
from model.vrnn.model import VRNN
from model.SketchRNN import SketchyRNN
import os
import argparse
import json
from shutil import copy2, copyfile, copytree
import gc
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='Batched Sequence to Sequence')
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument("--z_dim", type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument("--no_attn", action="store_true", default=True, help="Do not use AttnDecoder")
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default= 64)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--initial_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_every", type=int, default=10)
parser.add_argument("--lr_decay_factor", type=float, default=.10)
parser.add_argument("--lr_decay_beginning", type=int, default=20)
parser.add_argument("--print_every", type=int, default = 200)
parser.add_argument("--plot_every", type=int, default = 1)
parser.add_argument("--criterion", type=str, default="L1Loss")
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--down_sample", type=float, default=0.0, help="Keep this fraction of the training data")
# parser.add_argument("--data_dir", type=str, default="./data/reformattedTraffic/")
parser.add_argument("--model", type=str, default="sketch-rnn")
parser.add_argument("--lambda_l1", type=float, default=2e-5)
parser.add_argument("--lambda_l2", type=float, default=5e-4)
parser.add_argument("--no_schedule_sampling", action="store_true", default=False)
parser.add_argument("--scheduling_start", type=float, default=1.0)
parser.add_argument("--scheduling_end", type=float, default=0.0)
parser.add_argument("--tries", type=int, default=12)
parser.add_argument("--kld_warmup_until", type=int, default=5)
parser.add_argument("--kld_weight_max", type=float, default=0.10)
parser.add_argument("--no_shuffle_after_epoch", action="store_true", default=False)
parser.add_argument("--clip", type=int, default=10)
parser.add_argument("--dataset", type=str, default="traffic")
parser.add_argument("--predictOnTest", action="store_true", default=True)
parser.add_argument("--encoder_input_dropout", type=float, default=0.5)
parser.add_argument("--encoder_layer_dropout", type=float, default=0.5)
parser.add_argument("--decoder_input_dropout", type=float, default=0.5)
parser.add_argument("--decoder_layer_dropout", type=float, default=0.5)
parser.add_argument("--noEarlyStopping", action="store_true", default=False)
parser.add_argument("--earlyStoppingPatients", type=int, default=3)
parser.add_argument("--earlyStoppingMinDelta", type=float, default=0.0001)
parser.add_argument("--bidirectionalEncoder", type=bool, default=True)
def trainF(data = None, suggestions=None):
    experimentData = {}
    args = parser.parse_args()
    if not suggestions:
        saveDir = '../save/models/model0/'
        while os.path.isdir(saveDir):
            numStart = saveDir.rfind("model")+5
            numEnd = saveDir.rfind("/")
            saveDir = saveDir[:numStart] + str(int(saveDir[numStart:numEnd])+1) + "/"
        os.mkdir(saveDir)
        args.save_dir = saveDir
    if suggestions:
        args.model = suggestions["model"]
        args.save_dir = suggestions["save_dir"]
        args.h_dim = int(suggestions["h_dim"])
        args.initial_lr = float(suggestions["initial_lr"])
        args.batch_size = int(suggestions["batch_size"])
        args.lambda_l1 = float(suggestions["lambda_l1"])
        args.lambda_l2 = float(suggestions["lambda_l2"])
        args.n_layers = int(suggestions["n_layers"])
        args.encoder_layer_dropout = suggestions["encoder_layer_dropout"]
        args.encoder_input_dropout = suggestions["encoder_input_dropout"]
        args.decoder_layer_dropout = suggestions["decoder_layer_dropout"]
        args.decoder_input_dropout = suggestions["decoder_input_dropout"]
        # if args.predictOnTest:
        #     cats = ["train", "val", "test"]
        # else:
        #     cats = ["train", "val"]
        # for cat in cats:
        #     shuffle=False
        #     if cat == "train":
        #         shuffle=True
        #     if args.dataset == "traffic":
        #         data["{}_loader".format(cat)] = data['{}_loader'.format(cat)] = utils.DataLoaderWithTime(
        #             data['x_{}'.format(cat)],
        #             data['y_{}'.format(cat)],
        #             data["x_times_{}".format(cat)],
        #             data["y_times_{}".format(cat)],
        #             args.batch_size, shuffle=shuffle)
        #     else:
        #         data["{}_loader".format(cat)] = data['{}_loader'.format(cat)] = utils.DataLoader(
        #             data['x_{}'.format(cat)],
        #             data['y_{}'.format(cat)],
        #             args.batch_size, shuffle=shuffle)
    if data is None:
        print("loading data")
        if args.dataset == "traffic":
            dataDir = "/home/dan/data/traffic/trafficWithTime/"
            data = utils.load_traffic_dataset(dataDir, args.batch_size, down_sample=args.down_sample, load_test=args.predictOnTest)
        elif args.dataset == "human":
            dataDir = "/home/dan/data/human/Processed/"
            data = utils.load_human_dataset(dataDir, args.batch_size, down_sample=args.down_sample, load_test=args.predictOnTest)
    print("setting additional params")
    # Set additional arguments
    if args.model == "sketch-rnn":
        assert args.kld_warmup_until <= args.n_epochs, "KLD Warm up stop > n_epochs"
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args._device = "cuda" if args.cuda else "cpu"
    args.use_attn = not args.no_attn
    args.x_dim = data['x_dim']
    args.sequence_len = data['sequence_len']
    args.channels = data["channels"]
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
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    lr_decay_milestones = np.arange(args.lr_decay_beginning, args.n_epochs, args.lr_decay_every)
    scheduler = MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=args.lr_decay_factor)
    print("beginning training")
    bestLoss = None
    for epoch in range(1, args.n_epochs + 1):
        scheduler.step()
        print("epoch {}".format(epoch))
        kldLossWeight = args.kld_weight_max * min((epoch / (args.kld_warmup_until)), 1.0)
        avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss = \
                    utils.train(data['train_loader'].get_iterator(), data['val_loader'].get_iterator(),\
                                model, args, data, epoch, optimizer, kldLossWeight)
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
            modelWeights = model.cpu().state_dict()
            torch.save(modelWeights, fn)
            print('Saved model to '+fn)
            if args.cuda:
                model = model.cuda()
        if not args.noEarlyStopping:
            mostRecentLoss = experimentData["valReconLosses"][-1] + experimentData["valKLDLosses"][-1]
            if bestLoss is not None and mostRecentLoss + args.earlyStoppingMinDelta >= bestLoss:
                earlyStoppingCounter += 1
                if earlyStoppingCounter >= args.earlyStoppingPatients:
                    print("early stopping: stopping training")
                    break
            else:
                bestLoss = mostRecentLoss
                earlyStoppingCounter = 0

        if not args.no_shuffle_after_epoch:
            # Shuffle training examples for next epoch
            data['train_loader'].shuffle()
    if args.model == "sketch-rnn":
        utils.plotTrainValCurve(experimentData["trainReconLosses"], experimentData["valReconLosses"],\
            args.model, args.criterion, args, trainKLDLosses=experimentData["trainKLDLosses"],\
            valKLDLosses=experimentData["valKLDLosses"])
    elif args.model == "rnn":
        utils.plotTrainValCurve(experimentData["trainReconLosses"], experimentData["valReconLosses"],\
            args.model, args.criterion, args)
    model_fn = args.save_dir + '{}_full_model'.format(args.model) +".pth"
    # experimentData_fn = args.save_dir + "experimentData.pth"
    torch.save(model.cpu().state_dict(), model_fn)
    # np.save(experimentData, experimentData_fn)
    torch.save(experimentData["learningRates"], args.save_dir + "learningRates.pth")
    del model
    # del data
    ret = experimentData["trainReconLosses"][-1], experimentData["trainKLDLosses"][-1], experimentData["valReconLosses"][-1], experimentData["valKLDLosses"][-1], args.save_dir
    del experimentData
    torch.cuda.empty_cache()
    return ret

if __name__ == '__main__':
    trainF()
