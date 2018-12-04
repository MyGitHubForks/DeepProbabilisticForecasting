import logging, sys
import torch
import h5py
import os
import numpy as np
import torch.utils.data as torchUtils
import torch.optim as optim
from functools import partial
import torch.nn as nn
import json
from shutil import copy2, copyfile, copytree
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
parser.add_argument("--criterion", type=str, default="L1Loss")
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--down_sample", type=float, default=0.0, help="Keep this fraction of the training data")
# parser.add_argument("--data_dir", type=str, default="./data/reformattedTraffic/")
parser.add_argument("--model", type=str, default="sketch-rnn")
parser.add_argument("--lambda_l1", type=float, default=0)
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
parser.add_argument("--local", action="store_true", default=False)
args = parser.parse_args()
logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)

def plotLosses(trainLosses, valLosses, trainKLDLosses=None, valKLDLosses=None):
    torch.save(trainLosses, args.save_dir+"plot_train_recon_losses")
    torch.save(valLosses, args.save_dir+"plot_val_recon_losses")
    if trainKLDLosses and valKLDLosses:
        torch.save(trainKLDLosses, args.save_dir+"plot_train_KLD_losses")
        torch.save(valKLDLosses, args.save_dir+"plot_val_KLD_losses")
    plt.rcParams.update({'font.size': 8})
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(args.criterion, color="r")
    ax1.tick_params('y', colors='r')
    ax1.plot(np.arange(1, len(trainLosses)+1), trainLosses, "r--", label="train reconstruction loss")
    ax1.plot(np.arange(1, len(valLosses)+1), valLosses, color="red", label="validation reconstruction loss")
    ax1.legend(loc="upper left")
    ax1.grid()
    plt.title("Losses for {}".format(args.model))
    plt.savefig(args.save_dir + "train_val_loss_plot.png")

def getSaveDir():
    saveDir = '../save/models/model0/'
    while os.path.isdir(saveDir):
        numStart = saveDir.rfind("model")+5
        numEnd = saveDir.rfind("/")
        saveDir = saveDir[:numStart] + str(int(saveDir[numStart:numEnd])+1) + "/"
    os.mkdir(saveDir)
    return saveDir

def saveUsefulData():
    argsFile = args.save_dir + "args.txt"
    with open(argsFile, "w") as f:
        f.write(json.dumps(vars(args)))
    copy2("./train.py", args.save_dir+"train.py")
    copy2("./utils.py", args.save_dir+"utils.py")
    copy2("./gridSearchOptimize.py", args.save_dir+"gridsearchOptimize.py")
    copytree("./model", args.save_dir+"model/")

def getTrafficDataset(dataDir, category):
    f = np.load(os.path.join(dataDir, category + '.npz'))
    my_dataset = torchUtils.TensorDataset(torch.Tensor(f["inputs"]),torch.Tensor(f["targets"])) # create your datset
    scaler = getScaler(f["inputs"])
    sequence_len = f['inputs'].shape[1]
    x_dim = f['inputs'].shape[2]
    channels = f["inputs"].shape[3]
    return my_dataset, scaler, sequence_len, sequence_len, x_dim, channels

def getHumanDataset(dataDir, category):
    f = h5py.File(os.path.join(dataDir, category+".h5"))
    my_dataset = torchUtils.TensorDataset(torch.Tensor(f["input2d"]), torch.Tensor(f["target2d"]))
    scaler = getScaler(f["input2d"])
    input_sequence_len = f["input2d"].shape[1]
    target_sequence_len = f["target2d"].shape[1]
    x_dim = f["input2d"].shape[2]
    channels = f["input2d"].shape[3]
    return my_dataset, scaler, input_sequence_len, target_sequence_len, x_dim, channels

def getLoaderAndScaler(dataDir, category):
    logging.info("Getting {} loader".format(category))
    if args.dataset == "traffic":
        my_dataset, scaler, input_sequence_len, target_sequence_len, x_dim, channels = getTrafficDataset(dataDir, category)
    else:
        my_dataset, scaler, input_sequence_len, target_sequence_len, x_dim, channels = getHumanDataset(dataDir, category)
    shf = False
    if category == "train":
        shf = True
    loader = torchUtils.DataLoader(
        my_dataset,
        batch_size=args.batch_size,
        shuffle=shf,
        num_workers=0,
        pin_memory=False,
        drop_last=True
        )
    return loader, scaler, input_sequence_len, target_sequence_len, x_dim, channels # create your dataloader

def getDataLoaders(dataDir, debug=False):
    loaders = {}
    logging.info("Getting data from {}".format(dataDir))
    if debug:
        categories = ["test"]
        scalerSet = "test"
    else:
        categories = ["train", "val", "test"]
        scalerSet = "train"
    for category in categories:
        loader, scaler, input_sequence_len, target_sequence_len, x_dim, channels = getLoaderAndScaler(dataDir, category)
        if category == scalerSet:
            loaders["scaler"] = scaler
            loaders["input_sequence_len"] = input_sequence_len
            loaders["target_sequence_len"] = target_sequence_len
            loaders["x_dim"] = x_dim
            loaders["channels"] = channels
        loaders[category] = loader
    return loaders

def transformBatch(batch, scaler=None):
    x = scaler.transform(batch[0]).permute(1,0,3,2)
    y = scaler.transform(batch[1])[...,0].permute(1,0,2)
    if args.cuda:
        return x.cuda(), y.cuda()
    return x, y

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean0, std0, mean1, std1):
        self.mean0 = mean0
        self.std0 = std0
        self.mean1 = mean1
        self.std1 = std1

    def transform(self, data):
        mean = torch.zeros(data.size())
        mean[...,0] = self.mean0
        mean[...,1] = self.mean1
        std = torch.ones(data.size())
        std[...,0] = self.std0
        std[...,1] = self.std1
        return torch.div(torch.sub(data,mean),std)

class StandardScalerTraffic(StandardScaler):
    def __init__(self, mean0, std0):
        StandardScaler.__init__(self, mean0, std0, 0.0, 1.0)

    def inverse_transform(self, data):
        """
        Inverse transform is applied to output and target.
        These are only the speeds, so only use the first 
        """
        mean = torch.ones(data.size()) * self.mean0
        std = torch.ones(data.size()) * self.std0
        if args.cuda:
            mean = mean.cuda()
            std = std.cuda()
        transformed = torch.add(torch.mul(data, std), mean)
        del mean, std
        return transformed.permute(1,0,2)

    def transformBatchForEpoch(self, batch):
        x = self.transform(batch[0]).permute(1,0,3,2)
        y = self.transform(batch[1])[...,0].permute(1,0,2)
        if args.cuda:
            return x.cuda(), y.cuda()
        return x, y

class StandardScalerHuman(StandardScaler):
    """docstring for StandardScalerHuman"""
    def __init__(self, mean0, std0, mean1, std1):
        super().__init__(self, mean0, std0, mean1, std1)
    
    def transform(self, data):
        transormed = super().transorm(data)
        layer0, layer1 = torch.split(transormed, 1, dim=3)
        return torch.cat((layer0.squeeze(3), layer1.squeeze(3)), dim=2)

    def inverse_transform(self, data):
        """
        applied to output and target
        """
        l1, l2 = torch.split(data, data.size(2) / 2, 2)
        transed = torch.cat((l1.unsqueeze(3), l2.unsqueeze(3)), dim=3)
        mean = torch.zeros(transed.size())
        mean[...,0] = self.mean0
        mean[...,1] = self.mean1
        std = torch.ones(transed.size())
        std[...,0] = self.std0
        std[...,1] = self.std1
        transformed =  torch.add(torch.mul(transed, std), mean)
        return transformed.permute(1,0,3,2)

    def transformBatchForEpoch(self, batch):
        x = self.transform(batch[0]).permute(1,0,3,2)
        y = self.transform(batch[1]).permute(1,0,3,2)
        if args.cuda:
            return x.cuda(), y.cuda()
        return x, y

def getScaler(trainX):
    mean0 = np.mean(trainX[...,0])
    std0 = np.std(trainX[...,0])
    mean1 = np.mean(trainX[...,1])
    std1 = np.std(trainX[...,1])
    if args.dataset == "traffic":
        return StandardScalerTraffic(mean0, std0)
    elif args.dataset == "human":
        return StandardScalerHuman(mean0, std0, mean1, std1)
    else:
        assert False, "bad dataset"

def getReconLoss(output, target, scaler):
    output = scaler.inverse_transform(output)
    target = scaler.inverse_transform(target)
    assert output.size() == target.size(), "output size: {}, target size: {}".format(output.size(), target.size())
    if args.criterion == "RMSE":
        criterion = nn.MSELoss()
        return torch.sqrt(criterion(output, target))
    elif args.criterion == "L1Loss":
        criterion = nn.L1Loss()
        return criterion(output, target)
    else:
        assert False, "bad loss function"

def getLoss(model, output, target, scaler):
    reconLoss = getReconLoss(output, target, scaler)
    return reconLoss

def saveModel(modelWeights, epoch):
    fn = args.save_dir+'{}_state_dict_'.format(args.model)+str(epoch)+'.pth'
    torch.save(modelWeights, fn)
    logging.info('Saved model to '+fn)

class EarlyStoppingObject(object):
    """docstring for EarlyStoppingObject"""
    def __init__(self):
        super(EarlyStoppingObject, self).__init__()
        self.bestLoss = None
        self.bestEpoch = None
        self.counter = 0
        self.epochCounter = 0

    def checkStop(self, previousLoss):
        self.epochCounter += 1
        if not args.noEarlyStopping:
            if self.bestLoss is not None and previousLoss + args.earlyStoppingMinDelta >= self.bestLoss:
                self.counter += 1
                if self.counter >= args.earlyStoppingPatients:
                    logging.info("Stopping Early, haven't beaten best loss {:.4f} @ Epoch {} in {} epochs".format(
                        self.bestLoss,
                        self.bestEpoch,
                        args.earlyStoppingPatients))
                    return True
            else:
                self.bestLoss = previousLoss
                self.bestEpoch = self.epochCounter
                self.counter = 0
                return False

        else:
            return False
