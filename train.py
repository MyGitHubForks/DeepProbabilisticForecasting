import numpy as np
import cProfile
import model.Data as DF
import model.EncoderRNN as ERF
import model.AttnDecoder as ADRF
import utils
import pstats
import torch
from torch.utils import data
from model.Data import Dataset
from model.RoseSeq2Seq import Seq2Seq
import os

parser = argparse.ArgumentParser(description='Batched Sequence to Sequence')
parser.add_argument('--h_dim', type=int, default=64)
parser.add_argument('--no_cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument("--no_attn", action="store_true", default=False, help="Do not use AttnDecoder")
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batches_per_epoch", type=int, default= -1)
parser.add_argument("--batch_size", type=int, default= 40)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--initial_lr", type=float, default=1e-2)
parer.add_argument("--lr_decay_ratio", type=float, default=0.10)
parser.add_argument("--lr_decay_beginning", type=int, default=20)
parser.add_argument("--lr_decay_every", type=int, default=10)
parser.add_argument("--print_every", type=int, default = 1000)
parser.add_argument("--plot_every", type=int, default = 100)
parser.add_argument("--criterion", type=string, default="L1 Loss")
parser.add_argument("--save_freq", type=int, default=)
def main():
    saveDir = './save/'
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    args = parser.parse_args()

    print("loading data")
    # Get data objects
    trainData = Dataset("./data/train.npz")
    valData = Dataset("./data/val.npz")
    trainLoader = data.DataLoader(trainData,
        batch_size=args["batch_size"], shuffle=True)
    valLoader = data.DataLoader(valData,
        batch_size=args["batch_size"], shuffle=True)

    # Get data constants
    xExample, yExample = trainData.__getitem__(0)
    sequence_length = xExample.size(1)
    assert sequence_length == 12
    x_dim = xExample.size(2)

    # Set additional arguments
    args["cuda"] = not args.no_cuda and torch.cuda.is_available()
    args["use_attn"] = not no_attn
    args["x_dim"] = x_dim
    args["sequence_length"] = sequence_length
    if args["batches_per_epoch"] == -1:
        args["batches_per_epoch"] = np.ceil(trainData.__len__() / args["batch_size"])

    model = Seq2Seq(args)
    modelDescription = "Sequence to Sequence RNN with Attn"
    trainLosses = []
    valLosses = []
    for epoch in range(1, args.n_epochs + 1):
        avgTrainLoss, avgValLoss = train(train_loader, epoch, model, args)
        trainLosses.append(avgTrainLoss)
        valLosses.append(avgValLoss)
        #saving model
        fn = saveDir+'vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)

    plotTrainValCurve(trainLosses, valLosses, modelDescription, args["criterion"])

if __name__ == '__main__':
        cProfile.run("main()", "restats")
        p = pstats.Stats('restats')
        p.sort_stats("tottime").print_stats()
