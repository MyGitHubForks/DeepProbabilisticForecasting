import torch
import torch.nn as nn
from torch import optim
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import h5py
from model.Data import DataLoader, DataLoaderWithTime
from memory_profiler import profile

def memReport():
    print("~~~ Memory Report")
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def normalizeData(x, y, layer=None):
    allData = np.concatenate((x,y), axis=0)
    if layer is not None:
        mean = np.mean(allData[:,:,:,layer])
        std = np.std(allData[:,:,:,layer])
        m = np.zeros_like(x)
        s = np.ones_like(x)
        m[:,:,:,layer] = mean
        s[:,:,:,layer] = std
        return (x-m)/s, (y-m)/s, mean, std
    else:
        mean = np.mean(allData)
        std = np.std(allData)
        return (x-mean)/std, (y-mean)/std, mean, std
    

def load_human_dataset(dataset_dir, batch_size, down_sample=None, load_test=False, **kwargs):
    data = {}
    if load_test:
        cats = ["train", "val", "test"]
    else:
        cats = ["train", "val"]
    for category in cats:
        print(category)
        f = h5py.File(os.path.join(dataset_dir, category+".h5"), "r")
        nRows = f["input2d"].shape[0]
        if down_sample: 
            down_sampled_rows = np.random.choice(range(nRows), size=np.ceil(nRows * down_sample).astype(int),
                                                 replace=False)
            down_sampled_rows = sorted(down_sampled_rows)
        else:
            down_sampled_rows = range(nRows)
        data["x_"+category] = f["input2d"][down_sampled_rows,...]
        data["y_"+category] = f["target2d"][down_sampled_rows,...]
        data["action_"+category] = f["action"][down_sampled_rows,...]
        data["camera_"+category] = f["camera"][down_sampled_rows,...]
        data["inputId_"+category] = f["inputId"][down_sampled_rows,...]
        data["subaction_"+category] = f["subaction"][down_sampled_rows,...]
        data["subject_"+category] = f["subject"][down_sampled_rows,...]
        data["targetId_"+category] = f["targetId"][down_sampled_rows,...]
        data["x_"+category], data["y_"+category], data[category+"_mean"], data[category+"_std"] =\
         normalizeData(data["x_"+category], data["y_"+category])

    data['sequence_len'] = f['input2d'].shape[1]
    data['x_dim'] = f['input2d'].shape[2]
    data["channels"] = f["input2d"].shape[3]

    assert data['sequence_len'] == 12
    assert data['x_dim'] == 32
    # Data format
    for category in cats:
        data['{}_loader'.format(category)] = DataLoader(data['x_{}'.format(category)], data['y_{}'.format(category)], batch_size, shuffle=True)
    return data

def load_traffic_dataset(dataset_dir, batch_size, down_sample=None, load_test=False, **kwargs):
    data = {}
    if load_test:
        cats = ["train", "val", "test"]
    else:
        cats = ["train", "val"]
    for category in cats:
        print(category)
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if down_sample:
            nRows = cat_data['inputs'].shape[0]
            down_sampled_rows = np.random.choice(range(nRows), size=np.ceil(nRows * down_sample).astype(int),
                                                 replace=False)
            data['x_' + category] = cat_data['inputs'][down_sampled_rows, ...]
            data['y_' + category] = cat_data['targets'][down_sampled_rows, ...]
            data["x_times_"+category] = cat_data["inputTimes"][down_sampled_rows]
            data["y_times_"+category] = cat_data["targetTimes"][down_sampled_rows]
        else:
            data['x_' + category] = cat_data['inputs']
            data['y_' + category] = cat_data['targets']
            data["x_times_"+category] = cat_data["inputTimes"]
            data["y_times_"+category] = cat_data["targetTimes"]
        data["x_"+category], data["y_"+category], data[category+"_mean"], data[category+"_std"] =\
         normalizeData(data["x_"+category], data["y_"+category], layer=0)
    data['sequence_len'] = cat_data['inputs'].shape[1]
    data['x_dim'] = cat_data['inputs'].shape[2]
    data["channels"] = cat_data["inputs"].shape[3]
    assert data['sequence_len'] == 12
    assert data['x_dim'] == 207
    # Data format
    for category in cats:
        data['{}_loader'.format(category)] = DataLoaderWithTime(data['x_{}'.format(category)], data['y_{}'.format(category)], data["x_times_{}".format(category)], data["y_times_{}".format(category)], batch_size, shuffle=True)
    return data


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plotTrainValCurve(trainLosses, valLosses, model_description, lossDescription, args, trainKLDLosses=None, valKLDLosses=None):
    torch.save(trainLosses, args.save_dir+"plot_train_recon_losses")
    torch.save(valLosses, args.save_dir+"plot_val_recon_losses")
    if trainKLDLosses:
        torch.save(trainKLDLosses, args.save_dir+"plot_train_KLD_losses")
        torch.save(valKLDLosses, args.save_dir+"plot_val_KLD_losses")
    plt.rcParams.update({'font.size': 8})
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(lossDescription, color="r")
    ax1.tick_params('y', colors='r')
    ax1.plot(np.arange(1, len(trainLosses)+1)*args.plot_every, trainLosses, "r--", label="train reconstruction loss")
    ax1.plot(np.arange(1, len(valLosses)+1)*args.plot_every, valLosses, color="red", label="validation reconstruction loss")
    ax1.legend(loc="upper left")
    ax1.grid()
    if trainKLDLosses:
        ax2 = ax1.twinx()
        ax2.set_ylabel("KLD Loss", color="b")
        ax2.tick_params('y', colors='b')
        ax2.plot(np.arange(1, len(trainKLDLosses)+1)*args.plot_every, trainKLDLosses, "b--", label="train KLD loss")
        ax2.plot(np.arange(1, len(valKLDLosses)+1)*args.plot_every, valKLDLosses, color="blue", label="val KLD loss")
        ax2.legend(loc="upper right")
        ax2.grid()
    plt.title("Losses for {}".format(model_description))
    plt.savefig(args.save_dir + "train_val_loss_plot.png")
    if trainKLDLosses:
        totalTrainLoss = np.array(trainKLDLosses) + np.array(trainLosses)
        totalValLoss = np.array(valKLDLosses) + np.array(valLosses)
        plt.figure()
        plt.plot(np.arange(1, len(trainLosses)+1)*args.plot_every, totalTrainLoss, label="Total Train Loss")
        plt.plot(np.arange(1, len(valLosses)+1)*args.plot_every, totalValLoss, label="Total Validation Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(lossDescription)
        plt.savefig(args.save_dir + "total_train_val_loss.png")


def unNormalize(mat, mean, std):
    return (mat * std) + mean

def sketchRNNKLD(latentMean, latentStd):
    m2 = torch.zeros_like(latentMean)
    s2 = torch.ones_like(latentStd)
    return kld_gauss(latentMean, latentStd, m2, s2)
    #return torch.sum((-1 / (2*latentMean.size(1))) * (1 + latentStd - latentMean.pow(2) - torch.exp(latentStd)))

def kld_gauss(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""

    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element)

def getRNNLoss(output, target, mean, std, args):
    outputCopy = output.clone().cpu().detach().numpy()
    outputCopy = unNormalize(outputCopy, mean, std)
    targetCopy = target.clone().cpu().detach().numpy()
    targetCopy = unNormalize(targetCopy, mean, std)
    if args.criterion == "RMSE":
        loss = torch.sqrt(torch.mean((output - target) ** 2))
        unNLoss = np.sqrt(np.mean((outputCopy - targetCopy) ** 2))
    elif args.criterion == "L1Loss":
        loss = torch.mean(torch.abs(output - target))
        unNLoss = np.mean(np.abs(outputCopy - targetCopy))
    else:
        assert False, "bad loss function"
    del outputCopy, targetCopy
    return loss, unNLoss


def getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, datasetMean, datasetStd):
    # mean and std of shape (batch_size, z_dim)
    kld = sketchRNNKLD(latentMean, latentStd)
    # calculate reconstruction loss
    predCopy = predOut.clone().cpu().detach().numpy()
    targetCopy = target.clone().cpu().detach().numpy()
    unNPred = unNormalize(predCopy, datasetMean, datasetStd)
    unNTarget = unNormalize(targetCopy, datasetMean, datasetStd)
    if args.criterion == "RMSE":
        predLoss = torch.sqrt(torch.mean((predOut - target)**2))    
        unNormalizedLoss = np.sqrt(np.mean((unNPred - unNTarget)**2))
    elif args.criterion == "L1Loss":
        predLoss = torch.mean(torch.abs(predOut - target))
        unNormalizedLoss = np.mean(np.abs(unNPred - unNTarget))
    else:
        assert False, "bad loss function"
    del predCopy, targetCopy
    return kld, predLoss, unNormalizedLoss

def getValLoss(output, target, dataDict, args):
    if args.model == "sketch-rnn":
        latentMean, latentStd, z, predOut, predMeanOut, predStdOut = output
        kldLoss, predLoss, unNormalizedLoss = getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, dataDict["val_mean"], dataDict["val_std"])
        return kldLoss.item(), unNormalizedLoss
    elif args.model == "rnn":
        loss, unNormalizedLoss = getRNNLoss(output, target, dataDict["val_mean"], dataDict["val_std"], args)
        return 0.0, unNormalizedLoss
    else:
        assert False, "bad model"

def getRegularizationLosses(model):
    l2_reg = None
    l1_reg = None
    for W in model.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    return l1_reg, l2_reg

def train(train_loader, val_loader, model, args, dataDict, epoch, optimizer, kldLossWeight):
    epochKLDLossTrain = 0.0
    epochReconLossTrain = 0.0
    epochKLDLossVal = 0.0
    epochReconLossVal = 0.0
    nTrainBatches = 0
    nValBatches = 0
    for batch_idx, vals in enumerate(train_loader):
        if args.dataset == "traffic":
            data, target, dataTimes, targetTimes = vals
        else:
            data, target = vals
        nTrainBatches += 1
        data = torch.as_tensor(data, dtype=torch.float, device=args._device)
        target = torch.as_tensor(target, dtype=torch.float, device=args._device)
        optimizer.zero_grad()
        output = model(data, target, epoch, training=True)
        del data
        if args.model == "sketch-rnn":
            latentMean, latentStd, z, predOut, predMeanOut, predStdOut = output
            kldLoss, predLoss, unNormalizedLoss = getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, dataDict["train_mean"], dataDict["train_std"])
            loss = (kldLoss * kldLossWeight) + predLoss
            epochKLDLossTrain += kldLoss.item()
            epochReconLossTrain += unNormalizedLoss
            if batch_idx % args.print_every == 0:
                print("batch index: {}, recon loss: {}, kld loss: {}".format(batch_idx, unNormalizedLoss, kldLoss))
        elif args.model == "rnn":
            loss, unNormalizedLoss = getRNNLoss(output, target, dataDict["train_mean"], dataDict["train_std"], args) # unNormalized Loss
            l1_reg, l2_reg = getRegularizationLosses(model)
            loss = loss + args.l1_lambda * l1_reg + args.l2_lambda * l2_reg
            epochReconLossTrain += unNormalizedLoss
            if batch_idx % args.print_every == 0:
                print("batch_idx: {}, loss: {}".format(batch_idx, loss))
        else:
            assert False, "bad model"
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        torch.cuda.empty_cache()
    for batch_idx, vals in enumerate(val_loader):
        if args.dataset == "traffic":
            data, target, dataTimes, targetTimes = vals
        else:
            data, target = vals
        nValBatches += 1
        data = torch.as_tensor(data, dtype=torch.float, device=args._device)
        target = torch.as_tensor(target, dtype=torch.float, device=args._device)
        output = model(data, target, epoch, training=False)
        validationKldLoss, validationReconLoss = getValLoss(output, target, dataDict, args)
        epochKLDLossVal += validationKldLoss
        epochReconLossVal += validationReconLoss
        torch.cuda.empty_cache()
    avgTrainReconLoss = epochReconLossTrain / nTrainBatches
    avgTrainKLDLoss = epochKLDLossTrain / nTrainBatches
    avgValReconLoss = epochReconLossVal / nValBatches
    avgValKLDLoss = epochKLDLossVal / nValBatches
    if args.model == "sketch-rnn":
        print('====> Average Train Recon Loss: {} Average Train KLD Loss: {} Average Val Recon Loss: {} Average Val KLD Loss: {}'.format(avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss))
    elif args.model == "rnn":
        print("===> Average Train Loss: {} Average Val Loss: {}".format(avgTrainReconLoss, avgValReconLoss))
    else:
        assert False, "Bad model specified"
    return avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss
