import torch
import torch.nn as nn
from torch import optim
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from model.Data import DataLoader
from memory_profiler import profile

def normalizeData(x, y):
    allData = np.stack((x,y), axis=1)
    mean = np.mean(allData)
    std = np.std(allData)
    return (x-mean)/std, (y-mean)/std, mean, std

def load_dataset(dataset_dir, batch_size, down_sample=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        print(category)
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if down_sample:
            nRows = cat_data['x'].shape[0]
            down_sampled_rows = np.random.choice(range(nRows), size=np.ceil(nRows * down_sample).astype(int),
                                                 replace=False)
            data['x_' + category] = cat_data['x'][down_sampled_rows, :, :, 0]
            data['y_' + category] = cat_data['y'][down_sampled_rows, :, :, 0]
        else:
            data['x_' + category] = cat_data['x'][:,:,:,0]
            data['y_' + category] = cat_data['y'][:,:,:,0]
        data["x_"+category], data["y_"+category], data[category+"_mean"], data[category+"_std"] =\
         normalizeData(data["x_"+category], data["y_"+category])
    data['sequence_len'] = cat_data['x'].shape[1]
    data['x_dim'] = cat_data['x'].shape[2]
    assert data['sequence_len'] == 12
    assert data['x_dim'] == 207
    # Data format
    for category in ['train', 'val', 'test']:
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)

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
    plt.rcParams.update({'font.size': 8})
    plt.figure()
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


# Loss functions for VRNN

# computing losses
# kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
# nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
# nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

def unNormalize(val, mean, std):
    return (val * std) + mean


def getPredictions(args, data_loader, model, mean, std):
    targets = []
    preds = []
    datas = []
    means = []
    stds = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
            target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
            targets.append(unNormalize(target, mean, std))
            datas.append(unNormalize(data, mean, std))
            modelOutput = model(data, target, 0, noSample=True)
            del target
            del data
            if args.model == "vrnn":
                all_enc_mean, all_enc_std, all_dec_mean, all_dec_std, all_prior_mean, all_prior_std, all_samples = modelOutput
                del all_enc_mean
                del all_enc_std
                del all_prior_mean
                del all_prior_std
                del all_samples
                decoder_means_mat = np.concatenate([torch.unsqueeze(y, dim=0).cpu().data.numpy()\
                                                    for y in all_dec_mean], axis=0)
                decoder_std_mat = np.concatenate([torch.unsqueeze(y, dim=0).cpu().data.numpy()\
                                                    for y in all_dec_std], axis=0)
                means.append(decoder_means_mat)
                stds.append(decoder_std_mat)
            elif args.model=="rnn":
                output = modelOutput.cpu().detach()
                preds.append(unNormalize(output, mean, std))
            else:
                assert False, "can't match model"
        return preds, targets, datas, means, stds


def kld_gauss(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""

    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element)


def runBatch(data, target, optimizer, model, args, epoch):
    optimizer.zero_grad()
    output = model(data, target, epoch)
    return output


def getVRNNLoss(output, target, dataDict, args):
    encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
    # Calculate KLDivergence part
    loss = 0.0
    for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in \
            zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
        kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        loss += kldLoss
    # Calculate Prediction Loss
    pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
    unNPred = unNormalize(pred.detach(), dataDict["y_train_mean"], dataDict["y_train_std"])
    unNTarget = unNormalize(target.detach(), dataDict["y_train_mean"], dataDict["y_train_std"])
    if args.criterion == "RMSE":
        predLoss = torch.sqrt(torch.mean((pred - target) ** 2))
        unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))
        loss += predLoss

    elif args.criterion == "L1Loss":
        predLoss = torch.mean(torch.abs(pred - target))
        unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
        loss += predLoss
    return loss, unNormalizedLoss


def getRNNLoss(output, target, dataDict, args):
    if args.criterion == "RMSE":
        o = unNormalize(output, dataDict["y_train_mean"], dataDict["y_train_std"])
        t = unNormalize(target, dataDict["y_train_mean"], dataDict["y_train_std"])
        loss = torch.sqrt(torch.mean((o - t) ** 2))
    elif args.criterion == "L1Loss":
        o = unNormalize(output, dataDict["y_train_mean"], dataDict["y_train_std"])
        t = unNormalize(target, dataDict["y_train_mean"], dataDict["y_train_std"])
        loss = torch.mean(torch.abs(o - t))
    else:
        assert False, "bad loss function"
    return loss


def backProp(output, target, dataDict, args, optimizer, model, clip):
    if args.model == "vrnn":
        loss, unNormalizedLoss = getVRNNLoss(output, target, dataDict, args)
    else:
        loss = getRNNLoss(output, target, dataDict, args)
    loss.backward()
    optimizer.step()
    # grad norm clipping, only in pytorch version >= 1.10
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    if args.model == "vrnn":
        bLoss = unNormalizedLoss.data.item()
    else:
        bLoss = loss.data.item()
    del loss
    return bLoss


def runValBatch(data, target, args, model):
    output = model(data, target, 0, noSample=True)
    return output


def getValLoss(args, dataDict, target, output):
    if args.model == "vrnn":
        encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
        pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
        unNPred = unNormalize(pred.detach(), dataDict["y_val_mean"], dataDict["y_val_std"])
        unNTarget = unNormalize(target.detach(), dataDict["y_val_mean"], dataDict["y_val_std"])
        if args.criterion == "RMSE":
            predLoss = torch.sqrt(torch.mean((pred - target) ** 2))
            unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))
            loss = predLoss

        elif args.criterion == "L1Loss":
            predLoss = torch.mean(torch.abs(pred - target))
            unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
            loss = predLoss

    elif args.criterion == "RMSE":
        o = unNormalize(output, dataDict["y_val_mean"], dataDict["y_val_std"])
        t = unNormalize(target, dataDict["y_val_mean"], dataDict["y_val_std"])
        loss = torch.sqrt(torch.mean((o - t) ** 2))

    elif args.criterion == "L1Loss":
        o = unNormalize(output, dataDict["y_val_mean"], dataDict["y_val_std"])
        t = unNormalize(target, dataDict["y_val_mean"], dataDict["y_val_std"])
        loss = torch.mean(torch.abs(o - t))
    else:
        assert False, "bad loss function"

    if args.model == "vrnn":
        return unNormalizedLoss.item()
    else:
        return loss.item()

def train(train_loader, val_loader, model, lr, args, dataDict, epoch):
    clip = 10
    train_kld_loss = 0.0
    train_recon_loss = 0.0
    val_recon_loss = 0.0
    val_kld_loss = 0.0
    start = time.time()

    # Define Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Train
    nTrainBatches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        optimizer.zero_grad()
        output = model(data, target, epoch)
        del data
        if args.model == "vrnn":
            encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
            # Calculate KLDivergence part
            totalKLDLoss = 0.0
            for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
                kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                totalKLDLoss += kldLoss

            #Calculate Prediction Loss
            pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
            unNPred = unNormalize(pred.detach(), dataDict["train_mean"], dataDict["train_std"])
            unNTarget = unNormalize(target.detach(), dataDict["train_mean"], dataDict["train_std"])
            assert pred.size() == target.size()
            if args.criterion == "RMSE":
                predLoss = torch.sqrt(torch.mean((pred - target)**2))    
                unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))

            elif args.criterion == "L1Loss":
                predLoss = torch.mean(torch.abs(pred - target))
                unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
            loss = totalKLDLoss + predLoss

        elif args.criterion == "RMSE":
            o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
            t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
            loss = torch.sqrt(torch.mean((o - t)**2))

        elif args.criterion == "L1Loss":
            o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
            t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
            loss = torch.mean(torch.abs(o - t))
        else:
            assert False, "bad loss function"
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if args.model == "vrnn":
            bReconLoss = unNormalizedLoss.data.item()
            bKLDLoss = totalKLDLoss.data.item()
            if batch_idx % args.print_every == 0:
                print("batch index: {}, recon loss: {}, kld loss: {}".format(batch_idx, bReconLoss, bReconLoss))
            train_recon_loss += bReconLoss
            train_kld_loss += bKLDLoss
        else:
            bLoss = loss.data.item()
            train_recon_loss += bLoss
            if batch_idx % args.print_every == 0:
                print("batch index: {}, loss: {}".format(batch_idx, bLoss))

    nTrainBatches = batch_idx + 1
    # Validate
    nValBatches = 0
    with torch.no_grad():
        nValBatches += 1
        for batch_idx, (data, target) in enumerate(val_loader):
            data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
            target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
            output = model(data, target, 0, noSample=True)
            del data
            if args.model == "vrnn":
                encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
                pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
                unNPred = unNormalize(pred.detach(), dataDict["val_mean"], dataDict["val_std"])
                unNTarget = unNormalize(target.detach(), dataDict["val_mean"], dataDict["val_std"])
                totalKLDLoss = 0.0
                for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
                    kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                    totalKLDLoss += kldLoss
                if args.criterion == "RMSE":
                    predLoss = torch.sqrt(torch.mean((pred - target)**2))    
                    unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))

                elif args.criterion == "L1Loss":
                    predLoss = torch.mean(torch.abs(pred - target))
                    unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))

            elif args.criterion == "RMSE":
                o = unNormalize(output, dataDict["val_mean"], dataDict["val_std"])
                t = unNormalize(target, dataDict["val_mean"], dataDict["val_std"])
                loss = torch.sqrt(torch.mean((o - t)**2))

            elif args.criterion == "L1Loss":
                o = unNormalize(output, dataDict["val_mean"], dataDict["val_std"])
                t = unNormalize(target, dataDict["val_mean"], dataDict["val_std"])
                loss = torch.mean(torch.abs(o - t))
            else:
                assert False, "bad loss function"

            if args.model == "vrnn":
                val_recon_loss += unNormalizedLoss.data.item()
                val_kld_loss += totalKLDLoss.data.item()
            else:
                val_recon_loss += loss.data.item()
    nValBatches = batch_idx + 1
    avgTrainReconLoss = train_recon_loss / nTrainBatches
    avgTrainKLDLoss = train_kld_loss / nTrainBatches
    avgValReconLoss = val_recon_loss / nValBatches
    avgValKLDLoss = val_kld_loss / nValBatches
    if args.model == "vrnn":
        print('====> Average Train Recon Loss: {} Average Train KLD Loss: {} Average Val Recon Loss: {} Average Val KLD Loss: {}'.format(avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss))
    elif args.model == "rnn":
        print("===> Average Train Loss: {} Average Val Loss: {}".format(avgTrainReconLoss, avgValReconLoss))
    else:
        assert False
    return avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss
