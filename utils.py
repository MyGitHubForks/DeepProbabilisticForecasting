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

def normalizeData(data):
    return (data - np.mean(data))/np.std(data), np.mean(data), np.std(data)

def load_dataset(dataset_dir, batch_size, down_sample=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        print(category)
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if down_sample and category=="train":
            nRows = cat_data['x'].shape[0]
            down_sampled_rows = np.random.choice(range(nRows), size=np.ceil(nRows * down_sample).astype(int), replace=False)
            data['x_' + category] = cat_data['x'][down_sampled_rows,:,:,0]
            data['y_' + category] = cat_data['y'][down_sampled_rows,:,:,0]
        else:
            data['x_' + category] = cat_data['x'][:,:,:,0]
            data['y_' + category] = cat_data['y'][:,:,:,0]
        data["x_"+category], data["x_"+category+"_mean"], data["x_"+category+"_std"] = normalizeData(data["x_"+category])
        data["y_"+category], data["y_"+category+"_mean"], data["y_"+category+"_std"] = normalizeData(data["y_"+category])
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

def plotTrainValCurve(trainLosses, valLosses, model_description, lossDescription, args):
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    fig, ax = plt.subplots()
    plt.xlabel("Epoch")
    plt.ylabel(lossDescription)
    plt.plot(np.arange(1, len(trainLosses)+1)*args.plot_every, trainLosses, color="red", label="train loss")
    plt.plot(np.arange(1, len(valLosses)+1)*args.plot_every, valLosses, color="blue", label="validation loss")
    plt.grid()
    plt.legend()
    plt.title("Losses for {}".format(model_description))
    plt.savefig(args.save_dir+"train_val_loss_plot.png")

#Loss functions for VRNN

#computing losses
#kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
# nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

def unNormalize(val, mean, std):
    return (val *std)+mean

def getPredictions(args, data_loader, model, xMean, xStd, yMean, yStd):
    targets = []
    preds = []
    datas = []
    means = []
    stds = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
        targets.append(unNormalize(target.detach(), yMean,yStd))
        datas.append(unNormalize(data.detach(), xMean, xStd))

        modelOutput = model(data)
        if args.model == "vrnn":
            all_enc_mean, all_enc_std, all_dec_mean, all_dec_std, all_prior_mean, all_prior_std = modelOutput
            decoder_means_mat = torch.cat([torch.unsqueeze(y, dim=0) for y in all_dec_mean])
            decoder_std_mat = torch.cat([torch.unsqueeze(y, dim=0) for y in all_dec_std])
            means.append(unNormalize(decoder_means_mat, yMean, yStd))
            stds.append(unNormalize(decoder_std_mat, yMean, yStd))
        elif args.model="rnn":
            output = modelOutput
            preds.append(unNormalize(output.detach(), yMean, yStd))
        else:
            assert False, "can't match model"  
    return preds, targets, datas, means, stds

def kld_gauss(mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return  0.5 * torch.sum(kld_element)

def train(train_loader, val_loader, model, lr, args, dataDict):
    clip = 10
    train_loss = 0.0
    val_loss = 0.0
    start = time.time()

    # Define Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Train
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        optimizer.zero_grad()
        output = model(data)
        #print("output", output[0,0,:5])
        #print("target", target[0,0,:5])
        if args.model == "vrnn":
            encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds = output
            # Calculate KLDivergence part
            loss = 0.0
            for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t in zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds):
                kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                loss += kldLoss

            #Calculate Prediction Loss
            pred = torch.cat([torch.unsqueeze(y, dim=0) for y in decoder_means])
            unNPred = unNormalize(pred.detach(), dataDict["y_train_mean"], dataDict["y_train_std"])
            unNTarget = unNormalize(target.detach(), dataDict["y_train_mean"], dataDict["y_train_std"])
            if args.criterion == "RMSE":
                predLoss = torch.sqrt(torch.mean((pred - target)**2))    
                unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))
                loss += predLoss

            elif args.criterion == "L1Loss":
                predLoss = torch.mean(torch.abs(pred - target))
                unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
                loss += predLoss


        elif args.criterion == "RMSE":
                o = unNormalize(output, dataDict["y_train_mean"], dataDict["y_train_std"])
                t = unNormalize(target, dataDict["y_train_mean"], dataDict["y_train_std"])
                loss = torch.sqrt(torch.mean((o - t)**2))

        elif args.criterion == "L1Loss":
            o = unNormalize(output, dataDict["y_train_mean"], dataDict["y_train_std"])
            t = unNormalize(target, dataDict["y_train_mean"], dataDict["y_train_std"])
            loss = torch.mean(torch.abs(o - t))
        else:
            assert False, "bad loss function"
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if args.model == "vrnn":
            bLoss = unNormalizedLoss.data.item()
        else:
            bLoss = loss.data.item()

        if batch_idx % args.print_every == 0:
             print("batch index: {}, loss: {}".format(batch_idx, bLoss))
        train_loss += bLoss
    nTrainBatches = batch_idx + 1
    # Validate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
            target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
            output = model(data)
            if args.model == "vrnn":
                encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds = output
                pred = torch.cat([torch.unsqueeze(y, dim=0) for y in decoder_means])
                unNPred = unNormalize(pred.detach(), dataDict["y_val_mean"], dataDict["y_val_std"])
                unNTarget = unNormalize(target.detach(), dataDict["y_val_mean"], dataDict["y_val_std"])
                if args.criterion == "RMSE":
                    predLoss = torch.sqrt(torch.mean((pred - target)**2))    
                    unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)))
                    loss = predLoss

                elif args.criterion == "L1Loss":
                    predLoss = torch.mean(torch.abs(pred - target))
                    unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
                    loss = predLoss

            elif args.criterion == "RMSE":
                o = unNormalize(output, dataDict["y_val_mean"], dataDict["y_val_std"])
                t = unNormalize(target, dataDict["y_val_mean"], dataDict["y_val_std"])
                loss = torch.sqrt(torch.mean((o - t)**2))

            elif args.criterion == "L1Loss":
                o = unNormalize(output, dataDict["y_val_mean"], dataDict["y_val_std"])
                t = unNormalize(target, dataDict["y_val_mean"], dataDict["y_val_std"])
                loss = torch.mean(torch.abs(o - t))
            else:
                assert False, "bad loss function"

            if args.model == "vrnn":
                val_loss += unNormalizedLoss.item()
            else:
                val_loss += loss.item()
    nValBatches = batch_idx + 1
    avgTrainLoss = train_loss / nTrainBatches
    avgValLoss = val_loss / nValBatches
    print('====> Average Train Loss: {} Average Val Loss: {}'.format(avgTrainLoss, avgValLoss))
    return avgTrainLoss, avgValLoss
