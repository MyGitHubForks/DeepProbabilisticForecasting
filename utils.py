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


def normalizeData(x, y):
    allData = np.stack((x,y), axis=1)
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
        f = h5py.File(os.path.join(dataset_dir, category+".h5"))
        if down_sample:
            nRows = f["input2d"].shape[0]
            down_sampled_rows = np.random.choice(range(nRows), size=np.ceil(nRows * down_sample).astype(int),
                                                 replace=False)
        else:
            down_sampled_rows = range(nRows)
        data["x_"+category] = f["input2d"][down_sampled_rows,...]
        data["y_"+category] = f["targe2d"][down_sampled_rows,...]
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
         normalizeData(data["x_"+category], data["y_"+category])
    data['sequence_len'] = cat_data['inputs'].shape[1]
    data['x_dim'] = cat_data['inputs'].shape[2]

    assert data['sequence_len'] == 12
    assert data['x_dim'] == 207
    # Data format
    for category in cats:
        data['{}_loader'.format(category)] = DataLoader(data['x_{}'.format(category)], data['y_{}'.format(category)], data["x_times_{}".format(category)], data["y_times_{}".format(category)], batch_size, shuffle=True)
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


def unNormalize(val, mean, std):
    return (val * std) + mean

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

def getPredictions(args, data_loader, model, mean, std):
    targets = []
    preds = []
    datas = []
    means = []
    stds = []
    dataTimesArr = []
    targetTimesArr = []
    kldLossesArr = []
    zs = []
    meanKLDLosses = None
    with torch.no_grad():
        for batch_idx, (data, target, dataTimes, targetTimes) in enumerate(data_loader):
            data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
            target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
            targets.append(unNormalize(target, mean, std))
            datas.append(unNormalize(data, mean, std))
            dataTimesArr.append(dataTimes)
            targetTimesArr.append(targetTimes)
            modelOutput = model(data, target, 0, training=False)
            #del target
            #del data
            if args.model == "sketch-rnn":
                latentMean, latentStd, z, predOut, predMeanOut, predStdOut = modelOutput
                zs.append(z)
                kld = sketchRNNKLD(latentMean, latentStd)
                kldLossesArr.append([kld])
                pred_means_mat = np.concatenate([torch.unsqueeze(m, dim=0).cpu().data.numpy()\
                                                    for m in predMeanOut], axis=0)
                pred_std_mat = np.concatenate([torch.unsqueeze(s, dim=0).cpu().data.numpy()\
                                                    for s in predStdOut], axis=0)
                means.append(predMeanOut)
                stds.append(predStdOut)
            elif args.model == "vrnn":
                all_enc_mean, all_enc_std, all_dec_mean, all_dec_std, all_prior_mean, all_prior_std, all_samples = modelOutput
                kldLossArr = []
                for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in zip(all_enc_mean, all_enc_std, all_dec_mean, all_dec_std, all_prior_mean, all_prior_std, all_samples):
                    kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                    kldLossArr.append(kldLoss)
                kldLossesArr.append(kldLossArr)
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
        if args.model == "vrnn" or args.model=="sketch-rnn":
            kldLossesMat = np.array(kldLossesArr)
            meanKLDLosses = np.mean(kldLossesMat, axis=0)
        return preds, targets, datas, means, stds, meanKLDLosses, dataTimesArr, targetTimesArr, zs


def getVRNNLoss(output, target, dataDict, args):
    encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
    # Calculate KLDivergence part
    totalKLDLoss = 0.0
    for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in \
            zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
        kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        totalKLDLoss += kldLoss
    # Calculate Prediction Loss
    pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
    unNPred = unNormalize(pred.detach(), dataDict["train_mean"], dataDict["train_std"])
    unNTarget = unNormalize(target.detach(), dataDict["train_mean"], dataDict["train_std"])
    if args.criterion == "RMSE":
        predLoss = torch.sqrt(torch.mean((pred - target) ** 2))
        unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)**2))
    elif args.criterion == "L1Loss":
        predLoss = torch.mean(torch.abs(pred - target))
        unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
    else:
        assert False, "bad loss function"
    assert not np.isnan(predLoss.cpu().detach().numpy())
    assert not np.isnan(unNormalizedLoss.cpu().detach().numpy())
    return totalKLDLoss, predLoss, unNormalizedLoss


def getRNNLoss(output, target, dataDict, args):
    if args.criterion == "RMSE":
        o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
        t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
        loss = torch.sqrt(torch.mean((o - t) ** 2))
    elif args.criterion == "L1Loss":
        o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
        t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
        loss = torch.mean(torch.abs(o - t))
    else:
        assert False, "bad loss function"
    return loss


def getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, dataDict):
    # mean and std of shape (batch_size, z_dim)
    kld = sketchRNNKLD(latentMean, latentStd)
    # calculate reconstruction loss
    unNPred = unNormalize(predOut.detach(), dataDict["train_mean"], dataDict["train_std"])
    unNTarget = unNormalize(target.detach(), dataDict["train_mean"], dataDict["train_std"])
    if args.criterion == "RMSE":
        predLoss = torch.sqrt(torch.mean((predOut - target)**2))    
        unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)**2))
    elif args.criterion == "L1Loss":
        predLoss = torch.mean(torch.abs(predOut - target))
        unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
    else:
        assert False, "bad loss function"
    return kld, predLoss, unNormalizedLoss

def getValLoss(output, target, dataDict, args):
    if args.model == "sketch-rnn":
        latentMean, latentStd, z, predOut, predMeanOut, predStdOut = output
        kldLoss, predLoss, unNormalizedLoss = getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, dataDict)
        return kldLoss.item(), unNormalizedLoss.item()
    if args.model == "vrnn":
        totalKLDLoss, predLoss, unNormalizedLoss = getVRNNLoss(output, target, dataDict, args)
        return totalKLDLoss.item() / args.sequence_len, unNormalizedLoss.item()
    elif args.model == "rnn":
        unNormalizedLoss = getRNNLoss(output, target, dataDict, args)
        return 0.0, unNormalizedLoss.item()

def train(train_loader, val_loader, model, lr, args, dataDict, epoch, optimizer, kldLossWeight):
    epochKLDLossTrain = 0.0
    epochReconLossTrain = 0.0
    epochKLDLossVal = 0.0
    epochReconLossVal = 0.0
    nTrainBatches = 0
    nValBatches = 0
    for batch_idx, (data, target, dataTimes, targetTimes) in enumerate(train_loader):
        nTrainBatches += 1
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        optimizer.zero_grad()
        output = model(data, target, epoch)
        del data
        if args.model == "sketch-rnn":
            latentMean, latentStd, z, predOut, predMeanOut, predStdOut = output
            kldLoss, predLoss, unNormalizedLoss = getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut, args, target, dataDict)
            loss = (kldLoss * kldLossWeight) + predLoss
            epochKLDLossTrain += (kldLoss)
            epochReconLossTrain += (unNormalizedLoss)
            print("batch index: {}, recon loss: {}, kld loss: {}".format(batch_idx, unNormalizedLoss, kldLoss))
        elif args.model == "vrnn":
            kldLoss, predLoss, unNormalizedLoss = getVRNNLoss(output, target, dataDict, args)
            loss = (kldLoss * kldLossWeight) / args.sequence_len + predLoss
            epochKLDLossTrain += (kldLoss / args.sequence_len)
            epochReconLossTrain += (unNormalizedLoss)
            print("batch index: {}, recon loss: {}, kld loss: {}".format(batch_idx, unNormalizedLoss, kldLoss / args.sequence_len))
        elif args.model == "rnn":
            loss = getRNNLoss(output, target, dataDict, args) # unNormalized Loss
            epochReconLossTrain += (loss)
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    for batch_idx, (data, target, dataTimes, targetTimes) in enumerate(val_loader):
        nValBatches += 1
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
        output = model(data, target, epoch)
        validationKldLoss, validationReconLoss = getValLoss(output, target, dataDict, args)
        epochKLDLossVal += validationKldLoss
        epochReconLossVal += validationReconLoss

    avgTrainReconLoss = epochReconLossTrain / nTrainBatches
    avgTrainKLDLoss = epochKLDLossTrain / nTrainBatches
    avgValReconLoss = epochReconLossVal / nValBatches
    avgValKLDLoss = epochKLDLossVal / nValBatches
    if args.model == "vrnn" or args.model == "sketch-rnn":
        print('====> Average Train Recon Loss: {} Average Train KLD Loss: {} Average Val Recon Loss: {} Average Val KLD Loss: {}'.format(avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss))
    elif args.model == "rnn":
        print("===> Average Train Loss: {} Average Val Loss: {}".format(avgTrainReconLoss, avgValReconLoss))
    else:
        assert False, "Bad model specified"
    return avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss


# def train(train_loader, val_loader, model, lr, args, dataDict, epoch, optimizer, kldLossWeight):
#     clip = 10
#     train_kld_loss = 0.0
#     train_recon_loss = 0.0
#     val_recon_loss = 0.0
#     val_kld_loss = 0.0
#     start = time.time()
#     # Train
#     nTrainBatches = 0
#     for batch_idx, (data, target, dataTimes, targetTimes) in enumerate(train_loader):
#         data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
#         target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
#         optimizer.zero_grad()
#         output = model(data, target, epoch)
#         del data
#         if args.model == "sketch-rnn":
#             latentMean, latentStd, z, predOut, predMeanOut, predStdOut = output
#             kldLoss, predLoss, unNormalizedLoss = getSketchRNNLoss(latentMean, latentStd, predOut, predMeanOut, predStdOut)

#         if args.model == "vrnn":
#             encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
#             # Calculate KLDivergence part
#             totalKLDLoss = 0.0
#             for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
#                 kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
#                 totalKLDLoss += kldLossWeight * kldLoss

#             #Calculate Prediction Loss
#             pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
#             unNPred = unNormalize(pred.detach(), dataDict["train_mean"], dataDict["train_std"])
#             unNTarget = unNormalize(target.detach(), dataDict["train_mean"], dataDict["train_std"])
#             assert pred.size() == target.size()
#             if args.criterion == "RMSE":
#                 predLoss = torch.sqrt(torch.mean((pred - target)**2))    
#                 unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)**2))

#             elif args.criterion == "L1Loss":
#                 predLoss = torch.mean(torch.abs(pred - target))
#                 unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))
#             loss = ((totalKLDLoss / args.sequence_len) + predLoss)
#         elif args.criterion == "RMSE":
#             o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
#             t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
#             loss = torch.sqrt(torch.mean((o - t)**2))
#         elif args.criterion == "L1Loss":
#             o = unNormalize(output, dataDict["train_mean"], dataDict["train_std"])
#             t = unNormalize(target, dataDict["train_mean"], dataDict["train_std"])
#             loss = torch.mean(torch.abs(o - t))
#         else:
#             assert False, "bad loss function"
#         loss.backward()
#         optimizer.step()
#         #grad norm clipping, only in pytorch version >= 1.10
#         nn.utils.clip_grad_norm_(model.parameters(), clip)

#         #printing
#         if args.model == "vrnn":
#             bReconLoss = unNormalizedLoss.data.item()
#             bKLDLoss = totalKLDLoss.data.item() / args.sequence_len / kldLossWeight
#             if batch_idx % args.print_every == 0:
#                 print("batch index: {}, recon loss: {}, kld loss: {}".format(batch_idx, bReconLoss, bKLDLoss))
#             train_recon_loss += bReconLoss
#             train_kld_loss += bKLDLoss
#         else:
#             bLoss = loss.data.item()
#             train_recon_loss += bLoss
#             if batch_idx % args.print_every == 0:
#                 print("batch index: {}, loss: {}".format(batch_idx, bLoss))

#     nTrainBatches = batch_idx + 1
#     # Validate
#     nValBatches = 0
#     with torch.no_grad():
#         nValBatches += 1
#         for batch_idx, (data, target, dataTimes, targetTimes) in enumerate(val_loader):
#             data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
#             target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
#             output = model(data, target, 0, noSample=True)
#             del data
#             if args.model == "vrnn":
#                 encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples = output
#                 pred = torch.cat([torch.unsqueeze(y, dim=0) for y in all_samples])
#                 unNPred = unNormalize(pred.detach(), dataDict["val_mean"], dataDict["val_std"])
#                 unNTarget = unNormalize(target.detach(), dataDict["val_mean"], dataDict["val_std"])
#                 totalKLDLoss = 0.0
#                 for enc_mean_t, enc_std_t, decoder_mean_t, decoder_std_t, prior_mean_t, prior_std_t, sample in zip(encoder_means, encoder_stds, decoder_means, decoder_stds, prior_means, prior_stds, all_samples):
#                     kldLoss = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
#                     totalKLDLoss += kldLossWeight * kldLoss
#                 if args.criterion == "RMSE":
#                     predLoss = torch.sqrt(torch.mean((pred - target)**2))    
#                     unNormalizedLoss = torch.sqrt(torch.mean((unNPred - unNTarget)**2))

#                 elif args.criterion == "L1Loss":
#                     predLoss = torch.mean(torch.abs(pred - target))
#                     unNormalizedLoss = torch.mean(torch.abs(unNPred - unNTarget))

#             elif args.criterion == "RMSE":
#                 o = unNormalize(output, dataDict["val_mean"], dataDict["val_std"])
#                 t = unNormalize(target, dataDict["val_mean"], dataDict["val_std"])
#                 loss = torch.sqrt(torch.mean((o - t)**2))

#             elif args.criterion == "L1Loss":
#                 o = unNormalize(output, dataDict["val_mean"], dataDict["val_std"])
#                 t = unNormalize(target, dataDict["val_mean"], dataDict["val_std"])
#                 loss = torch.mean(torch.abs(o - t))
#             else:
#                 assert False, "bad loss function"

#             if args.model == "vrnn":
#                 val_recon_loss += unNormalizedLoss.data.item()
#                 val_kld_loss += totalKLDLoss.data.item() / args.sequence_len / kldLossWeight
#             else:
#                 val_recon_loss += loss.data.item()
#     nValBatches = batch_idx + 1
#     avgTrainReconLoss = train_recon_loss / nTrainBatches
#     avgTrainKLDLoss = train_kld_loss / nTrainBatches
#     avgValReconLoss = val_recon_loss / nValBatches
#     avgValKLDLoss = val_kld_loss / nValBatches
#     if args.model == "vrnn":
#         print('====> Average Train Recon Loss: {} Average Train KLD Loss: {} Average Val Recon Loss: {} Average Val KLD Loss: {}'.format(avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss))
#     elif args.model == "rnn":
#         print("===> Average Train Loss: {} Average Val Loss: {}".format(avgTrainReconLoss, avgValReconLoss))
#     else:
#         assert False
#     return avgTrainReconLoss, avgTrainKLDLoss, avgValReconLoss, avgValKLDLoss
