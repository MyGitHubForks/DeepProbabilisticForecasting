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

def load_dataset(dataset_dir, batch_size, down_sample=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if down_sample:
            nRows = cat_data['x'].shape[0]
            down_sampled_rows = np.random.choice(range(nRows), size=nRows * down_sample, replace=False)
            data['x_' + category] = cat_data['x'][down_sampled_rows,:,:,0]
            data['y_' + category] = cat_data['y'][down_sampled_rows,:,:,0]
        else:
            data['x_' + category] = cat_data['x'][:,:,:,0]
            data['y_' + category] = cat_data['y'][:,:,:,0]
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

def plotTrainValCurve(trainLosses, valLosses, model_description, lossDescription):
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    fig, ax = plt.subplots()
    plt.xlabel("Epoch")
    plt.ylabel(lossDescription)
    plt.plot(np.arange(len(trainLosses)), trainLosses, color="red", label="train loss")
    plt.plot(np.arange(len(valLosses)), valLosses, color="blue", label="validation loss")
    plt.grid()
    plt.legend()
    plt.title("Losses for {}".format(model_description))
    
    filestring = "./figs/train_val_loss_plot_0.png"
    while(os.path.isfile(filestring)):
        filestring = filestring[:-5] + str(int(filestring[-5]) + 1) + ".png"
    plt.savefig(filestring)

def train(train_loader, val_loader, model, lr, args):
    clip = 10
    train_loss = 0.0
    val_loss = 0.0
    start = time.time()

    # Define Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1).requires_grad_()
        optimizer.zero_grad()
        output = model(data)
        #print("output", output[0,0,:5])
        #print("target", target[0,0,:5])
        if args.criterion == "MSE":
            loss = torch.sqrt(torch.mean((output - target)**2))

        elif args.criterion == "L1 Loss":
            loss = torch.mean(torch.abs(output - target))
        else:
            assert False, "bad loss function"
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if batch_idx % args.print_every == 0:
             print("batch index: {}, loss: {}".format(batch_idx, loss.data.item()))
        train_loss += loss.item()
    nTrainBatches = batch_idx + 1
    # Validate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = torch.as_tensor(data, dtype=torch.float, device=args._device).transpose(0,1)
            target = torch.as_tensor(target, dtype=torch.float, device=args._device).transpose(0,1)
            output = model(data)
            if args.criterion == "MSE":
                loss = torch.sqrt(torch.mean((output - target)**2))

            elif args.criterion == "L1 Loss":
                loss = torch.mean(torch.abs(output - target))
            else:
                assert False, "bad loss function"
            val_loss += loss.item()
    nValBatches = batch_idx + 1
    avgTrainLoss = train_loss / nTrainBatches
    avgValLoss = val_loss / nValBatches
    print('====> Average Train Loss: {} Average Val Loss: {}'.format(avgTrainLoss, avgValLoss))
    return avgTrainLoss, avgValLoss
