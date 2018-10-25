import torch
import torch.nn as nn
from torch import optim
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# def trainOneBatch(input_batch, input_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
#     # Zero gradients of both optimizers
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#     batch_size = input_batch.size(1)
#     CUDA = torch.cuda.is_available()
#     _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Run words through encoder
#     encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, None)
    
#     # Prepare decoder input and output variables
#     if CUDA:
#         decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch_size, target_batch.size(2))), device=_device)).cuda()
#     else:
#         decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch_size, target_batch.size(2))), device=_device))
    
#     # use last encoder hidden as initial decoder hidden
#     if CUDA:
#         decoder_hidden = encoder_hidden.cuda()
#         all_decoder_outputs = torch.autograd.Variable(torch.zeros(max(target_lengths), batch_size, target_batch.size(2), device=_device)).cuda()
#     else:
#         decoder_hidden = encoder_hidden
#         all_decoder_outputs = torch.autograd.Variable(torch.zeros(max(target_lengths), batch_size, target_batch.size(2), device=_device))

#     # Decode
#     for t in range(max(target_lengths
#                       )):
#         decoder_output, decoder_hidden, decoder_attn = decoder(
#             decoder_input, decoder_hidden, encoder_outputs
#         )
#         if CUDA:
#             all_decoder_outputs[t] = decoder_output.cuda()
#             decoder_input = target_batch[t].detach().cuda() # Next input is current target
#         else:
#             all_decoder_outputs[t] = decoder_output
#             decoder_input = target_batch[t].detach() # Next input is current target
    
#     #Calculate Loss
#     if criterion[1] == "MSE":
#         if CUDA:
#             loss = torch.sqrt(criterion[0])(all_decoder_outputs.cuda(), target_batch.cuda())
#         else:
#             loss = torch.sqrt(criterion[0])(all_decoder_outputs, target_batch)
#     elif criterion[1] == "Mean Absolute Error":
#         if CUDA:
#             loss = criterion[0](all_decoder_outputs.cuda(), target_batch.cuda())
#         else:
#             loss = criterion[0](all_decoder_outputs, target_batch)
#     else:
#         assert False, "Cannot match loss"
    
#     loss.backward()
    
#     encoder_optimizer.step()
#     decoder_optimizer.step()
    
#     # average loss per timestep
#     return loss.item() / float(batch_size)


# # In[227]:


# def evaluate(input_batch, input_lengths, target_batch, target_lengths, encoder, decoder):
#     CUDA = torch.cuda.is_available()
#     _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     with torch.no_grad():
#         batch_size = input_batch.size(1)
#         num_features = input_batch.size(2)

#         # Run batch through encoder
#         encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, None)

#         # Prepare decoder input and output variables
#         if CUDA:
#             decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch_size, num_features)), device=_device)).cuda()
#         else:
#             decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch_size, num_features)), device=_device))

#         # use last encoder hidden as initial decoder hidden
#         if CUDA:
#             decoder_hidden = encoder_hidden.cuda()
#             all_decoder_outputs = torch.autograd.Variable(torch.zeros(max(target_lengths), batch_size, num_features, device="cuda")).cuda()
#         else:
#             decoder_hidden = encoder_hidden
#             all_decoder_outputs = torch.autograd.Variable(torch.zeros(max(target_lengths), batch_size, num_features, device=_device))

#         # Decode One sequence at a time
#         for t in range(max(target_lengths)):
#             decoder_output, decoder_hidden, decoder_attn = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs
#             )
#             if CUDA:
#                 all_decoder_outputs[t] = decoder_output.cuda()
#                 decoder_input = target_batch[t].cuda() # Next input is current target
#             else:
#                 all_decoder_outputs[t] = decoder_output
#                 decoder_input = target_batch[t] # Next input is current target
#         return all_decoder_outputs


# # In[233]:


# def validate(validationDataObj, batch_size, encoder, decoder, criterion):
#     CUDA = torch.cuda.is_available()
#     _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     valLoss = 0.0
#     num_batches = int(np.ceil(validationDataObj.getNumSamples() / batch_size))
#     for iteration in range(num_batches):
#         inputTensor, input_lengths, targetTensor, target_lengths = validationDataObj.random_batch(batch_size)
#         output = evaluate(inputTensor, input_lengths, targetTensor, target_lengths, encoder, decoder)
#         if CUDA:
#             valLoss += criterion[0](output.cuda(), targetTensor.cuda())
#         else:
#             valLoss += criterion[0](output, targetTensor)
#     return valLoss / num_batches / float(batch_size)

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
    if args.criterion == "L1 Loss":
        criterion = (nn.L1Loss(reduction="elementwise_mean"), "Mean Absolute Error")
    elif args.criterion == "MSE":
        criterion = (nn.MSELoss(),"MSE")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion[0](output, target)
        loss.backward()
        optimizer.step()
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        if batch_idx % args.print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / args.batch_size ))

        train_loss += loss.data[0]

    # Validate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = Variable(data), Variable(target)
            data = torch.transpose(data, 0, 1)
            target = torch.transpose(target, 0, 1)
            output = model(data)
            val_loss += criterion[0](output, target)

    avgTrainLoss = train_loss / len(train_loader.dataset)
    avgValLoss = val_loss / len(val_loader.dataset)
    print('====> Epoch: {} Average Train Loss: {:.4f} Average Val Loss: {:.4f}'.format(epoch, avgTrainLoss, avgValLoss))
    return avgTrainLoss, avgValLoss
