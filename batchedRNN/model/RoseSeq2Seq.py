from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
from memory_profiler import profile

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, args=None):
        super(EncoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size * self.args.channels, hidden_size, n_layers, dropout=self.args.dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = torch.unsqueeze(embedded, 0)
        embedded = embedded.view(1, args.batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.args.dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))
        #print("decoder output", output[10,31])
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args

        self.enc = EncoderRNN(self.args.x_dim, self.args.h_dim, n_layers=self.args.n_layers, args=args)

        self.dec = DecoderRNN(self.args.h_dim, self.args.x_dim, n_layers=self.args.n_layers, args=args)

        self.use_schedule_sampling = args.use_schedule_sampling
        self.scheduling_start = args.scheduling_start
        self.scheduling_end = args.scheduling_end

    def parameters(self):
        return list(self.enc.parameters()) + list(self.dec.parameters())

    def scheduleSample(self, epoch):
        eps = max(self.args.scheduling_start - 
            (self.args.scheduling_start - self.args.scheduling_end)* epoch / self.args.n_epochs,
            self.args.scheduling_end)
        return np.random.binomial(1, eps)

    def forward(self, x, target, epoch, training=True):
        encoder_hidden = self.enc.initHidden()
        hs = []
        for t in range(self.args.sequence_len):
            encoder_output, encoder_hidden = self.enc(x[t], encoder_hidden)
            hs += [encoder_output]

        decoder_hidden = hs[-1]
        # Prepare for Decoder
        inp = Variable(torch.zeros(self.args.batch_size, self.args.x_dim))
        if self.args.cuda:
            inp = inp.cuda()
        ys = []
        if not training:
            sample=0
        else:
            sample = self.scheduleSample(epoch)
        # Decode
        for t in range(self.args.sequence_len):
            decoder_output, decoder_hidden = self.dec(inp, decoder_hidden)
            if sample:
                inp = target[t-1]
            else:
                inp = decoder_output
            ys += [decoder_output]
        return torch.cat([torch.unsqueeze(y, dim=0) for y in ys])
