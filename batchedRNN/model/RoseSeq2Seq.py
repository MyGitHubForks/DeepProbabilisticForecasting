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
    def __init__(self, input_size, hidden_size, n_layers=2, bidirectional=True, args=None):
        super(EncoderRNN, self).__init__()
        self.args = args
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size * self.args.channels, hidden_size, n_layers, dropout=self.args.dropout, bidirectional=bidirectional)
        self.input_dropout = nn.Dropout(p=self.args.dropout)
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)
        embedded = torch.unsqueeze(embedded, 0)
        embedded = embedded.view(1, self.args.batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        if self.args.bidirectionalEncoder:
            directions = 2
        else:
            directions = 1
        result = Variable(torch.zeros(self.n_layers * directions, self.args.batch_size, self.hidden_size))
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
        self.input_dropout = nn.Dropout(p=self.args.dropout)
        if self.args.bidirectionalEncoder:
            directions = 2
        else:
            directions = 1
        # encoder hidden is (layers * directions, batch, hidden_size)
        # converted to (layers, batch, hidden_size * directions)
        self.gru = nn.GRU(hidden_size, directions * hidden_size, n_layers, dropout=self.args.dropout)
        # GRU output (seq_len, batch, directions * hidden_size)
        self.out = nn.Linear(directions * hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)
        embedded = F.relu(embedded)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))
        #print("decoder output", output[10,31])
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args

        self.enc = EncoderRNN(self.args.x_dim, self.args.h_dim, n_layers=self.args.n_layers, bidirectional=args.bidirectionalEncoder, args=args)

        self.dec = DecoderRNN(self.args.h_dim, self.args.x_dim, n_layers=self.args.n_layers, args=args)

        self.use_schedule_sampling = args.use_schedule_sampling
        self.scheduling_start = args.scheduling_start
        self.scheduling_end = args.scheduling_end

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

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
            encoder_output, encoder_hidden = self.enc(x[t].squeeze(), encoder_hidden)
            hs += [encoder_output]
        if self.args.bidirectionalEncoder:
            decoder_hidden = self._cat_directions(encoder_hidden)
        else:
            decoder_hidden = encoder_hidden
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
