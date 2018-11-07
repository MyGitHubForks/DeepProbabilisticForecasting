from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, args=None):
        super(EncoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        #print("encoder input",input[10,31])
        embedded = self.embedding(input)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        #print("hidden state cell", output[0,10,31])
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        #print("decoder input cell",input[10,31])
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



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.attn = nn.Linear(self.hidden_size * 2, self.args.sequence_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size * 2, output_size)

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, encoder_outputs):
        assert False, "Do not use AttnDecoder at the moment"
        # input  B x H
        # hidden 1 x B x H
        # output B x H

        embedded = F.relu(self.embedding(input))

        # Calculate attention weights and apply to encoder outputs

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden.squeeze(0)), 1))) # B x T

        # B x 1 x T * B x T x H = B x 1 x H = B x H
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs.transpose(0, 1)).squeeze(1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = self.attn_combine(torch.cat((embedded, context), 1))
        rnn_input = rnn_input.unsqueeze(0)

        output, hidden = self.gru(rnn_input, hidden)

        output = self.softmax(self.out( torch.cat((output.squeeze(0), context), 1) ))

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

        self.enc = EncoderRNN(self.args.x_dim, self.args.h_dim, args=args)

        if self.args.use_attn:
            self.dec = AttnDecoderRNN(self.args.h_dim, self.args.x_dim, args=args)
        else:
            self.dec = DecoderRNN(self.args.h_dim, self.args.x_dim, args=args)

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

    def forward(self, x, target, epoch, noSample=False):
        encoder_hidden = self.enc.initHidden()
        #print("seq2seq forward x size",x.size())
        hs = []
        for t in range(self.args.sequence_len):
            encoder_output, encoder_hidden = self.enc(x[t], encoder_hidden)
            hs += [encoder_output]

        decoder_hidden = hs[-1]

        hs = torch.cat(hs, 0)

        inp = Variable(torch.zeros(self.args.batch_size, self.args.x_dim))
        if self.args.cuda: inp = inp.cuda()
        ys = []
        if noSample:
            sample=0
        else:
            sample = self.scheduleSample(epoch)
        if self.args.use_attn:
            for t in range(self.args.sequence_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden, hs)
                if sample:
                    inp = target[t-1]
                else:
                    inp = decoder_output
                ys += [decoder_output]
        else:
            for t in range(self.args.sequence_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden)
                if sample:
                    inp = target[t-1]
                else:
                    inp = decoder_output
                ys += [decoder_output]
        #uncomment to stop after 1 iteration
        return torch.cat([torch.unsqueeze(y, dim=0) for y in ys])
