import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class SketchRNNEncoder(nn.Module):
    def __init__(self):
        super(SketchRNNEncoder, self).__init__()
        if args.bidirectionalEncoder:
	    	self.directions = 2
	    else:
	    	self.directions = 1
        # bidirectional lstm:
        self.lstm = nn.LSTM(args.x_dim * args.channels, args.encoder_h_dim, \
            args.n_layers, dropout=args.dropout, bidirectional=args.bidirectionalEncoder)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(args.n_layers * self.directions * args.encoder_h_dim, args.z_dim)
        self.fc_sigma = nn.Linear(args.n_layers * self.directions * args.encoder_h_dim, args.z_dim)
        

    def forward(self, input, hidden_cell=None):
    	if hidden_cell is None:
    		hidden_cell = self.init_hidden_cell()
    	# convert input from [sequence_len, batch_size, channels, x_dim]
    	# 				  to [sequence_len, batch_size, channels * x_dim]
    	embedded = input.view(args.sequence_len, args.batch_size, args.x_dim * args.channels)
    	_, (hidden, cell) = self.LSTM(embedded, hidden_cell)
    	# convert hidden size from (n_layers * directions, batch_size, h_dim)
    	#						to (batch_size, n_layers * directions * h_dim)
    	hiddenLayers = torch.split(hidden, 1, 0)
    	if self.directions == 2 and args.n_layers == 2:
    		assert len(hiddenLayers) == 4
    	hidden_cat = torch.cat([h.squeeze(0) for h in hiddenLayers], 1)
    	mu = self.fc_mu(hidden_cat)
    	sigma_hat = self.fc_sigma(hidden_cat)
    	sigma = torch.exp(sigma_hat / 2)
    	z_size = mu.size()
    	if args.cuda:
            N = Variable(torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda())
        else:
            N = Variable(torch.normal(torch.zeros(z_size),torch.ones(z_size)))
        z = mu + sigma*N
        return z, mu, sigma_hat



    def init_hidden_cell(self):
    	hidden = Variable(torch.zeros(self.directions * args.n_layers, args.batch_size, args.encoder_h_dim))
    	cell = Variable(torch.zeros(self.directions * args.n_layers, args.batch_size, args.encoder_h_dim))
    	if args.cuda:
	    	return (hidden.cuda(), cell.cuda())
	    else:
	    	return (hidden, cell)

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(args.z_dim, 2 * args.decoder_h_dim)
        # unidirectional lstm:
        self.numGMMParams = args.x_dim * 2 # mu, sigma
        self.lstm = nn.LSTM(args.z_dim + self.numGMMParams, args.decoder_h_dim, dropout=args.dropout)
        self.fc_params = nn.Linear(args.decoder_h_dim, (self.numGMMParams + 1) * args.num_mixtures) #6*M: mu_x, mu_y, rho, sigma_x, sigma_y for GMM and pi for weight of mixture in GMM

    def forward(self, inputs, z, hidden_cell=None):
    	if hidden_cell is None:
    		hidden, cell = torch.split(F.tanh(self.fc_hc(z)),args.decoder_h_dim,1)
    		hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
    	outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
    	# outputs size: (seq_len, batch, num_directions * hidden_size)
    	# hidden size: (num_layers * num_directions, batch, hidden_size)
    	# cell size: (num_layers * num_directions, batch, hidden_size)
    	if self.training:
    		y = self.fc_params(outputs.view(-1, args.decoder_h_dim))
    	else:
    		y = self.fc_params(hidden.view(-1, args.decoder_h_dim))
    	params = torch.split(y, self.numGMMParams + 1, 1)
    	params_mixture = torch.stack(params)
    	mixture_mu, mixture_sigma, pi = torch.split(params_mixture, args.x_dim, 2)
    	





