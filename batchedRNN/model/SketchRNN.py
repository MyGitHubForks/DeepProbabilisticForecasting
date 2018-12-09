import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class SketchRNNEncoder(nn.Module):
    def __init__(self, args):
        super(SketchRNNEncoder, self).__init__()
        self.args = args
        if self.args.bidirectionalEncoder:
            self.directions = 2
        else:
            self.directions = 1
        # bidirectional lstm:
        self.lstm = nn.LSTM(self.args.x_dim * self.args.channels, self.args.encoder_h_dim, \
            self.args.n_layers, dropout=self.args.encoder_layer_dropout, bidirectional=self.args.bidirectionalEncoder)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(self.args.n_layers * self.directions * self.args.encoder_h_dim, self.args.z_dim)
        self.fc_sigma = nn.Linear(self.args.n_layers * self.directions * self.args.encoder_h_dim, self.args.z_dim)
        

    def forward(self, input, hidden_cell=None):
        if hidden_cell is None:
            hidden_cell = self.init_hidden_cell()
        _, (hidden, cell) = self.lstm(input, hidden_cell)
        # convert hidden size from (n_layers * directions, batch_size, h_dim)
        #                       to (batch_size, n_layers * directions * h_dim)
        hiddenLayers = torch.split(hidden, 1, 0)
        if self.directions == 2 and self.args.n_layers == 2:
            assert len(hiddenLayers) == 4
        hidden_cat = torch.cat([h.squeeze(0) for h in hiddenLayers], 1)
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2)
        z_size = mu.size()
        if self.args.cuda:
            N = Variable(torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda())
        else:
            N = Variable(torch.normal(torch.zeros(z_size),torch.ones(z_size)))
        z = mu + sigma*N
        return z, mu, sigma_hat



    def init_hidden_cell(self):
        hidden = Variable(torch.zeros(self.directions * self.args.n_layers, self.args.batch_size, self.args.encoder_h_dim))
        cell = Variable(torch.zeros(self.directions * self.args.n_layers, self.args.batch_size, self.args.encoder_h_dim))
        if self.args.cuda:
            return (hidden.cuda(), cell.cuda())
        else:
            return (hidden, cell)

class SketchRNNDecoder(nn.Module):
    def __init__(self, args):
        super(SketchRNNDecoder, self).__init__()
        self.args = args
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(self.args.z_dim, 2 * self.args.n_layers * self.args.decoder_h_dim)
        # unidirectional lstm:
        self.lstm = nn.LSTM(self.args.z_dim + self.args.output_dim, self.args.decoder_h_dim, self.args.n_layers, dropout=self.args.decoder_layer_dropout)
        self.muLayer = nn.Linear(self.args.decoder_h_dim, self.args.output_dim * self.args.n_gaussians)
        self.sigmaLayer = nn.Linear(self.args.decoder_h_dim, self.args.output_dim * self.args.n_gaussians)
        self.piLayer = nn.Linear(self.args.decoder_h_dim, self.args.output_dim * self.args.n_gaussians)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            layers = torch.split(torch.tanh(self.fc_hc(z)),self.args.decoder_h_dim,1)
            hidden = torch.stack(layers[:int(len(layers) / 2)], dim=0)
            cell = torch.stack(layers[int(len(layers) / 2): ], dim=0)
            hidden_cell = (hidden.contiguous(), cell.contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        # outputs size: (seq_len, batch, num_directions * hidden_size)
        # hidden size: (num_layers * num_directions, batch, hidden_size)
        # cell size: (num_layers * num_directions, batch, hidden_size)
        mu = self.muLayer(outputs).view(-1, self.args.batch_size, self.args.output_dim, self.args.n_gaussians)
        sigma = self.sigmaLayer(outputs).view(-1, self.args.batch_size, self.args.output_dim, self.args.n_gaussians)
        pi = self.piLayer(outputs).view(-1, self.args.batch_size, self.args.output_dim, self.args.n_gaussians)
        pi = F.softmax(pi, 3)
        sigma = torch.exp(sigma)
        return (pi, mu, sigma), (hidden, cell)

class SketchRNN(nn.Module):
    def __init__(self, args):
        super(SketchRNN, self).__init__()
        self.args = args
        if self.args.cuda:
            self.encoder = SketchRNNEncoder(args).cuda()
            self.decoder = SketchRNNDecoder(args).cuda()
        else:
            self.encoder = SketchRNNEncoder(args)
            self.decoder = SketchRNNDecoder(args)

    def scheduleSample(self, epoch):
        eps = max(self.args.scheduling_start - 
            (self.args.scheduling_start - self.args.scheduling_end)* epoch / self.args.self.args.n_epochs,
            self.args.scheduling_end)
        return np.random.binomial(1, eps)

    def generatePred(self, pi, mu, sigma):
        if self.args.cuda:
            N = Variable(torch.randn(pi.size()).cuda())
        else:
            N = Variable(torch.randn(pi.size()))
        clusterPredictions = mu + sigma * N
        weightedClusterPredictions = clusterPredictions * pi
        pred = torch.sum(weightedClusterPredictions, dim=3)
        return pred

    def allSteps(self, target, z):
        sos = self.getStartOfSequence()
        batch_init = torch.cat([sos, target[:-1,...]], 0)
        z_stack = torch.stack([z]*(self.args.target_sequence_len))
        inp = torch.cat([batch_init, z_stack], 2)
        (pi, mu, sigma), (hidden, cell) = self.decoder(inp, z)
        return (pi, mu, sigma)

    def oneStepAtATime(self, z):
        sos = self.getStartOfSequence()
        inp = torch.cat([sos, z.unsqueeze(0)], 2)
        piList, muList, sigmaList = [], [], []
        for timeStep in range(self.args.target_sequence_len):
            (pi, mu, sigma), (hidden, cell) = self.decoder(inp, z)
            pred = self.generatePred(pi, mu, sigma)
            inp = torch.cat([pred, z.unsqueeze(0)], 2)
            piList.append(pi)
            muList.append(mu)
            sigmaList.append(sigma)
        Pi = torch.cat(piList, 0)
        Mu = torch.cat(muList, 0)
        Sigma = torch.cat(sigmaList, 0)
        return (Pi, Mu, Sigma)

    def getStartOfSequence(self):
        if self.args.cuda:
            return Variable(torch.zeros(1, self.args.batch_size, self.args.output_dim).cuda())
        else:
            return Variable(torch.zeros(1, self.args.batch_size, self.args.output_dim))

    def doEncoding(self,batch):
        # convert input from [input_sequence_len, batch_size, channels, x_dim]
        #                 to [input_sequence_len, batch_size, channels * x_dim]
        embedded = batch.contiguous().view(-1, self.args.batch_size, self.args.x_dim * self.args.channels)
        z, mu, sigma_hat = self.encoder(embedded)
        return z, mu, sigma_hat

    def forward(self, batch, target, epoch):
        z, latentMean, latentStd = self.doEncoding(batch)
        if self.training:
            (Pi, Mu, Sigma) = self.allSteps(target, z)
        else:
            (Pi, Mu, Sigma) = self.oneStepAtATime(z)
        return Pi, Mu, Sigma, latentMean, latentStd
