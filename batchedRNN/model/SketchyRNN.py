import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import numpy as np

class SketchRNN(nn.Module):
	def __init__(self, args):
		super(SketchRNN, self).__init__()

		self.x_dim = args.x_dim
		self.h_dim = args.h_dim
		self.z_dim = args.z_dim
		self.n_layers = 1
		self.useCuda = args.cuda
		self.args = args

		self.phi_x = nn.Sequential(
			nn.Linear(self.x_dim * self.args.channels, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())

		self.encoder = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bidirectional=True)

		self.mean = nn.Linear(self.h_dim, self.z_dim)

		self.std = nn.Sequential(
			nn.Linear(self.h_dim, self.z_dim),
			nn.Softplus())

		self.decoder_mean = nn.Sequential(
			nn.Linear(self.h_dim, self.args.output_dim),
			nn.Softplus())

		self.decoder_std = nn.Sequential(
			nn.Linear(self.h_dim, self.args.output_dim),
			nn.Softplus())

		self.getFirstDecoderHidden = nn.Sequential(
			nn.Linear(2 * self.z_dim, self.h_dim),
			nn.Tanh())

		self.decoder = nn.GRU(self.h_dim + 2 * self.z_dim, self.h_dim, self.n_layers)

		self.prepTargetForNextSequence = nn.Linear(self.x_dim, self.h_dim)

		self.use_schedule_sampling = args.use_schedule_sampling
		self.scheduling_start = args.scheduling_start
		self.scheduling_end = args.scheduling_end

	def scheduleSample(self, epoch):
		eps = max(self.scheduling_start - 
			(self.scheduling_start - self.scheduling_end)* epoch / self.args.n_epochs,
			self.scheduling_end)
		return np.random.binomial(1, eps)

	def forward(self, x, target, epoch):
		x = x.contiguous().view(x.size(0), x.size(1), -1)
		# extract features from x
		phiX = self.phi_x(x)
		# Get encoder hidden state
		h_0 = self.initHidden()
		# pass extracted x through encoder GRU
		
		encoder_output, encoder_hidden = self.encoder(phiX, h_0)
		# Reshape encoder_hidden from (2, batch, h_dim) to (batch, 2*h_dim)
		#hidden_forward, hidden_backward = torch.split(encoder_hidden,1,0)
		#encoder_hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
		# calculate normal distr parameters
		latentMean = self.mean(encoder_hidden)
		latentStd = self.std(encoder_hidden)
		# sample z from normal distribution with parameters calculated above
		# z Shape (2, batch, z_dim)
		z = self._reparameterized_sample(latentMean, latentStd)
		z_forward, z_backward = torch.split(z,1,0)
		# shape z_expanded (batch size, 2 * z_dim)
		z_expanded = torch.cat([z_forward.squeeze(0), z_backward.squeeze(0)],1)
		# get h_0 for decoder
		decoder_h = self.getFirstDecoderHidden(z_expanded).unsqueeze(0)
		# get first input to decoder (NULL Values)
		
		s_0 = Variable(torch.zeros(self.args.batch_size, self.h_dim))
		if self.useCuda:
			s_0 = s_0.cuda()
		inp = torch.cat((z_expanded, s_0), 1).unsqueeze(0)

		# If you are not training or you do not want to use schedule sampling during training
		if not self.training or not self.use_schedule_sampling:
			sample=0
		else:
			sample = self.scheduleSample(epoch)
		# List to collect decoder output
		ys = []
		means = []
		stds = []
		for t in range(self.args.target_sequence_len):
			#print("inp size", inp.size())
			#print("decoder_h size ", decoder_h.size())
			# input should be (seq_len, batch, h_dim + 2 * z_dim)
			# decoder_h should be (2, batch, h_dim)
			decoder_out, decoder_h = self.decoder(inp, decoder_h)
			#print("decoder_out size", decoder_out.size())
			#print("previous target size", target[t-1].size()) 
			if sample:
				#print("sample")
				preppedTarget =  self.prepTargetForNextSequence(target[t-1])
				inp = torch.cat((z_expanded, preppedTarget), 1).unsqueeze(0)
			else:
				#print("no sample")
				inp = torch.cat((z_expanded, decoder_out.squeeze()), 1).unsqueeze(0)
			outputMean = self.decoder_mean(decoder_out)
			outputStd = self.decoder_std(decoder_out)
			pred = self._reparameterized_sample(outputMean, outputStd)
			ys += [pred]
			means += [outputMean]
			stds += [outputStd]
		predOut = torch.cat(ys, dim=0)
		predMeanOut = torch.cat(means, dim=0)
		predStdOut = torch.cat(stds, dim=0)
		return latentMean, latentStd, z, predOut, predMeanOut, predStdOut

	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size(), device=self.args._device).normal_()
		if self.useCuda:
			eps = eps.cuda()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)

	def initHidden(self):
		# Encoder is bidirectional, so needs n_layers * 2
		result = Variable(torch.zeros(self.n_layers * 2, self.args.batch_size, self.h_dim))
		if self.useCuda:
			return result.cuda()
		else:
			return result
