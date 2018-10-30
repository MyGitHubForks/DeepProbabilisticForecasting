import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
	def __init__(self, args):
		super(VRNN, self).__init__()

		self.x_dim = args.x_dim
		self.h_dim = args.h_dim
		self.z_dim = args.z_dim
		self.n_layers = args.n_layers
		self.useCuda = args.cuda
		self.args = args

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(self.x_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())
		self.phi_z = nn.Sequential(
			nn.Linear(self.z_dim, self.h_dim),
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(self.h_dim + self.h_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(self.h_dim, self.z_dim),
			nn.Softplus())

		#prior
		self.prior = nn.Sequential(
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())
		self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(self.h_dim, self.z_dim),
			nn.Softplus())

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(self.h_dim + self.h_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(self.h_dim, self.x_dim),
			nn.Softplus())

		self.dec_mean = nn.Linear(self.h_dim, self.x_dim)
		# self.dec_mean = nn.Sequential(
		# 	nn.Linear(self.h_dim, x_dim),
		# 	nn.Sigmoid())

		#recurrence
		self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers)


	def forward(self, x):
		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		all_prior_mean, all_prior_std = [],[]

		h = self.init_hidden()
		for t in range(x.size(0)):
			
			phi_x_t = self.phi_x(x[t])

			#encoder
			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
			phi_z_t = self.phi_z(z_t)

			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			#computing losses
			#kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
			#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
			# nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

			all_enc_std.append(enc_std_t)
			all_enc_mean.append(enc_mean_t)
			all_prior_mean.append(prior_mean_t)
			all_prior_std.append(prior_std_t)
			all_dec_mean.append(dec_mean_t)
			all_dec_std.append(dec_std_t)

			return (all_enc_mean, all_enc_std, all_dec_mean, all_dec_std, all_prior_mean, all_prior_std)


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim)

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
		if self.cuda:
			h = h.cuda()

		for t in range(seq_len):

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			sample[t] = dec_mean_t.data
	
		return sample

	def init_hidden(self):
		result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.h_dim))
		if self.useCuda:
			return result.cuda()
		else:
			return result

	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size(), device=self.args._device).normal_()
		if self.useCuda:
			eps = eps.cuda()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)


	def _nll_bernoulli(self, theta, x):
		return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


	def _nll_gauss(self, mean, std, x):
		pass
