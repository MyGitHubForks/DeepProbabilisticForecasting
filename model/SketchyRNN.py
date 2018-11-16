import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 

class SketchyRNN(nn.Module):
	def __init__(self, args):
		super(SketchyRNN, self).__init__()

		self.x_dim = args.x_dim
		self.h_dim = args.h_dim
		self.z_dim = args.z_dim
		self.n_layers = args.n_layers
		self.useCuda = args.cuda
		self.args = args

		self.phi_x = nn.Sequential(
			nn.Linear(self.x_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU())

		self.encoder = nn.GRU(self.h_dim, self.h_dim, self.n_layers,
					 bidirectional=True, dropout=args.dropout, bidirectional=True)

		self.mean = nn.Linear(self.h_dim, self.z_dim)

		self.std = nn.Sequential(
			nn.Linear(self.h_dim, self.z_dim),
			nn.Softplus())



	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size(), device=self.args._device).normal_()
		if self.useCuda:
			eps = eps.cuda()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)

