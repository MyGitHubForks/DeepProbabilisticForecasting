import numpy as np
import torch
import torch.nn as nn

import model.Attn as AF
import model.ATTNDecoder as ADF
import model.DecoderRNN as DRF
import model.EncoderRNN as ERF


class TestFunctions(object):
	"""docstring for TestFunctions"""
	def __init__(self, DataObj):
		super(TestFunctions, self).__init__()
		self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.CUDA = torch.cuda.is_available()
		self.DataObj = DataObj

	def testModel(self):
	    small_batch_size = 3
	    input_batches, input_lengths, target_batches, target_lengths = self.DataObj.random_batch(small_batch_size)
	    print('input_batches', input_batches.size()) # (max_len x batch_size x NFeatues)
	    print('target_batches', target_batches.size()) # (max_len x batch_size x NFeatures)
	    small_hidden_size = 8
	    small_n_layers = 2
	    
	    #sequence_lengths, num_features, hidden_size, n_layers=1, dropout=0.1
	    encoderTest = ERF.EncoderRNN(input_lengths[0], input_batches.size(2), small_hidden_size, n_layers=small_n_layers)
	    
	    #attn_model, hidden_size, output_size, n_layers=1, dropout=0.1
	    #decoderTest = LuongAttnDecoderRNN('general', small_hidden_size, target_batches.size(2), n_layers=small_n_layers)
	    
	    #hidden_size, output_size, n_layers=1, dropout=0.1
	    decoderTest = DRF.DecoderRNN(small_hidden_size, target_batches.size(2), n_layers=small_n_layers)
	    
	    #Encode
	    encoder_outputs, encoder_hidden = encoderTest(input_batches, input_lengths, hidden=None)
	    print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
	    print('encoder_hidden', encoder_hidden.size()) # n_layers x batch_size x hidden_size
	    
	    # Decoder
	    max_target_length = max(target_lengths)

	    if self.CUDA:
	    	decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((small_batch_size, target_batches.size(2))), device=self._device))
	    else:
	    	decoder_input = torch.autograd.Variable(torch.FloatTensor(np.zeros((small_batch_size, target_batches.size(2)))))
	    
	    # use last encoder hidden as initial decoder hidden
	    decoder_hidden = encoder_hidden

	    if self.CUDA
	    	all_decoder_outputs = torch.autograd.Variable(torch.zeros(max_target_length, small_batch_size, target_batches.size(2), device=self._device)).cuda()
	    else:
	    	all_decoder_outputs = torch.autograd.Variable(torch.zeros(max_target_length, small_batch_size, target_batches.size(2), device=self._device))

	    # Run through decoder one time step at a time
	    for t in range(max_target_length):
	        #print("t=",t)
	        decoder_output, decoder_hidden = decoderTest(decoder_input, decoder_hidden)
	        all_decoder_outputs[t] = decoder_output
	        decoder_input = target_batches[t] # Next input is current target
	        
	    loss = nn.L1Loss()
	    print(type(all_decoder_outputs.detach()))
	    print(type(target_batches))
	    if self.CUDA:
	    	lossVal = loss(all_decoder_outputs.detach().cuda(), target_batches.cuda()) / max_target_length.data[0].float().cuda() / small_batch_size
	    else:
	    	lossVal = loss(all_decoder_outputs.detach(), target_batches) / max_target_length.data[0].float() / small_batch_size
	    print("loss", lossVal)

	def testAttnModel(self):
		small_batch_size = 3
	    input_batches, input_lengths, target_batches, target_lengths = self.DataObj.random_batch(small_batch_size)
	    print('input_batches', input_batches.size()) # (max_len x batch_size x NFeatues)
	    print('target_batches', target_batches.size()) # (max_len x batch_size x NFeatures)
	    small_hidden_size = 8
	    small_n_layers = 2
	    
	    #sequence_lengths, num_features, hidden_size, n_layers=1, dropout=0.1
	    encoderTest = EncoderRNN(input_lengths[0], input_batches.size(2), small_hidden_size, n_layers=small_n_layers)
	    
	    #self, hidden_size, output_size, n_layers=1, dropout=0.1
	    decoderTest = AttnDecoderRNN(small_hidden_size, target_batches.size(2), n_layers=small_n_layers, method="general")
	    
	    #Encode
	    encoder_outputs, encoder_hidden = encoderTest(input_batches, input_lengths[0], None)
	    print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
	    print('encoder_hidden', encoder_hidden.size()) # n_layers x batch_size x hidden_size
	    
	    # Decoder
	    max_target_length = max(target_lengths)
	    if self.CUDA:
		    decoder_input = torch.autograd.Variable(torch.zeros((small_batch_size, target_batches.size(2)), dtype=torch.float32, device=self._device)).cuda()
		else:
			decoder_input = torch.autograd.Variable(torch.zeros((small_batch_size, target_batches.size(2)), dtype=torch.float32, device=self._device))

	    # use last encoder hidden as initial decoder hidden
	    decoder_hidden = encoder_hidden
	    if self.CUDA:
	    	all_decoder_outputs = torch.autograd.Variable(torch.zeros(max_target_length, small_batch_size, target_batches.size(2), dtype= torch.float32, device=self._device)).cuda()
	    else:
	    	all_decoder_outputs = torch.autograd.Variable(torch.zeros(max_target_length, small_batch_size, target_batches.size(2), dtype= torch.float32, device=self._device))

	    # Run through decoder one time step at a time
	    for t in range(max_target_length):
	        #print("t=",t)
	        decoder_output, decoder_hidden, attnVec = decoderTest(decoder_input, decoder_hidden, encoder_outputs)
	        all_decoder_outputs[t] = decoder_output
	        decoder_input = target_batches[t] # Next input is current target
	        
	    loss = nn.L1Loss()
	    if self.CUDA:
	    	lossVal = loss(all_decoder_outputs.detach().cuda(), target_batches.cuda()) / small_batch_size / max_target_length.float()
	    else:
	    	lossVal = loss(all_decoder_outputs.detach(), target_batches) / small_batch_size / max_target_length.float()
	    print("loss", lossVal)
