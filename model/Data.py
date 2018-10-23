import torch
import numpy as np
class Data():
    def __init__(self, filepath):
        data = np.load(filepath)
        xs = data['x'][:,:,:,0]
        ys = data['y'][:,:,:,0]
        self.xExamples = xs.reshape((data['x'].shape[0], data['x'].shape[1], -1)) # num_examples x seq_length x num_features
        self.yExamples = ys.reshape((data['y'].shape[0], data['y'].shape[1], -1)) # num_examples x seq_length x num_features
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def getSequenceLength(self):
        return self.xExamples.shape[1]
    
    def getNumFeatures(self):
        return self.xExamples.shape[2]
    
    def getNumSamples(self):
        return self.xExamples.shape[0]
    
    def getXShape(self):
        return self.xExamples.shape

    def random_batch(self, batch_size):
        input_seqs = []
        target_seqs = []
        
        #Choose random pairs
        for i in range(batch_size):
            pairIDX = np.random.randint(0, self.getNumSamples())
            input_seqs.append(self.xExamples[pairIDX, :, :])
            target_seqs.append(self.yExamples[pairIDX, :, :])
            
        input_lengths = torch.IntTensor([len(s) for s in input_seqs], device=self._device).cuda()
        target_lengths = torch.IntTensor([len(s) for s in target_seqs], device=self._device).cuda()
        
        #convert to tensors, transpose into (max_len, x batch_size)
        inputTensor = torch.FloatTensor(input_seqs, device=self._device).cuda().transpose(0, 1)
        targetTensor = torch.FloatTensor(target_seqs, device=self._device).cuda().transpose(0, 1)
        return inputTensor, input_lengths, targetTensor, target_lengths
