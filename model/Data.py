import torch
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, dataPath):
        self.data = np.load(dataPath)
        self._device = "cuda: 0" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        #total number of samples
        return self.data["x"].shape[0]

    def __getitem__(self, index):
        x = self.data["x"][index, :, :, 0]
        y = self.data["y"][index, :, :, 0]
        return x, y
