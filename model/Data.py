import torch
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, dataDirectory):
        self.idList = torch.load(dataDirectory+"idList")
        self.dataDirectory = dataDirectory
        self._device = "cuda: 0" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        #total number of samples
        return len(self.idList)

    def __getitem__(self, index):
        ID = self.idList[index]
        X = torch.load(self.dataDirectory+"features/{}".format(str(ID)), map_location=self._device)
        Y = torch.load(self.dataDirectory+"labels/{}".format(str(ID)), map_location=self._device)
        return X, Y
