import torch
import numpy as np
from torch.utils import data

# class DataLoader(data.Dataset):
#     def __init__(self, dataPath, batch_size):
#         self.data = np.load(dataPath)
#         self.batch_size = batch_size
#         self._device = "cuda: 0" if torch.cuda.is_available() else "cpu"

#         self.size = 
#     def __len__(self):
#         #total number of samples
#         return self.data["x"].shape[0]

#     def __getitem__(self, index):
#         x = torch.FloatTensor(self.data["x"][index, :, :, 0], device=self._device)
#         y = torch.FloatTensor(self.data["y"][index, :, :, 0], device=self._device)
#         return x, y

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=True):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
