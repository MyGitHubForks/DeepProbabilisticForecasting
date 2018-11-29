import torch
import numpy as np
from torch.utils import data
 
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            # tx_padding = np.repeat(tx[-1:], num_padding, axis=0)
            # ty_padding = np.repeat(ty[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            # tx = np.concatenate([tx, tx_padding], axis=0)
            # ty = np.concatenate([ty, ty_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()
        

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = np.transpose(self.xs[start_ind: end_ind, ...], (1,0,3,2))
                y_i = np.transpose(self.ys[start_ind: end_ind, :,:,0], (1,0,2))
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs, self.ys = self.xs[permutation], self.ys[permutation]

class DataLoaderWithTime(object):
    def __init__(self, xs, ys, tx, ty, batch_size, pad_with_last_sample=True, shuffle=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            tx_padding = np.repeat(tx[-1:], num_padding, axis=0)
            ty_padding = np.repeat(ty[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            tx = np.concatenate([tx, tx_padding], axis=0)
            ty = np.concatenate([ty, ty_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.tx = tx
        self.ty = ty
        if shuffle:
            self.shuffle()

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                tx_i = self.tx[start_ind: end_ind, ...]
                ty_i = self.ty[start_ind: end_ind, ...]
                yield (x_i, y_i, tx_i, ty_i)
                self.current_ind += 1

        return _wrapper()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs, self.ys, self.tx, self.ty = self.xs[permutation], self.ys[permutation], self.tx[permutation], self.ty[permutation]
