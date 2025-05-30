import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import itertools

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, l_indices, ul_indices, batch_size, l_batch_size):
        self.l_indices = l_indices  # * self.cfg.DATA.REPEAT
        self.ul_indices = ul_indices
        self.l_batch_size = l_batch_size
        self.ul_batch_size = batch_size - l_batch_size
        assert len(self.l_indices) >= self.l_batch_size > 0
        assert len(self.ul_indices) >= self.ul_batch_size >= 0

    def __iter__(self):
        label_iter = iterate_once(self.l_indices)
        unlabel_iter = iterate_eternally(self.ul_indices)
        if self.ul_batch_size == 0:
            return (l_batch + l_batch for (l_batch, l_batch)in zip(grouper(label_iter, self.l_batch_size), grouper(label_iter, self.l_batch_size)))
        return (l_batch + ul_batch for (l_batch, ul_batch) in zip(grouper(label_iter, self.l_batch_size), grouper(unlabel_iter, self.ul_batch_size)))

    def __len__(self):
        return len(self.l_indices) // self.l_batch_size

