"""
Validation dataset sampler.

Author: Heng Wang
Date: 2/20/2024
"""
import numpy as np


class Sampler(object):

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(range(len(self)))


class RandomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        np.random.seed(self.epoch)
        self.epoch += 1
        return iter(np.random.permutation(len(self)))


class SortedSampler(Sampler):
    """
    Sorted Sampler.

    Sort each block of examples by key.
    """

    def __init__(self, sampler, sort_pool_size, key='src'):
        self.sampler = sampler
        self.sort_pool_size = sort_pool_size
        self.key = lambda idx: len(self.sampler.dataset[idx][key])

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        pool = []
        for idx in self.sampler:
            pool.append(idx)
            if len(pool) == self.sort_pool_size:
                pool = sorted(pool, key=self.key)
                yield from (
                    i for i in pool
                )

                pool = []
        if pool:
            pool = sorted(pool, key=self.key)
            yield from (
                i for i in pool
            )
