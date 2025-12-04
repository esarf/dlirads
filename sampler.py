import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data import DistributedSampler, Dataset
import itertools
from tqdm.auto import tqdm
import torch.distributed as dist
import os

from typing import Iterator, List, Optional, Union
from collections import Counter
import logging
from torch import Tensor
from operator import itemgetter
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union


class CustomSampler(Sampler):

    """A CustomSampler which aim is to provide a batch of volumes with no identical patient"""

    def __init__(self, dataset, batch_size, indices, k=10000, weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = k // batch_size
        self.subject_ids = np.unique(dataset.volumes)
        self.weights = weights
        self.indices = indices
        self.depth = 1

    def __iter__(self):

        self.batches = []

        for _ in tqdm(range(self.n_batches)):

            batch_idxs = []
            vols = []

            while len(batch_idxs) < self.batch_size:
                # sample in the volumes
                if self.weights is not None:
                    j = np.random.choice(range(len(self.subject_ids)),
                                         p=self.weights / np.sum(self.weights),
                                         size=1)[0]
                else:
                    j = random.choice(range(len(self.subject_ids)))
                vol = self.subject_ids[j]
                if vol not in vols:
                    # sample in the slices attributed to each volume
                    i = random.choice(self.indices[vol])
                    batch_idxs.append(i)
                    vols.append(vol)

            self.batches.append(batch_idxs)

        return iter(list(itertools.chain(*self.batches)))

    def __len__(self):
        return(len(self.n_batches))


"""
class DistributedWeightedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        print('num_replicas',num_replicas)
        print('rank',rank)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.has_hcc
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
"""


class CustomWeightedRandomSampler(Sampler[int]):

    def __init__(self, dataset,
                 num_samples: int, replacement: bool = True,
                 generator=None) -> None:

        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.trainset = dataset.df_ven  # .labels      # dataframe

    def __iter__(self) -> Iterator[int]:

        idx_no_hcc = list(self.trainset[self.trainset['has_hcc'] == 0].index)
        idnohcc = random.sample(population=idx_no_hcc, k=self.trainset.shape[0] // 2)
        trainset_resample = idnohcc

        # iterate over the training hcc and small ones
        while len(trainset_resample) < self.trainset.shape[0]:
            liste = list(self.trainset[(self.trainset['has_hcc'] == 1) &
                                  (self.trainset['diameter_2d'] < 40)]['dp_lesion'])
            random.shuffle(liste)
            for x in liste:
                size_x = self.trainset[self.trainset['dp_lesion'] == x]['diameter_2d'].item()
                neighbor_sizes = [max(0, size_x - 10), min(40, size_x + 10)]
                neighbor_lesions = list(self.trainset[(self.trainset['has_hcc'] == 1) &
                                                 (self.trainset['diameter_2d'] > neighbor_sizes[0]) &
                                                 (self.trainset['diameter_2d'] < neighbor_sizes[1])]['dp_lesion'].index)
            trainset_resample.extend(list(neighbor_lesions))

        random.shuffle(trainset_resample)

        yield from iter(trainset_resample[:self.num_samples])

    def __len__(self) -> int:
        return self.num_samples





class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



