import numpy as np
import torch
from sklearn.model_selection import KFold
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def generator_sampling_random(indices, args):
    validation_split = .3
    dataset_size = len(indices)
    for i in range(args.val_num):
        split = int(np.floor(validation_split * dataset_size))
        if args.shuffle_dataset:
            np.random.seed(args.random_seed+(10*i))
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        yield train_indices, val_indices


def size_by_archs(arch):
    if (arch == 'inception_v3'):
        size = (299,299)
    else:
        size = (224,224)
    return size

def get_indices(indices, args):
    if(args.val_mode=="k-fold"):
        kfolds = KFold(n_splits=args.k_fold_num, random_state=42, shuffle=True)
        print("Using a kfold of ", args.k_fold_num)
        kfolds.get_n_splits(indices)
        final_indices =  kfolds.split(indices)

    elif(args.val_mode=="randomsampler"):
        final_indices = generator_sampling_random(indices, args)
    elif(args.val_mode=="once"):
        args.k_fold_num = 1
        final_indices = generator_sampling_random(indices, args)
    return final_indices
