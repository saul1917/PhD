# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging

from . import architectures

#Increase the number of workers for better performance
NUM_WORKERS = 10
DEFAULT_SIZE = 256
DEFAULT_PATH = "/media/Data/saul/InBreastDataset"
RESULTS_PATH = "/results"
TEST_SPLIT = 0.2
#too large might overflow  gpu mem, use a not so small to have good gradient approxs. (32 recom. for Titan V)
BATCH_SIZE = 64

LOG_FILE = "/logs"
DEFAULT_ERROR_PRINT = 10
random_seed = 64325564

DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.00001
DEFAULT_MOMENTUM = 0.5
DEFAULT_NUM_SPLITS = 5

CHECKPOINT_EPOCHS = 2
DEFAULT_NAME_CHECKPOINT = 'checkpoint.ckpt'
DEFAULT_BEST_NAME_CHECKPOINT = 'best.ckpt'
#%50%, 25%, 20%, 16% 14%
#  of data unlabeled
DEFAULT_SPLITS_LABELED = 0


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    """parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
                            
                            """
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--exclude-unlabeled', default=True, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=NUM_WORKERS, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=DEFAULT_LEARNING_RATE, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=DEFAULT_MOMENTUM, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--checkpoint-epochs', default = CHECKPOINT_EPOCHS, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Should try to pick latest checkpoint from file?')
    parser.add_argument('--resumefile', default=DEFAULT_NAME_CHECKPOINT, type=str,
                        help='name of the latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--k_fold_num', type=int, default=DEFAULT_NUM_SPLITS,
                        help='Number of folds you want to use for k-fold validation')
    parser.add_argument('--random_seed', type=int, default=16031997,
                        help='Random seed to shuffle the dataset')

    parser.add_argument('--best', default=DEFAULT_BEST_NAME_CHECKPOINT, type=str,
                        help='name of the best checkpoint (default: none)')

    parser.add_argument('--splits_unlabeled', default=DEFAULT_SPLITS_LABELED, type=float,
                        help='Splits for the unlabeled/labeled data')

    parser.add_argument('--current_fold', default=1, type=float,
                        help='current fold of the unlabeled dataset, starting from 1')

    parser.add_argument('--weight_balancing', default=True, type=bool,
                        help='Balance the weights through the loss function')
    """
    BI-RADS readings of 1 and 2 as negative
    samples. Bi-rads 4, 5 and 6 are positive samples  and  BI-RADS readings of 3 were ignored
    """
    parser.add_argument('--number_classes', default=2, type=int,
                        help='Train a n-nary Bi-rads based classifier')
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
