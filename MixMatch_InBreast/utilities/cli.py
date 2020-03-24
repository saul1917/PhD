# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging



DEFAULT_PATH = "/media/Data/saul/Datasets/Inbreast_folder_per_class"
NUMBER_CLASSES = 6
NUMBER_LABELED_OBSERVATIONS = 150
BATCH_SIZE = 12
SIZE_IMAGE = 100
LAMBDA_DEFAULT = 100
# Modified from
K_DEFAULT = 2
T_DEFAULT = 0.25
ALPHA_DEFAULT = 0.75
LR_DEFAULT = 2e-4
WEIGHT_DECAY_DEFAULT = 0.02
DEFAULT_RESULTS_FILE = "Stats.csv"
LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--path_labeled', type=str, default=DEFAULT_PATH,
                        help='The directory with the labeled data')
    parser.add_argument('--path_unlabeled', type=str, default="",
                        help='The directory with the unlabeled data')

    parser.add_argument('--results_file_name', type=str, default=DEFAULT_RESULTS_FILE,
                        help='Name of results file')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=4, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=LR_DEFAULT, type=float,
                        metavar='LR', help='learning rate')
    parser.add_argument('--momentum', default=WEIGHT_DECAY_DEFAULT, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=WEIGHT_DECAY_DEFAULT, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--K_transforms', default=K_DEFAULT, type=int, metavar='K', help = 'Number of simple transformations')
    parser.add_argument('--T_sharpening', default=T_DEFAULT, type=float, metavar='T',
                        help='Sharpening coefficient')

    parser.add_argument('--alpha_mix', default=ALPHA_DEFAULT, type=float, metavar='A',
                        help='Mix alpha coefficient')

    parser.add_argument('--mode', default="ssdl", type=str,
                        help='Modes: fully_supervised, partial_supervised, ssdl')
    parser.add_argument('--balanced_losss', default=True, type=bool,
                        help='Balance the cross entropy loss')

    parser.add_argument('--lambda_unsupervised', default=LAMBDA_DEFAULT, type=float,
                        help='Unsupervised learning coefficient')

    parser.add_argument('--number_labeled', default=NUMBER_LABELED_OBSERVATIONS, type=float, metavar='A',
                        help='Number of labeled observations')

    parser.add_argument('--model', default="densenet", type=str, metavar='A',
                        help='Model to use')

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

    logging.info("Using these command line args: %s", " ".join(cmdline_args))

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
