import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from PKSoftmax import PKSoftmax

class CrossEntropyLossPKSoftmax(PKSoftmax):

    def __init__(self, hierarchy_matrix, cuda=False, verbose=False):
        super(CrossEntropyLossPKSoftmax, self).__init__(hierarchy_matrix, apply_softmax=True, cuda=cuda, verbose=verbose)

    #input shpuld be a softmax
    def forward(self, input, target):
        pred = super().forward(input)
        y = target.view(target.shape[0], 1)
        if self.cuda:
            y_onehot = torch.cuda.LongTensor(pred.shape[0], pred.shape[1])
        else:
            y_onehot = torch.LongTensor(pred.shape[0], pred.shape[1])
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        plogs = torch.log(pred)
        r = - y_onehot.float() * plogs
        return torch.mean(torch.sum(r, 1))
