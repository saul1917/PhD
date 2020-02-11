import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


#PriorKnowledgeSoftMax
def pksoftmax_np(probs, hierarchy_matrix, batch_size, verbose=False):
    #Step 2
    #gv = probs * hierarchy_matrix #batch_size=1
    gv = probs[:, np.newaxis] * hierarchy_matrix
    print("gv=x*g", gv) if verbose else None

    #Step 3
    #gvs = np.sum(gv, axis=1) #batch_size=1
    gvs = np.sum(gv, axis=2)
    print("gvs=np.sum(gv, axis=1)", gvs) if verbose else None

    #Step 4
    #gt = hierarchy_matrix.T * gvs #batch_size=1
    gt = gvs[:, np.newaxis] * hierarchy_matrix.T
    print("gt=g.T * gvs", gt) if verbose else None

    #Step 5
    #mask = np.sum(gt.T, axis=0) #batch_size=1
    mask = np.sum(gt, axis=2)
    print("mask=np.sum(gt.T, axis=0)", mask) if verbose else None

    return mask * probs

#PriorKnowledgeSoftMax in PyTorch
def pksoftmax(probs, hierarchy_matrix, batch_size, cuda=False, verbose=False):
    gv = probs.unsqueeze(1) * hierarchy_matrix
    print("gv=x*g", gv) if verbose else None

    gvs = torch.sum(gv, 2)
    print("gvs=np.sum(gv, axis=1)", gvs) if verbose else None

    gt = gvs.unsqueeze(1) * hierarchy_matrix.t()
    print("gt=g.T * gvs", gt) if verbose else None

    mask = torch.sum(gt, 2)
    print("mask=np.sum(gt.T, axis=0)", mask) if verbose else None

    return (mask * probs) + 1e-15

class PKSoftmax(nn.Module):

    def __init__(self, hierarchy_matrix, apply_softmax=False, cuda=False, verbose=False):
        super(PKSoftmax, self).__init__()
        self.hierarchy_matrix = hierarchy_matrix
        self.cuda = cuda
        self.verbose = verbose
        self.apply_softmax = apply_softmax
        if self.apply_softmax:
            if self.cuda:
                self.softmax = nn.Softmax().cuda()
            else:
                self.softmax = nn.Softmax()
        '''
        if self.cuda:
            self.lambda = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=True)
        else:
            self.lambda = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        '''
        self.init_weights()

    def init_weights(self):
        #self.lambda.data.uniform_(0.0001, 0.9999)
        pass

    #input shpuld be a softmax
    def forward(self, input):
        batch_size = input.size()[0]
        if self.apply_softmax:
            x = self.softmax(input)
        else:
            x = input
        #if self.cuda:
        #    x = x.cuda()
        return pksoftmax(x, self.hierarchy_matrix, batch_size, self.cuda, self.verbose)


'''
#Numpy
probs = np.array(   [[0.27 ,0.3, 0.13, 0.3],
                [0.2 ,0.1, 0.12, 0.58],
                [0.27 ,0.1, 0.13, 0.4]])
hierarchy_matrix = np.array([[1,0,1,0],[0,1,0,1]])
print("probs", probs)
batch_size = probs.shape[0]
print("hierarchy_matrix", hierarchy_matrix)
res = pksoftmax_np(probs, hierarchy_matrix, batch_size)
print("PKSoftmax numpy", res)

#CPU
probs = torch.FloatTensor(probs)
hierarchy_matrix = torch.FloatTensor(hierarchy_matrix)
batch_size = probs.size()[0]
res = pksoftmax(probs, hierarchy_matrix, batch_size)
print("PKSoftmax PyTorch Func", res)
l = PKSoftmax(hierarchy_matrix, apply_softmax=True, verbose=True)
probs = np.array(   [[34.3 ,34.4, 30.2, 33.2],
                    [10.2 ,11.7, 7, 10.9],
                    [0.27 ,0.1, 0.13, 0.4]])
probs = torch.FloatTensor(probs)
res = l(probs)
print("PKSoftmax PyTorch Layer", res)
'''
