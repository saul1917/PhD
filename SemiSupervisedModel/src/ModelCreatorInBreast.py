
import torch
import numpy as np
from torchvision import models, utils, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
import time

batch_size = 128
test_split = 0.25
random_seed = 64325564
epochs_top = 25
epochs = 100
LR_TOP = 0.01
lr = 0.001
momentum = 0.9
decay = 0.0005

class ConvolutionalNeuralNetwork(nn.Module):
    """
    Inherits from nn.Module of pytorch, constructor and forward functions
    """

    def __init__(self, useFilter=False):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # self.filter.assign_weight(4.759330)
        self.useFilter = useFilter

        # Load densenet121 with pretrained weights (using transfer learning)
        self.densenet121 = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them? must confirm
        for param in self.densenet121.parameters():
            param.requires_grad = True
        # two fullyconnected layers of 1024 units
        # 256 units the second one
        # 6 output units for the BI-RADS standard
        self.densenet121.classifier = nn.Sequential(nn.Linear(1024, 256),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.2),
                                                    nn.Linear(256, 6),
                                                    nn.LogSoftmax(dim=1))

    def forward(self, x):
        """
        param x: input observation
        return output: array of output units activation
        """
        output = self.densenet121(x)
        return output


class ModelCreatorInBreast:
    def __init__(self, device, useCuda):
        """
        :param device: Device where to init the model
        """
        self.device = device
        self.useCuda = useCuda
        self.numberClasses = 6

    def getAlexNet(self, isTeacher = False, preTrained = True, trainTopOnly = False):
        """
        :param isTeacher: defines wether is the teacher (with exponential moving average, or not)
        :return:
        """
        alexNetModel = models.alexnet(pretrained = preTrained)
        #print(alexNetModel)
        # Freeze all layers of pretrained model
        if(trainTopOnly):
            print("Train top only, freezing the rest")
            for param in alexNetModel.parameters():
                param.requires_grad = False

        # Reshape last layer, for INBREAST
        alexNetModel.classifier[6] = nn.Linear(in_features = 4096, out_features = self.numberClasses)
        #add a softmax activ. function
        alexNetModel.classifier = nn.Sequential(*list(alexNetModel.classifier) + [nn.Softmax(1)])
        #Put the model on the Device
        alexNetModel.to(self.device)
        #if is teacher, detach the parameters, to allow its averaging
        if isTeacher:
            for param in alexNetModel.parameters():
                param.detach_()


        return alexNetModel
    
    def get_vgg16(self, isTeacher = False, preTrained = True, trainTopOnly = False):
        """
        :param isTeacher: defines wether is the teacher (with exponential moving average, or not)
        :return:
        """
        vgg16 = models.vgg16(pretrained = preTrained)
        print(vgg16)
        # Freeze all layers of pretrained model
        if(trainTopOnly):
            print("Train top only, freezing the rest")
            for param in vgg16.parameters():
                param.requires_grad = False

        # Reshape last layer, for INBREAST
        vgg16.classifier[6] = nn.Linear(in_features = 4096, out_features = self.numberClasses)
        #add a softmax activ. function
        vgg16.classifier = nn.Sequential(*list(vgg16.classifier) + [nn.Softmax(1)])
        #Put the model on the Device
        vgg16.to(self.device)
        #if is teacher, detach the parameters, to allow its averaging
        if isTeacher:
            for param in vgg16.parameters():
                param.detach_()


        return vgg16

    def getOptimizer(self, args, model):
        """
        Creates the optimizer
        :param args: arguments for the optimizer
        :return:
        """
        #optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], args.lr)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov)
        return optimizer


