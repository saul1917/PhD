from __future__ import print_function
import argparse
#Pytorch library import
import torch
import torch.nn as nn




import torch.nn.functional as F
import torch.optim as optim
import numpy as np;
import pandas as pandas;
from scipy import ndimage
from torchvision import datasets, transforms, models
import torch.nn.functional as F


import os
#Para corregir error en depuracion usando pycharm, hacer
"""
Look for the Anaconda directory and set the Library\plugins subdir 
(here c:\ProgramData\Anaconda3\Library\plugins) as environment variable 
QT_PLUGIN_PATH
"""
"""
Clase que hereda del paquete neural net de pytorch
"""


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Convolutional neural network class
    """

    def __init__(self, numberClasses):
        """
        Class constructor
        :param numberClasses: number of output units
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.densenet121 = models.densenet121(pretrained = True)

        # Freeze parameters so we don't backprop through them
        for param in self.densenet121.parameters():
            param.requires_grad = True
        #N output classes
        self.densenet121.classifier = nn.Sequential(nn.Linear(1024, 256),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.2),
                                                    nn.Linear(256, 11),
                                                    nn.LogSoftmax(dim=1))

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input tensor
        :return: output units tensor
        """
        output = self.densenet121(x)
        return output

