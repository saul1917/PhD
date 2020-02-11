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
import matplotlib.pyplot as plt
from PIL import Image

archivoPesos = "dogsCats_weights1"


"""
Clase que hereda del paquete neural net de pytorch
"""


class ConvolutionalNeuralNetwork(nn.Module):


    def __init__(self, numberClasses):
        """
        Class constructor
        :param the number of classes to estimate
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        #load the pretrained weights from imagenet
        res50_model = models.resnet50(pretrained = True)
        self.res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
        #train all the layers
        for param in self.res50_conv.parameters():
            param.requires_grad = True


        self.dropOutConvolucion = nn.Dropout2d()

        # crea una transformacion lineal de la forma y = x A^t + b (sin funcion de activacion)
        # A tiene 320 filas y 50 columnas, x es la entrada de 1 fila y 320 columnas
        # Modelo completamente conectado

        # la salida del promediado global es 2048 promedios de los 2048 feature maps que genera resnet
        self.capaCompletamenteConectada1 = nn.Linear(2048, numberClasses)
        # 2 clases, perro o gato

    """
    Pasada hacia adelante, sobrecargado de la clase nn neural network
    @param x, muestra a estimar su salida
    """

    def forward(self, x):
        # Apila las capas
        # Convolucion1 -> maxPooling 2x2 (se escoge el maximo en una ventana de 2x2)

        x = self.res50_conv(x)

        # print("LAYER CONTENT")

        # outputs 2048 activation maps of 8x8
        # https://resources.wolframcloud.com/NeuralNetRepository/resources/ResNet-50-Trained-on-ImageNet-Competition-Data

        # print("RESNET 50 OUTPUT")
        # print(xResnet.shape)

        # print("Dims to do average")
        # print(xResnet.size()[2:])

        x = F.avg_pool2d(x, x.size()[2:])
        # x dims torch.Size([15, 2048, 1, 1])

        # print("AFTER AVERAGE OUTPUT")
        # print(x.shape)

        x = x.view(-1, 2048)

        # print("AFTER flattening")
        # print(x.shape)

        # Conv1 -> mP -> Conv2 -> DropOutConv2 -> MaxPool -> Relu -> FC1 -> ReLU
        x = F.relu(self.capaCompletamenteConectada1(x))
        # Conv1 -> mP -> Conv2 -> DropOutConv2 -> MaxPool -> Relu -> FC1 -> ReLU -> Dropout
        x = F.dropout(x, training=self.training)
        # Conv1 -> mP -> Conv2 -> DropOutConv2 -> MaxPool -> Relu -> FC1 -> ReLU -> Dropout -> FC2 

        output = F.log_softmax(x, dim=1)
        print("Output dims")
        print(output.shape)

        return output