# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
from __future__ import print_function
import re
import argparse

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, utils, transforms
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

useCuda = torch.cuda.is_available()
device = torch.device("cuda" if useCuda else "cpu")
print("Device to use:")
print(device)

DEFAULT_LEARNING_RATE = 0.00001
DEFAULT_MOMENTUM = 0.5

class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        features = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = torch.nn.Softmax(x)
        #CORREGIR! NO ES LOG SOFTMAX, ES SOFTMAX!
        output = F.log_softmax(x, dim=1)
        #for Softmax output
        output = output.exp()
        return (output, features)

trainTransformations = transforms.Compose([transforms.Grayscale(1), transforms.RandomVerticalFlip(), transforms.Resize((32, 32)), transforms.ToTensor()])
data_train = datasets.ImageFolder('/media/Data/saul/Datasets/Inbreast_folder_per_class_binary/', transform= trainTransformations)

data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=8)


lenet = LeNetFeatureExtractor().to(device)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(lenet.parameters(), lr=2e-6)
#optimizer = torch.optim.SGD(lenet.parameters(), lr=DEFAULT_LEARNING_RATE, momentum = DEFAULT_MOMENTUM, weight_decay = 1e-7, nesterov = True)


def train(epoch):
    global cur_batch_win
    lenet.train()
    loss_list, batch_list = [], []
    total_loss = 0
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)
        #labels = labels.view(1, -1)
        #print("TARGET")
        #print(labels.shape)


        optimizer.zero_grad()
        (output, _) = lenet(images)
        #print("Prediction")
        #print(output.shape)

        loss = criterion(output, labels)
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)
        loss_batch = loss.detach().cpu().item()

        # Update Visualization
        total_loss +=  loss_batch
        loss.backward()
        optimizer.step()
    print('Train - Epoch %d, Total Loss: %f' % (epoch, total_loss))


def train_and_test(epoch):
    train(epoch)

def toNumericalLabel( oneHotVector):
    """
    Transform one hot vector to numerical label
    :param oneHotVector:
    :return:
    """
    return oneHotVector.argmax(dim=1, keepdim=True)

def main():
    for e in range(1, 300):
        train_and_test(e)                 
    

main()
    
