import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls as resnet_urls
from torchvision.models.inception import Inception3, model_urls as inception_urls

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from USM import *

def lenet_usm(pretrained=False, num_classes=1000):
    return LeNetUSM()

class LeNetUSM(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=True, requires_grad=False)
        self.filter.assign_weight(4.759330)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.filter(x)
        out = F.relu(self.conv1(out))
        #out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return(out)

def resnet18_usm(pretrained=False, num_classes=1000):
    return ResNetUSM(pretrained)

class ResNetUSM(ResNet):

    def __init__(self, pretrained=True):
        #super(ResNet, self).__init__( block=BasicBlock, layers=[2, 2, 2, 2] )
        ResNet.__init__(self, block=BasicBlock, layers=[2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(resnet_urls['resnet18']))
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=True, requires_grad=True)
        #self.filter.assign_weight(1.33)
        self.filter_conv1 = USM(in_channels=64, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=True)

    def forward(self, x):
        x = self.filter(x)
        #x = self.filter_conv1(x)
        return super().forward(x)


def inception_v3_usm(pretrained=True, **kwargs):
    return InceptionUSM(pretrained, **kwargs)

class InceptionUSM(Inception3):

    def __init__(self, pretrained=True, num_classes=1000, aux_logits=True, transform_input=False):
        Inception3.__init__(self, num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(inception_urls['inception_v3_google']))
        self.aux_logit=False
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=True, requires_grad=True)

    def forward(self, x):
        x = self.filter(x)
        return super().forward(x)
