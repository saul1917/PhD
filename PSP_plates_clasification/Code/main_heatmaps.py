#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import cv2
import copy
import click
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from utils import load_model, str2bool
from torch.autograd import Variable
from torchvision import models, transforms
from models import get_cnn, get_optimizer
from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

# if model has LSTM
# torch.backends.cudnn.enabled = False


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',   default='resnet18',
                            choices=['resnet18', 'resnet50', 'vgg19_bn','inception_v3',
                            'resnetusm'],
                            help='The network architecture')
    parser.add_argument('--image-path', default="",
                            help='The folder where the image is')
    parser.add_argument('--topk', type=int, default = 2,
                            help='How many clases you want to see')
    parser.add_argument('--cuda', default=True,  type=str2bool)
    parser.add_argument('--pretrained', type=str2bool, default=True,
                            help='The model will be pretrained with ')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                            help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weights', default="",
                            help='The .pth doc to load as weights')
    return parser


def main():
    parser = make_parser()                                                      #creates the parser
    args = parser.parse_args()
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'resnet50': {
            'target_layer': 'layer4',
            'input_size': 224
        },
        'resnet18': {
            'target_layer': 'layer4',
            'input_size': 224
        },
        # Add your model
    }.get(args.arch)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if args.cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # Synset words
    classes = list()
    with open('samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)

    # Model
    print("1")
    class_names = ['no', 'yes']                                         #important to define the classes for prediction
    model_ft = get_cnn(len(class_names), args)                          #retrieves the cnn - architecture to be used
    print("2")
    criterion = nn.CrossEntropyLoss()                                   #creates the criterion (used in training and testing)
    optimizer_ft = get_optimizer(model_ft, args)                        #changes the weights based on error (using Stochastic Gradient Descent)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#helps with the learning rate, to be zigzagging to get into the right function
    model = load_model(model_ft, args.weights)               #load the model with weights
    print("3")
    model = model.to(device)
    model.eval()

    # Image
    print("image path: "+ str(args.image_path))
    raw_image = cv2.imread(args.image_path)[..., ::-1]

    raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)

    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image.to(device))

    for i in range(0, args.topk):
        print("idx is: " + str(idx))
        print("idx is: " + str(probs))
        print("i is: " + str(i))
        gcam.backward(idx=idx[i])
        print("idx AFTER is: " + str(idx))
        print("idx[i]: " + str(idx[i]))
        output = gcam.generate(target_layer=CONFIG['target_layer'])
        print("classes[idx[i]]: " + str(classes[idx[i]]))
        save_gradcam('results/{}_gcam_{}.png'.format(classes[idx[i]], args.arch), output, raw_image)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Vanilla Backpropagation')
    # =========================================================================
    bp = BackPropagation(model=model)
    probs, idx = bp.forward(image.to(device))

    for i in range(0, args.topk):
        bp.backward(idx=idx[i])
        output = bp.generate()

        save_gradient('results/{}_bp_{}.png'.format(classes[idx[i]], args.arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Deconvolution')
    # =========================================================================
    deconv = Deconvolution(model=copy.deepcopy(model))  # TODO: remove hook func in advance
    probs, idx = deconv.forward(image.to(device))

    for i in range(0, args.topk):
        deconv.backward(idx=idx[i])
        output = deconv.generate()

        save_gradient('results/{}_deconv_{}.png'.format(classes[idx[i]], args.arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Guided Backpropagation/Guided Grad-CAM')
    # =========================================================================
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image.to(device))

    for i in range(0, args.topk):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=CONFIG['target_layer'])

        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        h, w, _ = feature.shape
        region = cv2.resize(region, (w, h))[..., np.newaxis]
        output = feature * region

        save_gradient('results/{}_gbp_{}.png'.format(classes[idx[i]], args.arch), feature)
        save_gradient('results/{}_ggcam_{}.png'.format(classes[idx[i]], args.arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
