# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms
import torch
from . import data
from .utils import export


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


@export
def inbreast_full():
    meanDatasetComplete = torch.tensor([0.1779, 0.1779, 0.1779])
    stdDatasetComplete = torch.tensor([0.2539, 0.2539, 0.2539])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(20, (0.2, 0.2), (0.8, 1.2), 0.2),
        transforms.RandomVerticalFlip(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)

    ]))
    eval_transformation = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '/media/Data/saul/Datasets/Inbreast_folder_per_class',
        'num_classes': 6
    }


@export
def inbreast_binary():
    meanDatasetComplete = torch.tensor([0.1779, 0.1779, 0.1779])
    stdDatasetComplete = torch.tensor([0.2539, 0.2539, 0.2539])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(20, (0.2, 0.2), (0.8, 1.2), 0.2),
        transforms.RandomVerticalFlip(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)

    ]))
    eval_transformation = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '/media/Data/saul/Datasets/Inbreast_folder_per_class_binary',
        'num_classes': 2
    }