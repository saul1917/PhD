#Interesting example
#https://medium.com/predict/using-pytorch-for-kaggles-famous-dogs-vs-cats-challenge-part-1-preprocessing-and-training-407017e1a10c

# Load the Drive helper and mount
#from google.colab import drive

# This will prompt for authorization.
#drive.mount('../INbrest')
DEFAULT_PATH = "/media/Data/saul/InBreastDataset"
NUMBER_CLASSES = 6
import torch
import numpy as np
from torchvision import models, utils, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import pydicom
import os
import time
import re
import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def getTransformationsInBreast():
    """
    Get the transformations for the dataset
    :return:
    """
    meanDatasetComplete = torch.tensor([0.1779, 0.1779, 0.1779])
    stdDatasetComplete =  torch.tensor([0.2539, 0.2539, 0.2539])

    trainTransformations = transforms.Compose([transforms.Grayscale(3),
                                                transforms.RandomRotation(20),
                                               transforms.RandomAffine(20, (0.2, 0.2), (0.8, 1.2), 0.2),
                                               transforms.RandomVerticalFlip(),
                                                transforms.Resize((256,256)),
                                                transforms.RandomCrop((224,224)),
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)
                                               ])
    #Normalize must be last
    # CHECK NORMALIZATION !!!!!!!!
    # Normalize dataset
    validationTransformations = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meanDatasetComplete, std=stdDatasetComplete)
        ])
    # Normalize must be last
    return (trainTransformations, validationTransformations)

class SubsetSampler(Sampler):
    """
    Allows scipy k folds to iterate through
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        """
        Returns the iterator
        :return:
        """
        return iter(self.indices)

    def __len__(self):
        """
        Returns the partition length (subset)
        :return:
        """
        return len(self.indices)

def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    :param dataset:
    :return:
    """
    xIndices = dataset.getfilenames()
    sampler_training = SubsetRandomSampler(xIndices)
    batch_sampler_training = BatchSampler(sampler_training, batch_size = 1, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_training, num_workers= 5, pin_memory=True)

    #init the mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    k = 1
    for inputs, targets in data_loader:
        #mean and std from the image
        #print("Processing image: ", k)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
        k += 1

    #normalize
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean: " + str(mean))
    print("std: " + str(std))
    return mean, std

"Dataset loader for the InBreast dataset"
class INbreastDataset(Dataset):
    """
    INbresat dataset
    """
    def __init__(self, data_path, csv_path, useOneHotVector = False, transform = None):
        """
        Class constructor
        :param data_path: data path,
        :param csv_path: csv contains the labels
        :param transform:  transforms to use
        """
        self.useOneHotVector = useOneHotVector
        self.gt_data = pd.read_csv(csv_path, sep=';')
        filenames = []
        for filename in os.listdir(data_path):
            if ".dcm" in filename.lower():
                filenames.append(filename)
        self.gt_data['path']= self.gt_data["File Name"].astype(str).map(lambda x: os.path.join(data_path, list([filename for filename in filenames if x in filename])[0]))
        self.gt_data['exists'] = self.gt_data['path'].map(os.path.exists)
        #Delete gt data with no corresponding image
        if len(self.gt_data[self.gt_data.exists == False]) != 0:
            for index,row in self.gt_data.iterrows():
                if row['exists'] != True:
                    print('WARNING: ground truth value ' + row['id'] +
                      ' has no corresponding image! This ground truth value will be deleted')
                    self.gt_data.drop(index, inplace=True)
        #Get labels
        self.gt_data['label'] = self.gt_data['Bi-Rads'].map(lambda x: re.sub('[^0-9]','', x)).astype(int)
        self.le = LabelEncoder()
        self.le.fit(self.gt_data['label'].values)
        self.categories = self.le.classes_
        self.transform = transform

    def __getitem__(self, index):
        """
        Get item
        :param index:
        :return:
        """

        element = self.gt_data.loc[self.gt_data['File Name'] == index]
        #print("ELEM ", element)
        file_name = element.path.item()
        #print("FILE TO LOAD ", file_name)
        dc = pydicom.dcmread(file_name)
        img_as_arr = dc.pixel_array.astype('float64')
        img_as_arr *= (255.0/img_as_arr.max())
        img_as_img = Image.fromarray(img_as_arr.astype('uint8'))
        #img_as_img.save(str(index) + '.png')
        #thresh = threshold_otsu(img_as_img)
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)

        label = self.le.transform(self.gt_data.loc[self.gt_data['File Name'] == index, 'label'].values)[0]
        #check wether to use one hot vector notation
        if(self.useOneHotVector):
            label = self.toOneHotVector(label)

        return img_as_img, label

    def toOneHotVector(self, target):
        """
        Takes the target and translates it into a one hot vector notation
        :param target:
        :return:
        """
        target = torch.tensor([[target]])
        yOnehot = torch.LongTensor(1, NUMBER_CLASSES)
        yOnehot.zero_()
        yOnehot.scatter_(1, target, 1)
        yOnehotN = yOnehot[0]
        return yOnehotN

    def toNumericalLabel(self, oneHotVector):
        """
        Transform one hot vector to numerical label
        :param oneHotVector:
        :return:
        """
        return oneHotVector.argmax(dim=1, keepdim=True)


    def getfilenames(self):
        """
        Get all the file names
        :return:
        """
        return self.gt_data['File Name'].values

    def getlabels(self):
        """
        Get all the labels
        :return:
        """
        return self.le.transform(self.gt_data['label'].values)

    def __len__(self):
        """
        Length of the dataset
        :return:
        """
        return len(self.gt_data)



