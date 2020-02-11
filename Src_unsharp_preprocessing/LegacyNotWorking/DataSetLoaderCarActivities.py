import pandas as pd
import numpy as np
from PIL import Image
import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class DatasetCarsActivities(Dataset):
    def __init__(self, samplesDirectory, transform = None):
        """
        Inits the dataset
        :param samplesDirectory: path to the directory with the samples
        :param transform: Pytorch transforms list
        """
        self.transform = transform
        (sampleIds, labels) = self.fillSampleIdsAndLabels(samplesDirectory)
        self.labels = labels
        self.sampleIds = sampleIds


    def __getitem__(self, index):
        """
        :param index: dataset sample index
        :return: (sample, label)
        """
        samplePath = self.sampleIds[index]
        # Open the image
        imagePIL = Image.open(samplePath)
        #apply transformations
        if self.transform is not None:
            img_as_tensor2 = self.transform(imagePIL)
        #read the label
        y = self.labels[index]
        #transform label to numpy array
        #y = np.array([y]);
        return (img_as_tensor2, y)

    def __len__(self):
        """
        Returns the length of the dataset
        :return: number of samples
        """
        numSamples = len(self.sampleIds)
        return numSamples


    def fillSampleIdsAndLabels(self, samplesDirectory, extension = ".jpg"):
        """
        Extracts sample and label metadata
        :param samplesDirectory: directory of samples
        :param extension, extension of
        :return: (file names, label array)
        """
        files = []
        labels = []
        # r=root, d=directories, f = files
        #one root per subdirectory, or class in this case
        previousRoot = ''
        #current label number
        labelNumber = 0
        for r, d, f in os.walk(samplesDirectory):

            for file in f:
                if extension in file:
                    #get the path of the sample
                    samplePath = os.path.join(r, file)
                    files.append(samplePath)
                    #update the label, according to the subdirectory
                    if(r != previousRoot):
                        labelNumber += 1
                        previousRoot = r
                    #append the label to the list of labels
                    labels += [labelNumber]
        return (files, labels)
    
    
def pruebaDataset():
    #set of transformations to run
    #Resize and transform to tensor

    print("TESTS ARE RUNNING NOW")
    transformations = transforms.Compose([transforms.transforms.Resize((256, 256)), transforms.ToTensor() ])
    datasetCarActivities = DatasetCarsActivities("C:/Users/saul1/Desktop/Datasets_TEMP/auc.distracted.driver.dataset_v2/v1_cam1_no_split", transformations)

    
    
    datasetLoader = torch.utils.data.DataLoader(dataset = datasetCarActivities, batch_size = 12, shuffle = True)

    for i, (images, labels) in enumerate(datasetLoader):
        #takes only one batch of 12 observations
        images = Variable(images)
        labels = Variable(labels)
        print("Dimensions of IMAGES and LABELS")
        print(images.shape)
        print(labels.shape)
        break

    print("LABELS ")
    print(labels)
