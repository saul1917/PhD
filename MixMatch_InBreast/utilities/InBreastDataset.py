DEFAULT_PATH = "/media/Data/saul/InBreastDataset"
NUMBER_CLASSES = 6

import torch
import torchvision
from shutil import copy
from shutil import copy2
from shutil import copyfile
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
#import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import ntpath


#Dataset partitioner

def create_train_test_folder_partitions(datasetpath_base, percentage_evaluation = 0.25, random_state = 42, batch = 0, create_dirs = True):
    """

    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches:
    :param create_dirs:
    :param percentage_evaluation:
    :return:
    """
    datasetpath_test = datasetpath_base + "/batch_" + str(batch) + "/test/"
    datasetpath_train = datasetpath_base +  "/batch_" + str(batch) + "/train/"
    datasetpath_all = datasetpath_base + "/all"
    print("datasetpath_all")
    print(datasetpath_all)
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    #list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]



    if (create_dirs):
        # create the directories
        print("Trying to create dir")
        print(datasetpath_test)
        os.makedirs(datasetpath_test)
        print(datasetpath_test)
        os.makedirs(datasetpath_train)
        for i in range(0,6):
            os.makedirs(datasetpath_test + "/" + str(i))
            os.makedirs(datasetpath_train + "/" + str(i))


    # test and train  splitter for unlabeled and labeled data split

    X_train, X_test, y_train, y_test = train_test_split(list_file_names, list_labels, test_size = percentage_evaluation, random_state = random_state)
    print("TRAINING DATA-----------------------", len(X_train))
    for i  in range(0, len(X_train)):
        #print(X_train[i] + " LABEL: " + str(y_train[i]))
        path_src = X_train[i]
        #extract the file name
        file_name = ntpath.basename(path_src)
        #print("File name", file_name)
        path_dest = datasetpath_train + str(y_train[i]) +"/" + file_name
        #print("COPY TO: " + path_dest)
        copy2(path_src, path_dest)

    print("EVALUATION DATA-----------------------", len(X_test))
    for i  in range(0, len(X_test)):
        #print(X_test[i] + " LABEL: " + str(y_test[i]))
        path_src = X_test[i]
        file_name = ntpath.basename(path_src)
        #print("File name", file_name)

        path_dest = datasetpath_test + str(y_test[i])+ "/" + file_name
        #print("COPY TO: " + path_dest)
        copy2(path_src, path_dest)

    #store data

def create_partitions_multi_class():
    random_state_base = 42
    datasetpath_base = "/media/Data/saul/Datasets/Inbreast_folder_per_class_all"
    for i in range(0, 10):
        random_state_base += 1
        create_train_test_folder_partitions(datasetpath_base, percentage_evaluation=0.25,
                                            random_state=random_state_base, batch=i)

def create_partitions_binary():
    random_state_base = 42
    datasetpath_base = "/media/Data/saul/Datasets/Inbreast_folder_per_class_binary"
    for i in range(0, 10):
        random_state_base += 1
        create_train_test_folder_partitions(datasetpath_base, percentage_evaluation=0.25,
                                            random_state=random_state_base, batch=i)




if __name__ == '__main__':
    create_partitions_binary()