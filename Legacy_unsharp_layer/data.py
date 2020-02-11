import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from multilabel import XMLMultiLabelDataset
from arguments import args

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#dataset_root = r"/home/jcarranza/Datasets/RAW/PlantCLEF2018"
#dataset_root = r"/Users/josemariocarranza/Dropbox/Datasets/PlantCLEFStandard/PlantCLEF2015/train_separated/"
dataset_root = r"/home/jcarranza/Datasets/RAW/PlantCLEF2015/train"
#dataset_root = r"/home/jcarranza/Datasets/RAW/Herbaria255"
dset = XMLMultiLabelDataset(level_list=["ClassId", "Genus", "Family"], root=dataset_root, transform=data_transforms['train'], root_has_subfolders=False)

dataset_root_test = r"/home/jcarranza/Datasets/RAW/PlantCLEF2015/test_annotations"
dset_test = XMLMultiLabelDataset(level_list=["ClassId", "Genus", "Family"], hierarchy=dset.hierarchy, root=dataset_root_test, transform=data_transforms['val'], root_has_subfolders=False)

valid_size = 0.2
num_train = len(dset)
indices = list(range(num_train))
num_test = len(dset_test)
indices_test = list(range(num_test))

split = int(np.floor(valid_size * num_train))
#np.random.seed(random_seed)
np.random.shuffle(indices)
np.random.shuffle(indices_test)

#train_idx, valid_idx = indices[split:], indices[:split]
train_idx, valid_idx = indices, indices_test
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(dset, pin_memory=True, batch_size=args.train_batch,
                                           sampler=SubsetRandomSampler(train_idx))
dataloaders['val'] = torch.utils.data.DataLoader(dset_test, pin_memory=True, batch_size=args.test_batch,
                                               sampler=SubsetRandomSampler(valid_idx))
