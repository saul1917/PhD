
# coding: utf-8

# #Plug in google colab

# In[19]:


get_ipython().system('ls ../INbreast')


# In[20]:


#Interesting example
#https://medium.com/predict/using-pytorch-for-kaggles-famous-dogs-vs-cats-challenge-part-1-preprocessing-and-training-407017e1a10c

# Load the Drive helper and mount
#from google.colab import drive

# This will prompt for authorization.
#drive.mount('../INbrest')
DEFAULT_PATH = "../INbreast"
get_ipython().system('ls ../INbreast')

get_ipython().system('pip install pydicom')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')


# #Dataset loader

# In[ ]:


import torch
import numpy as np
from torchvision import models, utils, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import StratifiedShuffleSplit
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

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class INbreastDataset(Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        self.gt_data = pd.read_csv(csv_path, sep=';')

        print("GT DATA ", self.gt_data)
        filenames = []
        for filename in os.listdir(data_path):
            if ".dcm" in filename.lower():

                print("FILENAME OPENED: ", filename)
                filenames.append(filename)

        

        for fileName in filenames:
          fullpath = os.path.join(data_path, fileName)
          print(fullpath)

        

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
        dc = pydicom.dcmread(self.gt_data.loc[self.gt_data['File Name'] == index].path.item())
        img_as_arr = dc.pixel_array.astype('float64')
        img_as_arr *= (255.0/img_as_arr.max())
        img_as_img = Image.fromarray(img_as_arr.astype('uint8'))
        #img_as_img.save(str(index) + '.png')
        #thresh = threshold_otsu(img_as_img)


        if self.transform is not None:
            img_as_img = self.transform(img_as_img)

        label = self.le.transform(self.gt_data.loc[self.gt_data['File Name'] == index, 'label'].values)[0]
        
        return img_as_img, label

    def getfilenames(self):
        return self.gt_data['File Name'].values

    def getlabels(self):
        return self.le.transform(self.gt_data['label'].values)

    def __len__(self):
        return len(self.gt_data)



def show_batch_sample(x,y):
    grid = utils.make_grid(x, normalize=True, range=(0, 1))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(4):
        plt.scatter(x[i,:,0].numpy(),x[i,:,1].numpy())
        plt.title('Batch from dataloader')



########################## Parameter definition

train_transforms = transforms.Compose([transforms.Grayscale(3),
                                     #transforms.RandomRotation(20), 
                                     transforms.RandomAffine(20,(0.2,0.2),(0.8,1.2),0.2),
                                     transforms.RandomVerticalFlip(),
                                     #transforms.Resize((256,256)),
                                     #transforms.RandomCrop((224,224)),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([
                                     transforms.Grayscale(3),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

dataset_train = INbreastDataset(DEFAULT_PATH + '/AllDICOMs/', DEFAULT_PATH + '/INbreast.csv', train_transforms)
dataset_val = INbreastDataset(DEFAULT_PATH + '/AllDICOMs/', DEFAULT_PATH + '/INbreast.csv', val_transforms)
batch_size = 128
test_split = 0.25
random_seed= 64325564 
epochs_top = 25
epochs = 100
lr_top=0.01
lr=0.001
momentum=0.9
decay = 0.0005


model_ft = models.alexnet(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

do_train_top_model=True
retrain_model=True


np.random.seed(random_seed) 
torch.manual_seed(random_seed)
########################## Getting data

x_indices = dataset_train.getfilenames()
y = dataset_train.getlabels()

print(len(x_indices))
train_indices, val_indices, train_labels, val_labels  = train_test_split(x_indices, y, test_size=test_split, random_state=random_seed)
#sss = StratifiedShuffleSplit(n_splits=5, test_size=test_split)
#for train_idx,test_idx in sss.split(x_indices, y):
#    train_indices, val_indices = x_indices[train_idx], x_indices[test_idx]
#    train_labels, val_labels = y[train_idx], y[test_idx]



#split = int(np.floor(test_split * len(dataset)))
#np.random.shuffle(indices)
#train_indices, val_indices = indices[split:], indices[:split]

top_train_sampler = SubsetSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=top_train_sampler, num_workers=8)
val_loader = DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler, num_workers=8)



########### Use cudnn
cudnn.benchmark = True




########################## Train top model 

#Freeze all layers of pretrained model
for param in model_ft.parameters():
    param.requires_grad = False

#Reshape last layer
model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=6)
            
model_ft.to(device)

top_model_loss = nn.CrossEntropyLoss()
top_model_optimizer = torch.optim.Adam([param for param in model_ft.parameters() if param.requires_grad], lr_top)

def train_top_model(train_loader, device, model, batch_size, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    corrects = 0
    train_mae = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            #Foward
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
        
            _, predicted = outputs.max(1)
            #Backward
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(predicted == targets.data)
        train_mae += torch.dist(targets.data.float(), predicted.float(), 1).item() * inputs.size(0)    
   
    epoch_acc = corrects.double() / len(train_loader.sampler)   
    epoch_loss = train_loss / len(train_loader.sampler)
    epoch_mae = train_mae / len(train_loader.sampler)

    print('Train Epoch: {} \tLoss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch+1, epoch_loss, epoch_acc, corrects, len(train_loader.sampler), epoch_mae ))   

    torch.save(model_ft.state_dict(),'./top_train_weights.data')

    return epoch_loss,epoch_acc,epoch_mae


def val_top_model(val_loader, device, model, batch_size, criterion, optimizer, epoch):
    model.eval()
    val_loss = 0
    corrects = 0
    val_mae = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, targets)        
        
            _, predicted = outputs.max(1)
        
        val_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(predicted == targets.data)
        val_mae  += torch.dist(targets.data.float(), predicted.float(), 1).item() * inputs.size(0)

    epoch_loss = val_loss / len(val_loader.sampler)
    epoch_acc = corrects.double() / len(val_loader.sampler)
    epoch_mae = val_mae / len(val_loader.sampler)
  
    print('Val Epoch: {} \tLoss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch+1, epoch_loss, epoch_acc, corrects, len(val_loader.sampler), epoch_mae ))

    return epoch_loss,epoch_acc,epoch_mae

#Sample dataset
'''for i_batch, (x,y) in enumerate(train_loader):

    # observe 4th batch and stop.
    if i_batch == 2:
        #plt.figure()
        #show_batch_sample(x,y)
        #plt.axis('off')
        #plt.ioff()
        #plt.savefig('sample.png')
        utils.save_image(x, 'sample.png', normalize=True)
        break
'''
if do_train_top_model:
    train_top_loss_hist = []
    val_top_loss_hist = []
    train_top_acc_hist = []
    val_top_acc_hist = []
    train_top_mae_hist = []
    val_top_mae_hist = []

    print('Top model training!')
    for epoch in range(0,epochs_top):
        train_loss,train_acc,train_mae = train_top_model(train_loader, device, model_ft, batch_size, top_model_loss, top_model_optimizer, epoch)
        val_loss,val_acc,val_mae = val_top_model(val_loader, device, model_ft, batch_size, top_model_loss, top_model_optimizer, epoch)
        train_top_loss_hist.append(train_loss)
        val_top_loss_hist.append(val_loss)
        train_top_acc_hist.append(train_acc.item())
        val_top_acc_hist.append(val_acc.item())
        train_top_mae_hist.append(train_mae)
        val_top_mae_hist.append(val_mae)

    d = {'train_loss':train_top_loss_hist, 'train_acc':train_top_acc_hist, 'train_mae':train_top_mae_hist, 'val_loss':val_top_loss_hist, 'val_acc':val_top_acc_hist, 'val_mae':val_top_mae_hist}

    if not os.path.isdir('results'):
        os.mkdir('results')

    df = pd.DataFrame.from_dict(d)
    df.to_csv('./results/top_training.txt', sep=',', encoding='utf-8')
    


########################## Retraining

trainable_layers = {'features': ["{}".format(n) for n in range(6,13)], 'classifier': ["{}".format(n) for n in range(0,7)]}


class fullModel(nn.Module):
       def __init__(self, model, model_weights, trainable_layers, with_usm):
            super(fullModel, self).__init__()
            #self.with_usm = with_usm
            #Layers
            #if self.with_usm:
               #self.usm=USM(in_channels=3, kernel_size=3, fixed_coeff=True, sigma=0.2, cuda=True)
            model.load_state_dict(torch.load(model_weights))
            self.model = torch.nn.Sequential(model)
            for name, child in list(self.model.children())[0].features.named_children():
                if name in trainable_layers['features']:
                    print("trainable ft %d",name)
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False
            for name, child in list(self.model.children())[0].classifier.named_children():
                if name in trainable_layers['classifier']:
                    print("trainable fc %d",name)
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                       param.requires_grad = False
       def forward(self, x):
            #if self.with_usm:
                #x = self.usm(x) 
            x = self.model.forward(x)
            return x


full_model = fullModel(model_ft, './top_train_weights.data', trainable_layers, False)

print(full_model)

full_model.to(device)

#retraining_loss = nn.CrossEntropyLoss(class_weights)
retraining_loss = nn.CrossEntropyLoss()

retraining_optimizer = torch.optim.Adam([param for param in full_model.parameters() if param.requires_grad],
                       lr)

def train_full_model(train_loader, device, full_model, batch_size, criterion, optimizer, epoch):
    full_model.train()
    train_loss = 0
    corrects = 0
    train_mae = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = full_model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)

            loss.backward()
            optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_mae  += torch.dist(targets.data.float(), predicted.float(), 1).item() * inputs.size(0)       
        corrects += torch.sum(predicted == targets.data)

    epoch_loss = train_loss / len(train_loader.sampler)
    epoch_acc = corrects.double() / len(train_loader.sampler)        
    epoch_mae = train_mae / len(train_loader.sampler)

    print('Train Epoch: {} \tLoss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch+1, epoch_loss, epoch_acc, corrects, len(train_loader.sampler), epoch_mae ))

    torch.save(full_model.state_dict(),'./retraining_weights_nousm.data')

    return epoch_loss,epoch_acc,epoch_mae

def val_full_model(val_loader,  device, full_model, batch_size, criterion, optimizer, epoch):
    full_model.eval()
    val_loss = 0
    corrects = 0 
    val_mae = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = full_model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
 
        val_loss += loss.item() * inputs.size(0)
        val_mae  += torch.dist(targets.data.float(), predicted.float(), 1).item() * inputs.size(0)
        corrects += torch.sum(predicted == targets.data)

    epoch_loss = val_loss / len(val_loader.sampler)
    epoch_acc = corrects.double() / len(val_loader.sampler)
    epoch_mae = val_mae / len(val_loader.sampler)

    print('Val Epoch: {} \tLoss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch+1, epoch_loss, epoch_acc, corrects, len(val_loader.sampler), epoch_mae )) 
    
    return epoch_loss,epoch_acc,epoch_mae


full_train_sampler = SubsetRandomSampler(train_indices)
full_val_sampler = SubsetRandomSampler(val_indices)

full_train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=full_train_sampler, num_workers=8)
full_val_loader = DataLoader(dataset_val, batch_size=batch_size, sampler=full_val_sampler, num_workers=8)


if retrain_model==True: 
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(retraining_optimizer, mode='min', factor=0.1, patience=10, verbose=1, threshold=0.0001, cooldown=5, min_lr=0.00001)
    full_train_loss_hist = []
    full_val_loss_hist = []
    full_train_acc_hist = []
    full_val_acc_hist = []
    full_train_mae_hist = []
    full_val_mae_hist = []
    #alpha_hist = []
    epoch_times = []

    best_weights = copy.deepcopy(full_model.state_dict())
    best_acc = 0

    print('Full model training!')
    for epoch in range(0,epochs):
        start = time.time()
        train_loss,train_acc,train_mae = train_full_model(full_train_loader, device, full_model, batch_size, top_model_loss, retraining_optimizer, epoch)
        val_loss,val_acc, val_mae = val_full_model(full_val_loader, device, full_model, batch_size, top_model_loss, retraining_optimizer, epoch)
        elapsed = time.time() - start
        epoch_times.append(elapsed)
        full_train_loss_hist.append(train_loss)
        full_val_loss_hist.append(val_loss)
        full_train_acc_hist.append(train_acc.item())
        full_val_acc_hist.append(val_acc.item())
        full_train_mae_hist.append(train_mae)
        full_val_mae_hist.append(val_mae)
        #alpha_hist.append(full_model.usm.alpha.item())
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(full_model.state_dict())
        #scheduler.step(val_loss)
        print('Average time for each epoch:{}'.format(sum(epoch_times)/(epoch+1)))

    d = {'train_loss':full_train_loss_hist, 'train_acc':full_train_acc_hist, 'train_mae':full_train_mae_hist, 'val_loss':full_val_loss_hist, 'val_acc':full_val_acc_hist, 'val_mae':full_val_mae_hist}

    if not os.path.isdir('results'):
        os.mkdir('results')

    df = pd.DataFrame.from_dict(d)
    df.to_csv('./results/retraining.txt', sep=',', encoding='utf-8')

    torch.save(best_weights,'./retraining_weights.data')

