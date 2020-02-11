from __future__ import print_function, division
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from ResNetUSM import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
k_fold_validation = 0
arch_ = ""
def eval_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler):
    phase = 'val'
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

    if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_acc


def train_model(dataloaders, dataset_sizes, name_pth, model, criterion, optimizer, scheduler, num_epochs=25, csv_name="results"):
    global k_fold_validation
    global arch_
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with open(csv_name+'_total.csv', 'w', newline='') as csvfile:
        fieldnames = ["k_cross_validation", "epoch", "phase",
        "acc", "loss", "alpha"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #print("phase that is: " + phase)
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    ##en val si entra aqui pero no aplica el step ni el backguard
                    if(phase == 'train' and arch_=="inception_v3"):
                        outputs, aux = model(inputs)
                    else:
                        outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #print("dentro train para hacer loss.backward y optimizer.step")
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            with open(csv_name+'_total.csv', 'a', newline='') as csvfile:
                csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if(arch_ == "resnetusm"):
                    csvwriter.writerow({'k_cross_validation': k_fold_validation, 'epoch': epoch, 'phase': phase, 'acc':epoch_acc.item(), 'loss':epoch_loss, 'alpha':model.filter.alpha.item()})
                elif(arch_ == "resnet" or arch_ == "vgg19_bn" or arch_ == "inception_v3"):
                    csvwriter.writerow({'k_cross_validation': k_fold_validation, 'epoch': epoch, 'phase': phase, 'acc':epoch_acc.item(), 'loss':epoch_loss, 'alpha':0})
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    torch.save(model.state_dict(), "./models/" + name_pth+".pth")
    return model, best_acc


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#lo que quiero hacer es guardar los resultados de visualizar el modelo en un csv
def csv_val_model(class_names, dataloaders, model, csv_name="results"):
    was_training = model.training
    print("was training: " + str(was_training))

    model.eval()
    with open(csv_name+'.csv', 'w', newline='') as csvfile:
        fieldnames = ["Expected_label", "Given_label"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs, preds = torch.max(outputs, 1)
            label = ""
            # segun lo que entiendo aqui hace todos y solo toma 6
            for j in range(inputs.size()[0]):
                with open(csv_name+'.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if(labels.cpu().data[j].item() == 1):
                        label = "yes"
                    else:
                        label = "no"
                    csvwriter.writerow({'Expected_label': label, 'Given_label': class_names[preds[j]]})
        model.train(mode=was_training)
    print("ok?")


def load_model(dataloaders, dataset_sizes, name_pth, model, path, criterion, optimizer, scheduler, always_train = False, num_epochs=25):
    # Checks is a model pre-exists in the giving path, .pth file if not call the training
    print(always_train)
    if (os.path.exists(path) and always_train==False):
        for fname in os.listdir(path):
            print(fname)
            if fname.endswith('.pth'):
                # do stuff on the file
                print("entra a esta cosa ")
                src_file = os.path.join(path, fname)
                print(src_file)
                model.load_state_dict(torch.load(src_file))
                best_acc = eval_model(dataloaders, dataset_sizes, model, criterion, optimizer,  scheduler)
                return model, best_acc

    model, best_acc = train_model(dataloaders, dataset_sizes, name_pth, model, criterion, optimizer, scheduler,
                          num_epochs=25)

    best_acc = eval_model(dataloaders, dataset_sizes, model, criterion, optimizer,  scheduler)
    best_acc = eval_model(dataloaders, dataset_sizes, model, criterion, optimizer,  scheduler)
    return model, best_acc

def model_generation_weights(arch, name_pth):
    global arch_
    arch_ = arch
    if(arch.startswith("resnet") or arch.startswith("inception")):
        # Load the pretrained model from pytorch
        if(arch == "resnet50"):
            model_ft = models.resnet50(pretrained=True)
        elif(arch == "resnet18"):
            model_ft = models.resnet18(pretrained=True)
        elif(arch == "resnetusm"):
            model_ft = resnet18_usm(pretrained=True, cuda=use_cuda)
        elif(arch == "inception_v3"):
            model_ft = models.inception_v3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
    elif(arch == "vgg19_bn"):
        # Load the pretrained model from pytorch
        model_ft =  models.vgg19_bn(pretrained=True)
        # Newly created modules have require_grad=True by default
        num_features = model_ft.classifier[6].in_features
        features = list(model_ft.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
        model_ft.classifier = nn.Sequential(*features) # Replace the model classifier
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    lr = 0.001
    if(arch == "resnetusm"):
        optimizer_ft = torch.optim.SGD([{'params':model_ft.filter.parameters(), 'lr':lr*10},
                                    {'params':model_ft.filter_conv1.parameters(), 'lr':lr*10},
                                    {'params': model_ft.conv1.parameters()},
                                    {'params': model_ft.layer1.parameters()},
                                    {'params': model_ft.layer2.parameters()},
                                    {'params': model_ft.layer3.parameters()},
                                    {'params': model_ft.layer4.parameters()},
                                    {'params': model_ft.fc.parameters()}],
                                      lr=lr,
                                      momentum=0.9)
    else:
        optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay  Learning rate(LR) by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    _, best_acc = load_model(dataloaders, dataset_sizes, name_pth, model_ft, "./models/", criterion, optimizer_ft, exp_lr_scheduler, always_train,
                           num_epochs=5)
    # do stuff if a file .true doesn't exist.
    #
    return model_ft, best_acc
######################################################################


def model_generation(arch, dataloaders, dataset_sizes, name_pth, always_train=False):
    global arch_
    arch_ = arch
    if(arch.startswith("resnet") or arch.startswith("inception")):
        # Load the pretrained model from pytorch
        if(arch == "resnet50"):
            model_ft = models.resnet50(pretrained=True)
        elif(arch == "resnet18"):
            model_ft = models.resnet18(pretrained=True)
        elif(arch == "resnetusm"):
            model_ft = resnet18_usm(pretrained=True, cuda=use_cuda)
        elif(arch == "inception_v3"):
            model_ft = models.inception_v3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
    elif(arch == "vgg19_bn"):
        # Load the pretrained model from pytorch
        model_ft =  models.vgg19_bn(pretrained=True)
        # Newly created modules have require_grad=True by default
        num_features = model_ft.classifier[6].in_features
        features = list(model_ft.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
        model_ft.classifier = nn.Sequential(*features) # Replace the model classifier
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    lr = 0.001
    if(arch == "resnetusm"):
        optimizer_ft = torch.optim.SGD([{'params':model_ft.filter.parameters(), 'lr':lr*10},
                                    {'params':model_ft.filter_conv1.parameters(), 'lr':lr*10},
                                    {'params': model_ft.conv1.parameters()},
                                    {'params': model_ft.layer1.parameters()},
                                    {'params': model_ft.layer2.parameters()},
                                    {'params': model_ft.layer3.parameters()},
                                    {'params': model_ft.layer4.parameters()},
                                    {'params': model_ft.fc.parameters()}],
                                      lr=lr,
                                      momentum=0.9)
    else:
        optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay  Learning rate(LR) by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    _, best_acc = load_model(dataloaders, dataset_sizes, name_pth, model_ft, "./models/", criterion, optimizer_ft, exp_lr_scheduler, always_train,
                           num_epochs=5)
    # do stuff if a file .true doesn't exist.
    #
    return model_ft, best_acc
######################################################################
#



def generator_sampling_random(k_cross_validation, indices, random_seed, shuffle_dataset = True):
    validation_split = .3
    dataset_size = len(indices)
    for i in range(k_cross_validation):
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed+(10*i))
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        yield train_indices, val_indices


def main(type="k-fold"):
    global k_fold_validation
    name_pth = "usmresnet_25epo224_2900usmresnet_25epo224_2900norotsigma2kernel5range051_results_total"
    size = (224,224)
    data_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir =  'mi_data'

    dataset = datasets.ImageFolder(data_dir, data_transforms)

    print(dataset)
    print(torch.__version__)
    batch_size = 4
    num_workers = 4
    k_fold_validation = 10
    random_seed= 16031997
    total = 0

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if(type=="k-fold"):
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        kfolds = KFold(n_splits=k_fold_validation)
        kfolds.get_n_splits(indices)
        final_indices =  kfolds.split(indices)
    elif(type=="randomsampler"):
        final_indices = generator_sampling_random(k_fold_validation, indices, random_seed)
    elif(type=="once"):
        k_fold_validation = 1
        final_indices = generator_sampling_random(k_fold_validation, indices, random_seed)
    else:
        exit(0)
    for train_indices, val_indices in final_indices:
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=valid_sampler, num_workers=num_workers)

        dataloaders = {'train':train_loader, "val":validation_loader}
        dataset_sizes = {'train':len(train_indices), "val":len(val_indices)}

        class_names = ['no', 'yes']
        model, best_acc = model_generation("resnet18",dataloaders, dataset_sizes, name_pth, False)
        total += best_acc
    print("k-fold valid acc is: " + str(total/k_fold_validation))
    csv_val_model(class_names, dataloaders, model, name_pth)

main("k-fold")
