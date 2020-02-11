import time
import copy
import torch
from utils import *
import pandas as pd


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, args):
    since = time.time()
    best_acc = 0.0
    epochs_details_list = []
    results_detailed_list = []
    class_names = ['no', 'yes']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device ", device)


    for epoch in range(args.num_epochs):
        results_detailed_list = []
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
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
                # track history if only i nb n train
                with torch.set_grad_enabled(phase == 'train'):
                    ##en val si entra aqui pero no aplica el step ni el backguard
                    if(phase == 'train' and args.arch == "inception_v3"):
                        outputs, aux = model(inputs)
                    else:
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    if(phase == 'val'):
                        for i in range(inputs.size()[0]):
                            if(labels.cpu().data[i].item() == 1):
                                results_detailed_list.append({'Expected_label': "yes", 'Given_label': class_names[preds[i]]})
                            else:
                                results_detailed_list.append({'Expected_label': "no", 'Given_label': class_names[preds[i]]})

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #print("running corrects train: " + str(running_corrects.double()))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if(args.arch == "resnetusm"):
                epochs_details_list.append({'epoch': epoch, 'phase': phase, 'acc':epoch_acc.item(), 'loss':epoch_loss, 'alpha':model.filter.alpha.item()})
            else:
                epochs_details_list.append({'epoch': epoch, 'phase': phase, 'acc':epoch_acc.item(), 'loss':epoch_loss, 'alpha':0})
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc, epochs_details_list, results_detailed_list


def eval_model_dataloader(model, dataloader,dataset_sizes, criterion, class_names, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    results_detailed_list = []
    expected_label_list = []
    given_label_list = []
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            for i in range(inputs.size()[0]):
                if(labels.cpu().data[i].item() == 1):
                    results_detailed_list.append({'Expected_label': "yes", 'Given_label': class_names[preds[i]]})
                else:
                    results_detailed_list.append({'Expected_label': "no", 'Given_label': class_names[preds[i]]})
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print("running_corrects: " + str(running_corrects))
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.double() / dataset_sizes['val']
    print("accuracy directly: "+ str(epoch_acc))
    model.train(mode=was_training)
    return results_detailed_list, epoch_acc


def eval_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, class_names, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


"""
phase = 'val'
    since = time.time()

    best_acc = 0.0
    model.eval()   # Set model to evaluate mode
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

"""
