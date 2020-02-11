import os
import csv
import torch
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean: " + str(mean))
    print("std: " + str(std))
    return mean, std



def load_model(model, weights):
    """load weights(arg) into a certain model(arg)"""
    if (os.path.exists(weights)):
        model.load_state_dict(torch.load(weights))
    else:
        print("Error: weights file doesn't exist")
        exit(0)
    return model

def save_model(model, total, args):
    """save the weights from a model"""
    """
    PARAMETERS:
        model: the model with the weights and architecture
        total: size of the dataset
        args: the arguments taken from parser in the main
    """
    pth_name = args.arch + "_epochs_" + str(args.num_epochs) + "_dataset_" + str(total) + "_weights_" +  datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+ ".pth"
    if(args.folder == ""):
        if not os.path.exists("./models/"):
            os.makedirs("./models/")
        torch.save(model.state_dict(), "./models/" + pth_name)
    else:
        if not os.path.exists("./" + args.folder + "/"):
            os.makedirs("./" + args.folder + "/")
        torch.save(model.state_dict(), "./" + args.folder + "/" + pth_name)


def save_epochs_results(epochs_details_list, total, args):
    """saves the accuracy, the loss and used alpha"""
    """
    PARAMETERS:
        epochs_details_list: each epoch with the information, ready to be saved
        total: size of dataset
        args: arguments from parser of main file
    """
    csv_name = args.arch + "_epochs_" + str(args.num_epochs) + "_dataset_" + str(total) + "_details_" +  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+ ".csv"
    with open(csv_name, 'w', newline='') as csvfile:
        fieldnames = ["epoch", "phase", "acc", "loss", "alpha"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for line in epochs_details_list:
            csvwriter.writerow(line)

def str2bool(v):
    """convert the string inputs to boolean values"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_best_epoch_results(results_detailed_list, total, args):
    """the details from the best epoch is saved"""
    df = pd.DataFrame(results_detailed_list)
    csv_name = args.arch + "_epochs_" + str(args.num_epochs) + "_dataset_" + str(total) + "_details_best_epoch_" +  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+ ".csv"
    df.to_csv(csv_name)


def get_stats(df, class_names):
    """getting confusion matrix information, acc, specificity, sensitivity from a dataframe"""
    expected_label_column = list(df["Expected_label"])
    given_label_column = list(df["Given_label"])
    cm = confusion_matrix(expected_label_column, given_label_column, labels=class_names)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    accuracy=(tn+tp)/total
    sensitivity = tp/(fn+tp)
    specificity = tn/(tn+fp)
    return accuracy, sensitivity, specificity, cm

def print_stats(results_detailed_list, class_names):
    """print the acc, specificity, sensitivity from a list """
    df = pd.DataFrame(results_detailed_list)
    accuracy, sensitivity, specificity, confusion_matrix = get_stats(df, class_names)

    print ('Accuracy : ', accuracy)
    print('Sensitivity : ', sensitivity)
    print('Specificity : ', specificity)
    return accuracy, sensitivity, specificity, confusion_matrix
