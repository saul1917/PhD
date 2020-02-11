from numpy import *
import math
import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(name_csv1, name_csv2=None):
    df = pd.read_csv(name_csv1)
    train_df =  df.loc[df['phase'] == 'train']
    val_df =  df.loc[df['phase'] == 'val']

    epochs_list = list(train_df["epoch"])
    acc_train = list(train_df["acc"])
    loss_train = list(train_df["loss"])

    acc_test = list(val_df["acc"])
    loss_test = list(val_df["loss"])
    alphas = list(val_df["alpha"])

    if(name_csv2 != None):
        df2 = pd.read_csv(name_csv2)
        train_df2 =  df2.loc[df2['phase'] == 'train']
        val_df2 =  df2.loc[df2['phase'] == 'val']

        epochs_list2 = list(train_df2["epoch"])
        acc_train2 = list(train_df2["acc"])
        loss_train2 = list(train_df2["loss"])

        acc_test2 = list(val_df2["acc"])
        loss_test2 = list(val_df2["loss"])
        alphas2 = list(val_df2["alpha"])


    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0, 0])
    plt.plot(epochs_list, acc_train, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy training')
    if(name_csv2 != None):
        plt.plot(epochs_list2, acc_train2, 'r')
    plt.subplot(grid[0, 1])
    plt.plot(epochs_list, loss_train, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss in training')
    if(name_csv2 != None):
        plt.plot(epochs_list2, loss_train2, 'r')
    plt.subplot(grid[1, 0])
    plt.plot(epochs_list, acc_test, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Test')
    if(name_csv2 != None):
        plt.plot(epochs_list2, acc_test2, 'r')
    plt.subplot(grid[1, 1])
    plt.plot(epochs_list, loss_test, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss in Test')
    if(name_csv2 != None):
        plt.plot(epochs_list2, loss_test2, 'r')
    plt.subplot(grid[2, 0:])
    plt.plot(epochs_list, alphas, 'b')
    plt.xlabel('Epoch')
    #axes = plt.gca()
    #axes.set_ylim([2,5])
    plt.ylabel('Alphas')
    plt.show()

plot_epochs("results_total.csv", "resnet18_25epo224_2900norot_results_total.csv")
