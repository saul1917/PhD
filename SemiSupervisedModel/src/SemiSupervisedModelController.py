import re
import argparse
import os
import shutil
import time
import math
import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision import models, utils, transforms
import torchvision.datasets
from MeanTeacherUtilities.utils import *
import MeanTeacherUtilities.cli as cli
import MeanTeacherUtilities.data as data
import MeanTeacherUtilities.utils as utils
import MeanTeacherUtilities.losses as losses
from MeanTeacherUtilities.run_context import *
from sklearn.model_selection import KFold
from BreastDatasetLoader import INbreastDataset
from BreastDatasetLoader import SubsetSampler
from ModelCreatorInBreast import ModelCreatorInBreast
import BreastDatasetLoader as BreastDataset
from sklearn.model_selection import train_test_split

#Default variables
DEFAULT_PATH = "/media/Data/saul/InBreastDataset"

DEFAULT_CSV_PATH = DEFAULT_PATH + '/INbreast.csv'
DEFAULT_DATA_PATH = DEFAULT_PATH + '/AllDICOMs/'
NO_LABEL = -1
random_seed = 64325564
TEST_SPLIT = 0.2


class ModelController:
    def __init__(self, context, args):
        """
        Constructor
        :param context:
        :param args:
        """
        self.LOG = context.get_logger()
        self.context = context
        self.args = args
        # use gpu?
        self.useCuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.useCuda else "cpu")
        cudnn.benchmark = True
        # create all the log files to write
        self.checkpointPath = context.result_dir
        self.trainingLog = context.create_train_log("training")
        self.validationLog = context.create_train_log("validation")
        self.emaValidationLog = context.create_train_log("ema_validation")
        # get alexnet model
        self.modelCreatorInBreast = ModelCreatorInBreast(self.device, self.useCuda)
        # create student model
        self.studentModel = self.modelCreatorInBreast.getAlexNet(trainTopOnly = False)
        # create teacher model
        self.teacherModel = self.modelCreatorInBreast.getAlexNet(isTeacher=True)
        # only the student must be optimized!
        self.optimizerStudent = self.modelCreatorInBreast.getOptimizer(self.args, self.studentModel)
        #labeled data criterion
        #size_average=False, ignore_index = NO_LABEL

        #default loss
        self.labeledCriterion = nn.CrossEntropyLoss().cuda()
        #unlabeled data criterion, we can use the MSE or Kullback-Leibler loss
        if args.consistency_type == 'mse':
            self.consistencyCriterion = losses.softmax_mse_loss
        elif args.consistency_type == 'kl':
            self.consistencyCriterion = losses.softmax_kl_loss
        else:
            assert False, args.consistency_type
        self.residualLogitCriterion = losses.symmetric_mse_loss



    def load_checkpoint(self):
        """
        Loads the dictionary with the state of training existing weights from files
        :return:
        """
        # self.save_checkpoint({'epoch': epoch + 1, 'state_dict': self.studentModel.state_dict(), 'best_prediction': best_test_loss,
        #                                           'optimizer': self.optimizerStudent.state_dict()}, is_best)
        if self.args.resume:
            filename = args.resumefile
            checkpoint_path = os.path.join(self.checkpointPath, filename)
            assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
            self.LOG.warning("=> Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.args.start_epoch = checkpoint['epoch']
            self.studentModel.load_state_dict(checkpoint['state_dict'])
            self.optimizerStudent.load_state_dict(checkpoint['optimizer'])
            self.LOG.info("=> loaded checkpoint '{}' (epoch {})".format(self.args.resumefile, checkpoint['epoch']))

    def save_checkpoint(self, state, isBest):
        """
        Saves the dictionary with the state of the system
        :param state:  Dictionary with the variables saved
        :param isBest: is the best epoch
        :param epoch:  epoch number
        :return:
        """
        #overwrite the same file

        filename = self.args.resumefile
        checkpoint_path = os.path.join(self.checkpointPath, filename)
        best_path = os.path.join(self.checkpointPath, self.args.best)
        torch.save(state, checkpoint_path)
        self.LOG.warning("--- checkpoint saved to %s ---" % checkpoint_path)
        if isBest:
            shutil.copyfile(checkpoint_path, best_path)
            self.LOG.warning("--- checkpoint copied to %s ---" % best_path)

    def evaluate_model(self):
        """
        Evaluation mode
        :return:
        """
        self.LOG.info("Evaluating the primary model:")

    """def create_data_loaders(self):
        
        (trainTransformations, validationTransformations) = BreastDataset.getTransformationsInBreast()
        #load dataset
        train_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector = False, transform = trainTransformations)
        self.LOG.info("Dataset loaded from: " + DEFAULT_DATA_PATH)
        #data_path, csv_path, useOneHotVector = False, transform = None)

        eval_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector = True, transform =  validationTransformations)
        xIndices = train_dataset.getfilenames()
        self.LOG.warning("Dataset loaded with: " + str(len(xIndices)) + " observations")
        y = train_dataset.getlabels()
        # Make the first split (labeled/unlabeled)
        train_indices, val_indices, train_labels, val_labels = train_test_split(xIndices, y, test_size = TEST_SPLIT, random_state=random_seed)
        self.LOG.warning("Number of training observations: " + str(len(train_indices)))
        self.LOG.warning("Number of validation observations: " + str(len(val_indices)))
        supervised_indices, unsupervised_indices, supervised_labels, unsupervised_labels = train_test_split(train_indices, train_labels, test_size = TEST_SPLIT, random_state=random_seed)

        #assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
        #only labeleda data
        if self.args.splits_unlabeled == 0:
            sampler_training = SubsetRandomSampler(train_indices)
            sampler_validation = SubsetRandomSampler(val_indices)
            #??
            batch_sampler_training = BatchSampler(sampler_training, self.args.batch_size, drop_last=True)
            batch_sampler_validation = BatchSampler(sampler_validation, self.args.batch_size, drop_last=True)
        else:
            #batch sampler for unlabeled and labeled data
            batch_sampler_training = data.TwoStreamBatchSampler(unsupervised_indices, supervised_indices, self.args.batch_size, self.args.labeled_batch_size)

        #train loader with the defined inbreast dataset
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_training,  num_workers = self.args.workers, pin_memory=True)
        #evalLoader with the defined inbreast dataset
        #drop_last=False
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_sampler = batch_sampler_validation, num_workers= self.args.workers, pin_memory = True)
        #return both loaders
        return train_loader, eval_loader"""

    def create_data_loaders_k_folds(self):
        """
        Create dataset and K folds or partitions
        :return:
        """
        (trainTransformations, validationTransformations) = BreastDataset.getTransformationsInBreast()
        train_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector = False, transform = trainTransformations)
        eval_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector=True, transform=validationTransformations)
        self.LOG.info("Dataset loaded from: " + DEFAULT_DATA_PATH)
        xIndices = train_dataset.getfilenames()

        kfolds = KFold(n_splits = self.args.k_fold_num, random_state=42, shuffle = True)
        self.LOG.info("Using a kfold of "+ str(self.args.k_fold_num))
        #kfolds.get_n_splits(xIndices)
        final_indices = kfolds.split(xIndices)
        return (train_dataset, eval_dataset, final_indices, xIndices)

    def create_data_loaders_k_folds_unlabeled(self):
        """
        Create dataset and K folds or partitions
        :return:
        """
        (trainTransformations, validationTransformations) = BreastDataset.getTransformationsInBreast()
        train_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector = False, transform = trainTransformations)
        eval_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector=True, transform=validationTransformations)
        self.LOG.warning("Dataset loaded from: " + DEFAULT_DATA_PATH)
        input_file_numbers_all = train_dataset.getfilenames()
        targets_all = train_dataset.getlabels()
        self.LOG.warning("Number splits labeled/unlabeled " + str(self.args.splits_unlabeled))
        kfolds_labeled_unlabeled = KFold(n_splits= int(self.args.splits_unlabeled), random_state=52, shuffle=True)
        kfolds_validation_training = KFold(n_splits = self.args.k_fold_num, random_state=42, shuffle = True)
        self.LOG.warning("Using a k-fold of: "+ str(self.args.k_fold_num))
        #kfolds.get_n_splits(xIndices)
        labeled_unlabeled_indices = kfolds_labeled_unlabeled.split(input_file_numbers_all)
        current_fold_labeled = 1
        for labeled_indices, unlabeled_indices in labeled_unlabeled_indices:
            if(current_fold_labeled == self.args.current_fold):
                validation_training_indices_k_folded = kfolds_validation_training.split(labeled_indices)
            current_fold_labeled += 1



        self.LOG.warning("Split selected for labeled/unlabeled " + str(self.args.current_fold))
        percentage_labeled = len(labeled_indices) / len(input_file_numbers_all)
        self.LOG.warning("Number of labeled observations: " + str(len(labeled_indices)) + " proportion: " + str(percentage_labeled))



        return (train_dataset, eval_dataset, validation_training_indices_k_folded, input_file_numbers_all, unlabeled_indices)

    def train_model_supervised_k_fold(self):
        """
        Train model using K folds cross validation
        :return:
        """
        #average metrics counter
        meters = utils.AverageMeterSet()




        if(self.args.splits_unlabeled == 0):
            (train_dataset, eval_dataset, validation_training_indices_k_folded, input_file_numbers) = self.create_data_loaders_k_folds()
            self.LOG.warning("Fully supervised training")
        else:
            self.LOG.warning("Using the labelled data partially")
            (train_dataset, eval_dataset, validation_training_indices_k_folded, input_file_numbers, unlabeled_indices) = self.create_data_loaders_k_folds_unlabeled()
        fold = 1
        accuracies_k_folds_torch = []
        #balance the loss function
        if(self.args.weight_balancing):
            self.LOG.warning("Weight balancing through the loss function on")
            self.balance_loss(train_dataset)


        for train_indices, val_indices in validation_training_indices_k_folded:
            #IMPORTANT: The train indices splitted by k-folds are 0-n indices
            #the get function in the INBREAST receives
            train_indices = input_file_numbers[train_indices]
            val_indices = input_file_numbers[val_indices]
            self.LOG.warning("--------Excecuting fold: {} with: {} observations for training and: {} observations for validation".format(fold, len(train_indices), len(val_indices)))
            # the indices are shuffled again so the training is random (with the seed) because it's in order
            np.random.seed(self.args.random_seed)
            #independent shuffling, doesnt matter
            np.random.shuffle(train_indices)
            # independent shuffling, doesnt matter
            np.random.shuffle(val_indices)
            # creates samplers for the dataloaders
            train_sampler = SubsetSampler(train_indices)
            valid_sampler = SubsetSampler(val_indices)
            #create the data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.args.batch_size, sampler = train_sampler,num_workers= self.args.workers)
            validation_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = self.args.batch_size, sampler = valid_sampler, num_workers = self.args.workers)
            #return the test loss for the k-fold
            (best_test_loss, best_accuracy) = self.train_model_supervised(train_loader, validation_loader, fold)
            #concatenate the best test loss achieved for the k-fold
            accuracies_k_folds_torch += [best_accuracy]
            fold += 1
        accuracies_k_folds_torch = torch.tensor(accuracies_k_folds_torch)
        k_folds_mean = accuracies_k_folds_torch.mean().item()
        k_folds_std = accuracies_k_folds_torch.std().item()
        self.LOG.warning("K-folds stats, mean: " + str(k_folds_mean) + " std: " + str(k_folds_std))
        self.trainingLog.record("K folds stats", {'Mean': k_folds_mean, 'Std':k_folds_std})

    def balance_loss(self, dataset_train):
        number_classes = 6
        y = dataset_train.getlabels()

        #by default float tensor
        weights = torch.zeros(6)
        #get the ammount of labels per class
        for i in range(0, len(y)):
            label = y[i]
            weights[label] += 1
        #normalize
        weights = weights / torch.sum(weights)
        print(weights)
        class_weights = torch.FloatTensor(weights).cuda()

        self.labeledCriterion = nn.CrossEntropyLoss(weight=class_weights).cuda()


    def train_model_supervised(self, train_loader, eval_loader, k_fold):
        """
        Model trainer
        :return:
        """
        best_test_loss = 9999
        best_accuracy = 9999
        for epoch in range(self.args.start_epoch, self.args.epochs):
            startTime = time.time()
            # train for one epoch, using the student (trainable model)
            (epochLoss, epochAccuracy, epochMAE) = self.trainModelSupervisedEpoch(train_loader, self.studentModel, self.labeledCriterion, self.optimizerStudent, epoch)
            self.LOG.warning("Training epoch in {:.3f} seconds, for k-fold: {}".format((time.time() - startTime), k_fold))
            #is always recommended to use another loss for testing
            (test_loss, accuracy) = self.test_model_supervised_epoch(self.studentModel, eval_loader, nn.MSELoss().cuda())

            #write to the training log, for epoch error information
            row_id = "K-" + str(k_fold) + "__" + "E-" + str(epoch)
            self.trainingLog.record(row_id,{'Epoch_Training_loss': epochLoss, 'Epoch_Training_accuracy': epochAccuracy.item(), 'Epoch_Training_MAE': epochMAE, "Epoch_Acc_validation":accuracy})
            is_best = test_loss < best_test_loss
            #Save weights to recover the state later
            if self.args.checkpoint_epochs and (epoch + 1) % self.args.checkpoint_epochs == 0:
                #save checkpoint receives a dictionary
                self.LOG.warning("Saving chechpoint...")
                self.save_checkpoint({'epoch': epoch + 1, 'state_dict': self.studentModel.state_dict(), 'best_prediction': best_test_loss, 'optimizer': self.optimizerStudent.state_dict()}, is_best)
            if(is_best):
                best_test_loss = test_loss
                best_accuracy = accuracy
        #save the information of the best
        self.LOG.warning("Best accuracy: " + str(best_accuracy) + " for k-fold: " + str(k_fold))
        self.trainingLog.record("Best_test_acc_k-fold: " + str(k_fold),{'Accuracy test loss': best_accuracy})
        return (best_test_loss, best_accuracy)


    def trainModelSupervisedEpoch(self, trainLoader, fullModel, criterion, optimizer, epoch):
        """
        Trains the model using a train loader, from scipy, to ease data splitting
        :param device:
        :param fullModel:
        :param batchSize:
        :param criterion:
        :param optimizer:
        :param epoch:
        :return:
        """
        # print(fullModel.type())  # remove this logging line later rg
        fullModel.train()
        trainLoss = 0
        corrects = 0
        trainMAE = 0
        for inputs, targets in trainLoader:
            # put observations and targets to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # inputs = inputs.cuda()
            # targets = targets.cuda()
            # put optimizer gradients in zero
            optimizer.zero_grad()
            # with torch.set_grad_enabled(True):
            # get outputs
            outputs = fullModel(inputs)
            # get loss
            loss = criterion(outputs, targets)
            # get value predicted as maximum
            _, predicted = outputs.max(1)
            # update gradients
            loss.backward()
            optimizer.step()
            # train loss function, defined in the optimizer
            trainLoss += loss.item() * inputs.size(0)
            # L1 distance
            trainMAE += torch.dist(targets.data.float(), predicted.float(), 1).item() * inputs.size(0)
            #amount of correct predictions
            corrects += torch.sum(predicted == targets.data)
        # normalized loss
        epochLoss = trainLoss / len(trainLoader.sampler)
        # Accuracy, amount of corrects / dataset length
        epochAccuracy = corrects.double() / len(trainLoader.sampler)
        epochMAE = trainMAE / len(trainLoader.sampler)
        self.LOG.warning('Train epoch: {} --- Loss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch + 1, epochLoss,epochAccuracy,corrects, len(trainLoader.sampler),  epochMAE))
        return epochLoss, epochAccuracy, epochMAE

    def test_model_supervised_epoch(self, model, testLoader, testLossFunction):
        """
        Test model
        :param args:
        :param model:
        :param device:
        :param testLoader:
        :return:
        """
        model.eval()
        test_loss_total = 0
        correct = 0
        #reduce=True nn.MSELoss().cuda()
        cudnn.benchmark = True
        i = 0
        with torch.no_grad():
            for input, target in testLoader:

                input, target = input.to(self.device), target.to(self.device)
                output = model(input)
                # sum up batch loss
                local_test_loss = testLossFunction(output, target).item()
                test_loss_total += local_test_loss
                # get the index of the max log-probability, to get the number of correct predictions
                prediction_numbers = output.argmax(dim = 1, keepdim = True)
                target_numbers = target.argmax(dim = 1, keepdim = True)
                #accumulate the number of correct predictions
                correct += prediction_numbers.eq(target_numbers.view_as(prediction_numbers)).sum().item()
                i += 1
        test_loss_total /= len(testLoader.sampler)
        #make sure is normalized by the lenght of the partition
        accuracy = 100. * correct / len(testLoader.sampler)
        information = 'Test MSE loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( test_loss_total, correct,len(testLoader.sampler), accuracy)
        self.LOG.warning(information)
        return (test_loss_total, accuracy)

def get_mean_stds_dataset():
    transformations = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    #complete train dataset
    train_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector=False, transform=transformations)
    BreastDataset.get_mean_and_std(train_dataset)

if __name__ == '__main__':
    #arguments from the command line
    args = cli.parse_commandline_args()
    #creates the necessary folders for the results, and inits the logging
    context = RunContext(__file__, 0, logging)
    modelController = ModelController(context, args)
    #(trainLoader, evalLoader) = modelController.create_data_loaders()

    #(train_dataset, eval_dataset, validation_training_indices_k_folded, input_file_numbers,  unlabeled_indices) = modelController.create_data_loaders_k_folds_unlabeled()


    #modelController.train_model_supervised(trainLoader, evalLoader)
    modelController.train_model_supervised_k_fold()
    #get_mean_stds_dataset()
	