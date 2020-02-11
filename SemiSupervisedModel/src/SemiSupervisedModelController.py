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
import torchvision.datasets
from MeanTeacherUtilities.utils import *
import MeanTeacherUtilities.cli as cli
import MeanTeacherUtilities.data as data
import MeanTeacherUtilities.utils as utils
import MeanTeacherUtilities.losses as losses
from MeanTeacherUtilities.run_context import *
from BreastDatasetLoader import INbreastDataset
from BreastDatasetLoader import SubsetSampler
from Architectures import ModelCreatorInBreast
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
        self.LOG = logging.getLogger('main')
        self.init_logger()
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
        self.studentModel = self.modelCreatorInBreast.getAlexNet()
        # create teacher model
        self.teacherModel = self.modelCreatorInBreast.getAlexNet(isTeacher=True)
        # only the student must be optimized!
        self.optimizerStudent = self.modelCreatorInBreast.getOptimizer(self.args, self.studentModel)
        #labeled data criterion
        #size_average=False, ignore_index = NO_LABEL
        self.labeledCriterion = nn.CrossEntropyLoss().cuda()
        #unlabeled data criterion, we can use the MSE or Kullback-Leibler loss
        if args.consistency_type == 'mse':
            consistencyCriterion = losses.softmax_mse_loss
        elif args.consistency_type == 'kl':
            consistencyCriterion = losses.softmax_kl_loss
        else:
            assert False, args.consistency_type
        residualLogitCriterion = losses.symmetric_mse_loss

    def init_logger(self):
        """
        Sets logging details
        :return:
        """
        #self.LOG.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.LOG.addHandler(handler)

    def load_checkpoint(self):
        """
        Loads the dictionary with the state of training existing weights from files
        :return:
        """
        if self.args.resume:
            assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
            self.LOG.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #global_step = checkpoint['global_step']
            #best_prec1 = checkpoint['best_prec1']
            self.studentModel.load_state_dict(checkpoint['state_dict'])
            self.teacherModel.load_state_dict(checkpoint['ema_state_dict'])
            self.optimizerStudent.load_state_dict(checkpoint['optimizer'])
            self.LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    def save_checkpoint(self, state, isBest):
        """
        Saves the dictionary with the state of the system
        :param state:  Dictionary with the variables saved
        :param isBest: is the best epoch
        :param epoch:  epoch number
        :return:
        """
        #overwrite the same file
        filename = 'checkpoint.ckpt'
        checkpoint_path = os.path.join(self.checkpointPath, filename)
        best_path = os.path.join(self.checkpointPath, 'best.ckpt')
        torch.save(state, checkpoint_path)
        self.LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
        if isBest:
            shutil.copyfile(checkpoint_path, best_path)
            self.LOG.info("--- checkpoint copied to %s ---" % best_path)

    def evaluate_model(self):
        """
        Evaluation mode
        :return:
        """
        self.LOG.info("Evaluating the primary model:")

    def create_data_loaders(self, args):
        """
        Create the dataset loaders
        :param args:
        :return:
        """
        (trainTransformations, validationTransformations) = BreastDataset.getTransformationsInBreast()
        self.LOG.info("Dataset loaded from: " + DEFAULT_DATA_PATH)
        train_dataset = INbreastDataset(DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, useOneHotVector = False, transform = trainTransformations)
        #data_path, csv_path, useOneHotVector = False, transform = None)
        # same thing except transforms?
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
        if args.exclude_unlabeled:
            sampler_training = SubsetRandomSampler(train_indices)
            sampler_validation = SubsetRandomSampler(val_indices)
            #??
            batch_sampler_training = BatchSampler(sampler_training, args.batch_size, drop_last=True)
            batch_sampler_validation = BatchSampler(sampler_validation, args.batch_size, drop_last=True)
        elif args.labeled_batch_size:
            #batch sampler for unlabeled and labeled data
            batch_sampler_training = data.TwoStreamBatchSampler(unsupervised_indices, supervised_indices, args.batch_size, args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(args.labeled_batch_size)
        #train loader with the defined inbreast dataset
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_training,  num_workers = args.workers, pin_memory=True)
        #evalLoader with the defined inbreast dataset
        #drop_last=False
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_sampler = batch_sampler_validation, num_workers= args.workers, pin_memory = True)
        #return both loaders
        return train_loader, eval_loader

    def train(self, epoch):
        #average metrics counter
        meters = utils.AverageMeterSet()


    def train_model_supervised(self, train_loader, eval_loader):
        """
        Model trainer
        :return:
        """
        for epoch in range(self.args.start_epoch, self.args.epochs):
            startTime = time.time()
            # train for one epoch, using the student (trainable model)
            (epochLoss, epochAccuracy, epochMAE) = self.trainModelSupervisedEpoch(train_loader, self.studentModel, self.labeledCriterion, self.optimizerStudent, epoch)
            #is always recommended to use another loss for testing
            testLoss = self.testModelSupervisedEpoch(self.studentModel, eval_loader, nn.MSELoss().cuda())
            #testLoss = 0
            self.LOG.warning("--- Training epoch in %s seconds ---" % (time.time() - startTime))
            #write to the training log, for epoch error information
            self.trainingLog.record(epoch,{'Epoch_Training_loss': epochLoss, 'Epoch_Training_accuracy': epochAccuracy.item(), 'Epoch_Training_MAE': epochMAE, "Epoch_MSE_validation":testLoss})
            isBest = False
            #Save weights to recover the state later
            if self.args.checkpoint_epochs and (epoch + 1) % self.args.checkpoint_epochs == 0:
                #save checkpoint receives a dictionary
                self.LOG.warning("Saving chechpoint...")
                self.save_checkpoint({'epoch': epoch + 1, 'state_dict': self.studentModel.state_dict(), 'best_prediction': 0, 'optimizer': self.optimizerStudent.state_dict()}, isBest)

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
        self.LOG.warning('Train Epoch: {} \tLoss: {:.6f} | Acc: {:.6f} {}/{} | MAE: {:.3f}'.format(epoch + 1, epochLoss,epochAccuracy,corrects, len(trainLoader.sampler),  epochMAE))
        return epochLoss, epochAccuracy, epochMAE

    def testModelSupervisedEpoch(self, model, testLoader, testLossFunction):
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
        test_loss_total /= len(testLoader.dataset)
        information = 'Test MSE loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( test_loss_total, correct, len(testLoader.dataset), 100. * correct / len(testLoader.dataset))
        self.LOG.warning(information)
        return test_loss_total


if __name__ == '__main__':
    #arguments from the command line
    args = cli.parse_commandline_args()
    #creates the necessary folders for the results, and inits the logging
    context = RunContext(__file__, 0, logging)
    modelController = ModelController(context, args)
    (trainLoader, evalLoader) = modelController.create_data_loaders(args)
    modelController.train_model_supervised(trainLoader, evalLoader)
	