# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from sklearn.metrics import f1_score
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

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import BreastDatasetLoader as BreastDataset

logger = logging.getLogger('main')


best_prec1 = 0
global_step = 0

"""
We trained the network using stochastic gradient descent with initial learning rate 0.2 and Nesterov
momentum 0.9. We trained for 180 epochs (when training with 1000 labels) or 300 epochs (when
training with 4000 labels), decaying the learning rate with cosine annealing [14] so that it would

python main.py \
    --dataset cifar10 \
    --labels data-local/labels/cifar10/1000_balanced_labels/00.txt \
    --arch cifar_shakeshake26 \
    --consistency 100.0 \
    --consistency-rampup 5 \
    --labeled-batch-size 62 \
    --epochs 180  \
    --lr-rampdown-epochs 200



"""

class MeanTeacherController():
    def __init__(self):

        self.args = cli.parse_commandline_args()
        self.context = RunContext(logging)
        self.training_log = self.context.create_train_log("training")
        useCuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if useCuda else "cpu")

    def train_model(self):
        """
        Train the model for the specified number of epochs inside
        :return:
        """
        global global_step
        global best_prec1
        args = self.args
        context = self.context
        checkpoint_path = context.transient_dir
        training_log = context.create_train_log("training")
        validation_log = context.create_train_log("validation")
        teacher_validation_log = context.create_train_log("ema_validation")
        #dictionary with all the dataset information
        """
        For instance, for CIFAR
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
        """
        dataset_config = datasets.__dict__[self.args.dataset]()
        num_classes = dataset_config.pop('num_classes')
        # MODIFY IN HERE
        # train loader
        train_loader, eval_loader = self.create_data_loaders(**dataset_config, args=self.args)





        # ema is exponential moving average
        # student
        student_model = self.create_model(num_classes)
        # teacher
        teacher_model = self.create_model(num_classes, is_teacher = True)

        logger.info(parameters_string(student_model))
        # create the optimizer
        optimizer = torch.optim.SGD(student_model.parameters(), self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay,
                                    nesterov=self.args.nesterov)

        if self.args.consistency:
            logger.info("Using the consistency loss, semi-supervised training about to start: " + str(args.consistency))
        else:
            logger.info("No consistency loss used")
        # optionally resume from a checkpoint
        if self.args.resume:
            assert os.path.isfile(self.args.resume), "=> no checkpoint found at '{}'".format(self.args.resume)
            logger.info("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_prec1 = checkpoint['best_prec1']
            student_model.load_state_dict(checkpoint['state_dict'])
            teacher_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))

        cudnn.benchmark = True

        if self.args.evaluate:
            #pure evaluation
            logger.info("Evaluating the student model:")
            logger.info("Evaluating the Teacher model:")
            return

        best_test_loss = 9999
        best_accuracy = 9999
        best_mse_score = 9999
        is_best = False
        for epoch in range(self.args.start_epoch, self.args.epochs):
            start_time = time.time()
            # train for one epoch
            (loss_epoch, epoch_accuracy, epoch_MAE) = self.train_epoch(train_loader, student_model, teacher_model, optimizer, epoch, training_log)



            if self.args.evaluation_epochs and (epoch + 1) % self.args.evaluation_epochs == 0:
                start_time = time.time()
                logger.info("Evaluating the student model:")

                (MAE_loss, accuracy, mse_loss) = self.test_model_epoch(student_model, eval_loader, nn.MSELoss().cuda(), epoch)
                #test_model_epoch(self, model, test_loader, test_loss_function)

                logger.info("Evaluating the teacher model:")
                (MAE_loss_teacher, accuracy_teacher, mse_loss_teacher) = self.test_model_epoch(student_model, eval_loader, nn.MSELoss().cuda(), epoch)
                logger.info("--- validation in %s seconds ---" % (time.time() - start_time))
                is_best = False
                self.training_log.record(row_id, {'Epoch_Test_MAE_loss': MAE_loss})
                if(MAE_loss < best_test_loss):
                    best_test_loss = MAE_loss
                    best_mse_score = mse_loss
                    best_accuracy = accuracy
                    is_best = True
                    logger.info("Lowest test loss so far: " + str(MAE_loss) + " and highest accuracy: " + str(accuracy))


            if self.args.checkpoint_epochs and (epoch + 1) % self.args.checkpoint_epochs == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': self.args.arch,
                    'state_dict': student_model.state_dict(),
                    'ema_state_dict': teacher_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)
            logger.info("Epochs finshed! Highest accuracy: " + str(best_accuracy) + " Lowest test loss: " + str(MAE_loss))



    def create_model(self, num_classes, is_teacher = False):

        """
        Create wether the student or teacher model
        :param num_classes:
        :param is_teacher:
        :return:
        """
        logger.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if self.args.pretrained else '',
            ema='EMA ' if is_teacher else '',
            arch=self.args.arch))
        #model factory to build the architecture
        model_factory = architectures.__dict__[self.args.arch]
        #number of classes is specified
        model_params = dict(pretrained=self.args.pretrained, num_classes=num_classes)
        #create actual model
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()
        #if it is the teacher, the parameters are not trainable, are just the EMA of the past parameters
        if is_teacher:
            for param in model.parameters():
                param.detach_()

        return model

    def parse_dict_args(self, **kwargs):
        """
        Dont know?
        :param kwargs:
        :return:
        """
        global args

        def to_cmdline_kwarg(key, value):
            if len(key) == 1:
                key = "-{}".format(key)
            else:
                key = "--{}".format(re.sub(r"_", "-", key))
            value = str(value)
            return key, value

        kwargs_pairs = (to_cmdline_kwarg(key, value)
                        for key, value in kwargs.items())
        cmdline_args = list(sum(kwargs_pairs, ()))
        args = parser.parse_args(cmdline_args)


    def create_data_loaders(self, train_transformation, eval_transformation, datadir, args):
        """
        Creates the dataset loaders
        :param train_transformation:
        :param eval_transformation:
        :param datadir:
        :param args:
        :return:
        """
        logger.info("Loading data from: " + datadir)
        traindir = os.path.join(datadir, self.args.train_subdir)
        evaldir = os.path.join(datadir, self.args.eval_subdir)
        assert_exactly_one([self.args.exclude_unlabeled, self.args.labeled_batch_size])
        dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

        if self.args.labels:

            with open(self.args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            #takes the file names in the labels dictionary as labeled data, and the rest, as unlabeled
            #MODIFICATION FOR A MAXIMUM OF UNLABELED OBSERVATIONS, TO STUDY THE BEHAVIOUR WITH DIFFERENT NUMBER OF UNLABELED OBSERVATIONS
                labeled_idxs, unlabeled_idxs, validation_idxs, dataset = data.relabel_dataset(dataset, labels)
                logger.info("Number of labeled training observations: " + str(len(labeled_idxs)))
                logger.info("Number of labeled validation observations: " + str(len(validation_idxs)))
                logger.info("Number of unlabeled observations: " + str(len(unlabeled_idxs)))
                if(len(labeled_idxs) < self.args.batch_size or len(validation_idxs) < self.args.batch_size or len(unlabeled_idxs) < self.args.batch_size):
                    logger.warning("Warning, the batch size is larger than a subset of data")

        if self.args.exclude_unlabeled:
            logger.info("Not using unlabeled data")
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, self.args.batch_size, drop_last=False)
        elif self.args.labeled_batch_size:
            logger.info("Using unlabeled data")
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, self.args.batch_size, self.args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(self.args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.args.workers, pin_memory=True)
        # evaluation loader
        sampler_eval = SubsetRandomSampler(validation_idxs)
        #what is drop last and pin_memory???
        batch_sampler_eval = BatchSampler(sampler_eval, self.args.batch_size, drop_last=False)
        eval_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_eval, num_workers=  self.args.workers, pin_memory=True)
        return train_loader, eval_loader



    def update_teacher_variables(self, model, ema_model, alpha, global_step):
        """
        Implements the exponential moving average of the student weights
        :param model:
        :param ema_model:
        :param alpha:
        :param global_step:
        :return:
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)




    def train_epoch(self, train_loader, student_model, teacher_model, optimizer, epoch, log):
        """
        Actions for training the model in one epoch
        :param train_loader:
        :param student_model:
        :param teacher_model:
        :param optimizer:
        :param epoch:
        :param log:
        :return:
        """

        meters = AverageMeterSet()
        global global_step

        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        if self.args.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif self.args.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, self.args.consistency_type
        residual_logit_criterion = losses.symmetric_mse_loss
        # switch to train mode
        #Student
        student_model.train()
        #TEACHER?? yes, the exponentially averaged model
        teacher_model.train()

        end = time.time()
        for i, ((input, teacher_input), target) in enumerate(train_loader):



            # measure data loading time
            meters.update('data_time', time.time() - end)
            #how they adjust the learning rate??
            self.adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
            #input variable
            input_var = torch.autograd.Variable(input)
            #volatile??
            #Basically, set the input to a network to volatile if you are doing inference only and won't be running backpropagation in order to conserve memory.
            with torch.no_grad():
                ema_input_var = torch.autograd.Variable(teacher_input)
            target_var = torch.autograd.Variable(target.cuda())

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)
            #output mean teacher
            teacher_model_out = teacher_model(ema_input_var)
            #output student
            student_model_out = student_model(input_var)

            if isinstance(student_model_out, Variable):
                assert self.args.logit_distance_cost < 0
                #logger.warning("Using only one output per model")
                #case only one output
                #output student
                logit1 = student_model_out
                #output mean teacher
                teacher_logit = teacher_model_out
            else:
                #trying to to use two outputs per model")

                #two outputs per model??? whyyyyy
                assert len(student_model_out) == 2
                assert len(teacher_model_out) == 2
                #model output in two parts, logit1 and logit2
                logit1, logit2 = student_model_out
                #ema_logit is the ema output, the student??
                teacher_logit, _ = teacher_model_out
            #weights??
            #detach the teacher weights to keep its weights non optimizable
            teacher_logit = Variable(teacher_logit.detach().data, requires_grad=False)
            #residual logit criterion?????????????
            if self.args.logit_distance_cost >= 0:
                # Using two outputs per model
                class_logit, cons_logit = logit1, logit2
                res_loss = self.args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.item())
            else:
                #Using ONE output per model for sure
                class_logit, cons_logit = logit1, logit1
                res_loss = 0

            class_loss = class_criterion(class_logit, target_var) / minibatch_size
            meters.update('class_loss', class_loss.item())

            teacher_class_loss = class_criterion(teacher_logit, target_var) / minibatch_size
            meters.update('ema_class_loss', teacher_class_loss.item())

            if self.args.consistency:
                consistency_weight = self.get_current_consistency_weight(epoch)
                meters.update('cons_weight', consistency_weight)
                #consistency between the teacher and the student??
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, teacher_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)

            loss = class_loss + consistency_loss + res_loss
            assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
            meters.update('loss', loss.item())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            #UPDATE THE TEACHER WEIGHTS
            self.update_teacher_variables(student_model, teacher_model, self.args.ema_decay, global_step)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()
            # L1 distance and corrects
            (mae, corrects) = self.calculate_MAE_accuracy(target, class_logit)
            meters.update("MAE", mae)
            # amount of correct predictions
            meters.update("Corrects", corrects)
            if i % self.args.print_freq == 0:
                logger.info("Epoch: {} \t Batch: {} \t Time: {:.4f}, \t Loss: {:.4f} \t Corrects: {} / {}  \t MAE: {:.4f} ".format(epoch, i,  meters["batch_time"], meters["loss"], meters["Corrects"], self.args.batch_size, meters["MAE"]))

        averages = meters.averages()
        sums = meters.sums()
        #stats for the epoch
        epoch_accuracy = sums["Corrects/sum"].double() / len(train_loader.sampler)
        epoch_MAE = averages["MAE/avg"]
        train_loss_epoch_norm = sums["loss/sum"]
        #logging
        row_id = "K-" + str(0) + "__" + "E-" + str(epoch)
        self.training_log.record(row_id, {'Epoch_Training_loss': train_loss_epoch_norm})
        logger.info(
            "Totals for epoch: {} \t Training accuracy: {:.4f} \t Corrects: {}/{} \t Training MAE {:.4f} \t Training Loss: {:.4f}".format(
                epoch, epoch_accuracy, sums["Corrects/sum"].int(), len(train_loader.sampler), epoch_MAE, train_loss_epoch_norm))

        return (train_loss_epoch_norm, epoch_accuracy, epoch_MAE)

    def calculate_MAE_accuracy(self, target, class_logit):
        #put in number format
        class_numbers = class_logit.argmax(dim=1, keepdim=True)
        MAE = torch.dist(target.data.cpu().float(), class_numbers.cpu().float(), 1).item()
        #just to be sure, reshape one of the tensors
        equals_vec = class_numbers.cpu() == target.view(class_numbers.shape).cpu()
        corrects = torch.sum(equals_vec)
        #print("Number of corrects")
        #print(corrects.item())
        return (MAE, corrects)

    def test_model_epoch(self, model, test_loader, test_loss_function, epoch):
        """
        Test model
        :param args:
        :param model:
        :param device:
        :param test_loader:
        :return:
        """
        meters = AverageMeterSet()
        model.eval()
        # reduce=True nn.MSELoss().cuda()
        cudnn.benchmark = True
        #No grad for evalaution
        with torch.no_grad():
            for input, target in test_loader:
                input = input[0]
                target = target.float()
                input = input.to(self.device)
                target = target.to(self.device)
                output = model(input)
                # sum up batch loss
                #from logits to number representation
                prediction_numbers = output.argmax(dim=1, keepdim=True)
                prediction_numbers = prediction_numbers[:,0]
                #evaluating test loss
                local_test_loss = test_loss_function(prediction_numbers, target).item()
                meters.update("MSE_loss", local_test_loss)
                (mae, corrects) = self.calculate_MAE_accuracy(target, output)
                meters.update("MAE_loss", mae)
                meters.update("Corrects", corrects)

        total_observations = len(test_loader.batch_sampler)
        averages = meters.averages()
        sums = meters.sums()
        accuracy = 100. * sums["Corrects/sum"].item() / total_observations
        information = 'Epoch: {} Test MSE loss: {:.4f} \t  Accuracy: {}/{} ({:.0f}%) \t MAE:{:.4f}\n'.format(epoch, averages["MSE_loss/avg"], meters["Corrects"], total_observations, accuracy, averages["MAE_loss/avg"])
        logger.info(information)
        return (sums["MAE_loss/sum"], accuracy, averages["MSE_loss/avg"])

    def calculate_f1_score_batch(self, prediction, targets):
        # macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        f1_score_data = f1_score(targets, prediction, average="macro")
        return f1_score_data



    def save_checkpoint(self, state, is_best, dirpath, epoch):
        """
        Save weights checkpoint
        :param state:
        :param is_best:
        :param dirpath:
        :param epoch:
        :return:
        """
        #filename = 'checkpoint.{}.ckpt'.format(epoch)
        #overwrite
        filename = 'checkpoint.ckpt'
        checkpoint_path = os.path.join(dirpath, filename)
        best_path = os.path.join(dirpath, 'best.ckpt')
        torch.save(state, checkpoint_path)
        logger.info("--- checkpoint saved to %s ---" % checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_path)
            logger.info("--- checkpoint copied to %s ---" % best_path)


    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
        """
        Adjust the learning rate
        :param optimizer:
        :param epoch:
        :param step_in_epoch:
        :param total_steps_in_epoch:
        :return:
        """
        args = self.args
        lr = self.args.lr
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        """
        With these simple techniques, our Caffe2-
        based system trains ResNet-50 with a minibatch size of 8192
        on 256 GPUs in one hour, while matching small minibatch
        accuracy. Using commodity hardware, our implementation
        achieves ∼90% scaling efficiency when moving from 8 to
        256 GPUs. Our findings enable training visual recognition
        models on internet-scale data with high efficiency
        """
        lr = ramps.linear_rampup(epoch, self.args.lr_rampup) * (self.args.lr - self.args.initial_lr) + self.args.initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.args.lr_rampdown_epochs:
            assert self.args.lr_rampdown_epochs >= self.args.epochs
            lr *= ramps.cosine_rampdown(epoch, self.args.lr_rampdown_epochs)

        #logger.warning("Learning rate: " + str(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        #unsupervised weight ramp-up function
        """
        we noticed that optimization tended to explode during the ramp-up period, and we
        eventually found that using a lower value for Adam β2 parameter (e.g., 0.99 instead of 0.999) seems
        to help in this regard.
        In our implementation, the unsupervised loss weighting function w(t) ramps up, starting from zero,
        along a Gaussian curve during the first 80 training epochs. See Appendix A for further details about
        this and other training parameters. In the beginning the total loss and the learning gradients are thus
        dominated by the supervised loss component, i.e., the labeled data only. We have found it to be
        very important that the ramp-up of the unsupervised loss component is slow enough—otherwise,
        the network gets easily stuck in a degenerate solution where no meaningful classification of the data
        is obtained.

        """

        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def test_loaders(self):

        dataset_config = datasets.__dict__[self.args.dataset]()
        num_classes = dataset_config.pop('num_classes')
        # MODIFY IN HERE
        # train loader
        train_loader, eval_loader = self.create_data_loaders(**dataset_config, args=self.args)
        targets_all = []
        print("Evaluation data")
        for (input, targets) in eval_loader:
            print(targets)
            print(targets.shape)
        print("Training data")
        for ((input, teacher_input), target1) in train_loader:
            print(target1)
            print(target1.shape)


    def calculate_accuracy(self, output, target, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        :param output:
        :param target:
        :param topk:
        :return:
        """
        maxk = max(topk)
        labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
        return res

def create_data_partitions():
    dataset_path = "/media/Data/saul/Datasets/Inbreast_folder_per_class_binary/train"
    percentage_labeled_observations = 0.5
    BreastDataset.create_labeled_unlabeled_partitions_indices(dataset_path, percentage_labeled_observations)






if __name__ == '__main__':
    mean_teacher_controller = MeanTeacherController()
    #mean_teacher_controller.test_loaders()
    mean_teacher_controller.train_model()
    #create_data_partitions()