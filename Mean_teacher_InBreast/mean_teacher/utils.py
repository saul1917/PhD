# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Utility functions and classes"""
import torch
import sys
from sklearn.metrics import f1_score


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)

class MetricsStatistics:
    def __init__(self, csv_name):
        self.csv_name = csv_name
        self.torch_tensor = torch.tensor([1.0])
        self.is_empty = True

    def update(self, value):
        temp_array = torch.tensor([value], dtype = float)
        if(self.is_empty):
            self.torch_tensor = temp_array
            self.is_empty = False
        else:
            self.torch_tensor = torch.cat((self.torch_tensor, temp_array ), 0)

    def write_stats_to_log(self, log, training_log):
        """

        :param log:
        :param training_log:
        :return:
        """
        mean = torch.mean(self.torch_tensor).item()
        std = torch.std(self.torch_tensor).item()
        name_in_csv = "K-folds stats " + self.csv_name
        log.warning(name_in_csv + ", mean: "+ str(mean) + " std: " + str(std))
        training_log.record(name_in_csv, {'Mean': mean, 'Std': std})


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()



    def write_csv_row_testing(self, training_log, epoch, total_observations):
        row_id = "E-" + str(epoch)
        sums = self.sums()
        averages = self.averages()

        accuracy = 100. * sums["Corrects/sum"].item() / total_observations
        training_log.record(row_id, {'Epoch_Testing_accuracy': accuracy, 'Corrects_Testing': sums["Corrects/sum"].item(), "Test_MAE":averages["MAE_loss/avg"]})

    def write_csv_row_training(self, training_log, epoch):
        row_id = "E-" + str(epoch)
        sums = self.sums()
        train_loss_epoch_sum = sums["loss/sum"]
        consistency_loss_epoch_sum = sums["cons_loss/sum"]
        training_log.record(row_id, {'Epoch_Training_loss': train_loss_epoch_sum, 'Consistency_training_loss':consistency_loss_epoch_sum})


    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 9999
        self.max = -999

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if(val < self.min):
            self.min = val
        if(val > self.max):
            self.max = val

    def __format__(self, format):
        return "{self.val:{format}}".format(self=self, format=format)
        #return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

def calculate_f1_score_batch(prediction, targets):
    #macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    f1_score_data = f1_score(targets, prediction, average="macro")
    return f1_score_data


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())


def write_csv_row_final_results(training_log, labels_batch, highest_testing_accuracy, lowest_mae):
    row_id = "LB_" + str(labels_batch)
    training_log.record(row_id, {'Highest_Testing_accuracy': highest_testing_accuracy, 'Lowest_MAE': lowest_mae})