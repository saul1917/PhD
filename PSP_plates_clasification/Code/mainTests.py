import torch
import argparse
from utils import *
from models import *
from mode_validation import *
from train_and_eval import *
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
import time
import pandas as pd

RESULTS_PATH = "Results/"

"""
The parameters to use when the file is called
Create each parameter with type and default (with a help comment)
Args: void
Returns: the parser
"""
def make_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--arch',   default='resnet18',
                        choices=['resnet18', 'resnet50', 'vgg19_bn','inception_v3',
                                 'resnetusm'],
                        help='The network architecture')
    parser.add_argument('--val_mode', default='k-fold',
                        choices=['once', 'k-fold', 'randomsampler'],
                        help='Type of validation you want to use: f.e: k-fold')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Input batch size for using the model')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--k_fold_num', type=int, default=5,
                        help='Number of folds you want to use for k-fold validation')
    parser.add_argument('--val_num', type=int, default=20,
                        help='Number of times you want to run the model to get a mean')
    parser.add_argument('--random_seed', type=int, default=16031997,
                        help='Random seed to shuffle the dataset')
    parser.add_argument('--data_dir', default='/media/Data/saul/PSP_dataset12/',
                        help='Directory were you take images, they have to be separeted by classes')
    parser.add_argument('--cuda', type=str2bool, default= torch.cuda.is_available(),
                        help='Use gpu by cuda')
    parser.add_argument('--shuffle_dataset', type=str2bool, default=True,
                        help='Number of folds you want to use for k-fold validation')
    parser.add_argument('--pretrained', type=str2bool, default=True,
                        help='The model will be pretrained with ')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weights', default="",
                        help='The .pth doc to load as weights')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='The .pth doc to load as weights')
    parser.add_argument('--num_epochs', type=int, default=7, help='number of epochs to train for')
    parser.add_argument('--save', type=str2bool, default=False,
                        help='If you want to save weights and csvs from the trained model or evaluation')
    parser.add_argument('--image', default="",
                        help='The source file path of image you want to process')
    parser.add_argument('--folder', default="",
                        help='The folder where you want to save ')
    return parser
"""
Stats for all experiments reported
"""
def writeReport(sensitivities, specificities, accuracies, folds, experiments, args):
    # dictionary of lists

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)
    mean_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)
    dictionary = {"Experiment":experiments, "Fold":folds, 'Sensitivity': sensitivities, 'Specificity': specificities, 'Accuracy': accuracies}

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fileNameReportCSV =  RESULTS_PATH + str(args.arch) + "_" + timestr + ".csv"
    fileNameReportTXT = RESULTS_PATH + str(args.arch) + "_" + timestr + ".txt"
    #write the report
    fileReportTXT = open(fileNameReportTXT, "w")
    fileReportTXT.write(str(args))
    fileReportTXT.close()
    dataFrame = pd.DataFrame(dictionary)
    dataFrame.to_csv(fileNameReportCSV)


def main():

    newTotalAccuracy = []
    meanAccuracyExperimentsList = []
    meanSensitivityExperimentsList = []
    meanSpecificityExperimentsList = []
    parser = make_parser()                                                      #creates the parser
    args = parser.parse_args()

    #writeReport([20, 3, 4], [200, 30, 40], [2000, 300, 400], args)
    #takes the arguments from console (with the help of the parser)
    if(args.weights != "" and args.image != ""):                                #we have weights and an image -> that means evaluate an image
        print("Soon")
    else:
        size = size_by_archs(args.arch)                                         #get the correspondant size for the architecture
        batch_size = args.batch_size                                            #setting variables
        num_workers = args.workers
        random_seed = args.random_seed
        data_transforms = transforms.Compose([                                  #transformations applied to all images
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.0466, 0.1761, 0.3975], [0.8603, 0.8790, 0.8751])
        ])

        data_dir =  args.data_dir                                               #where the dataset is located (needs to be separated by class)
        dataset = datasets.ImageFolder(data_dir, data_transforms)               #A generic data loader where the images are arranged by classes


        numberTotalExperiments = args.val_num
        print("Total number of experiments: ", numberTotalExperiments)

        total = 0
        final_acc =0
        std = 0                                                       #variables to report
        sensitivity = 0
        specificity = 0
        folds = []
        experiments = []
        #accumulate accuracies
        total = []
        for experimentNum in range(0, numberTotalExperiments):
            print("Experiment number ", experimentNum, " ----------------------------")
            dataset_size = len(dataset)
            indices = list(range(dataset_size))                                     #make an index to later associate them to each item
            final_indices = get_indices(indices, args)

            fold = 0
            specificity = []
            sensitivity = []                           #depending in the validation mode, there will be different distribuitions and how many sets will be used
            for train_indices, val_indices in final_indices:                        #final indices is a generator item, that has train indices and validation indices (if is default k fold will be 4 sets of each and default random subsampling will be 10 sets )
                experiments += [experimentNum]
                folds += [fold]
                print("Fold number: ", fold, ".........................")
                fold += 1
                #print(train_indices)
                if(args.val_mode=="k-fold"):
                    np.random.seed(args.random_seed)                                #the indices are shuffled again so the training is random (with the seed) because it's in order
                    np.random.shuffle(train_indices)
                    np.random.shuffle(val_indices)
                train_sampler = SubsetSampler(train_indices)                        #creates samplers for the dataloaders
                valid_sampler = SubsetSampler(val_indices)
                train_loader = torch.utils.data.DataLoader(dataset,                 #takes the sampler and the dataset to make it for pytorch
                                                           batch_size=batch_size, sampler=train_sampler,
                                                           num_workers=num_workers)
                validation_loader = torch.utils.data.DataLoader(dataset,
                                                                batch_size=batch_size, sampler=valid_sampler,
                                                                num_workers=num_workers)

                dataloaders = {'train':train_loader, "val":validation_loader}       #we used the dataloaders in dictionaries for better management
                dataset_sizes = {'train':len(train_sampler), "val":len(valid_sampler)}#we used the datasets sizes in dictionaries for better management
                print(dataset_sizes)
                class_names = ['no', 'yes']                                         #important to define the classes for prediction
                model_ft = get_cnn(len(class_names), args)                          #retrieves the cnn - architecture to be used
                criterion = nn.CrossEntropyLoss()                                   #creates the criterion (used in training and testing)
                optimizer_ft = get_optimizer(model_ft, args)                        #changes the weights based on error (using Stochastic Gradient Descent)
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#helps with the learning rate, to be zigzagging to get into the right function
                print(args.train)
                if(args.train==True):
                    print("About to train for one epoch ")

                    #bug correction 1
                    #model, best_acc, epochs_details_list


                    model_ft, best_acc, epochs_details_list, results = train_model(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, args)    #trains the model
                    print("Yielded results ")
                    #save epochs results to harddrive
                    #save_epochs_results(epochs_details_list, dataset_size, args)
                    print("best_acc: " + str(best_acc))
                    print("el acc directo")
                    acc, sen, spe, confusion_matrix = print_stats(results, class_names)                                                               #prints acc, sensitivity, specificity, confusion_matrix
                    results_detailed_list, acc = eval_model_dataloader(model_ft, dataloaders['val'], dataset_sizes, criterion, class_names, args)                   #evaluates the model to get acc and lists
                    total.append(best_acc)                                          #adding to create mean value from: accuracy
                    sensitivity.append(sen)                                         #sensitivity
                    specificity.append(spe)                                         #specificity
                    if(args.save == True):                                          #we save in files the weights(.pth) and epochs results(.csv)
                        save_model(model_ft, dataset_size, args)
                        save_epochs_results(epochs_details_list, dataset_size, args)
                else:
                    if(args.weights != "" and args.image == ""):                    #evaluate the given wieghts with the validation set
                        model_ft = load_model(model_ft, args.weights)               #load the model with weights
                        results_detailed_list, acc = eval_model_dataloader(model_ft, dataloaders['val'], dataset_sizes, criterion, class_names, args)#evaluate the model
                        total.append(best_acc)                                      #adding to create mean value from: accuracy
                        acc, sen, spe, confusion_matrix = print_stats(results_detailed_list, class_names)#prints acc, sensitivity, specificity, confusion_matrix
                        final_acc = np.mean(total)
                        std = np.std(total)
                        if(args.save == True):
                            save_best_epoch_results(results_detailed_list, dataset_size, args)#save a csv with the best epoch results

                    else:
                        print("There are no source file path for the weights")
                        exit(0)

            #estimates the mean depending in the validation model
            if(args.val_mode == "once"):
                print("once")
            elif(args.val_mode == "k-fold"):
                print("kfold: " + str(args.k_fold_num))
                #totalNp = np.array(total.cpu())
                print("Total ................")
                print(total)
                #results are in the cpu


                print("newTotalAccuracy ................")
                print(newTotalAccuracy)
                mean_acc = np.mean(newTotalAccuracy)
                std_acc = np.std(newTotalAccuracy)
                specificity_mean = np.mean(specificity)
                sensitivity_mean = np.mean(sensitivity)

                meanSensitivityExperimentsList += specificity
                meanSpecificityExperimentsList += sensitivity
            else:
                print("Final acc: " +  str(mean_acc))
                print("Standard deviation: " + str(std))
                print("Final specificity: " +  str(np.mean(specificity)))
                print("Final sensitivity: " +  str(np.mean(sensitivity)))
            #empty memory
            del model_ft
            torch.cuda.empty_cache()
        # write results
        for i in total:
            element = i.cpu().numpy()
            # print(type(element[0]))
            #bug, conversion!!
            newTotalAccuracy += [float(element)]
        print("Accuracies to send........................")
        meanAccuracyExperimentsList += newTotalAccuracy
        print(meanAccuracyExperimentsList)
        writeReport(meanSensitivityExperimentsList, meanSpecificityExperimentsList, meanAccuracyExperimentsList, folds, experiments, args)


main()
