
from __future__ import print_function
import argparse
#Pytorch library import
import torch
import torch.nn as nn
from torch.autograd import Variable

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://github.com/pytorch/examples/blob/master/mnist/main.py
import torch.nn.functional as F
import torch.optim as optim
import numpy as np;
import pandas as pandas;
from scipy import ndimage
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from DataSetLoaderCarActivities import DatasetCarsActivities
from CNN import ConvolutionalNeuralNetwork

DEFAULT_SIZE = 256
DEFAULT_PATH = "C:/Users/saul1/Desktop/Datasets_TEMP/auc.distracted.driver.dataset_v2/v1_cam1_no_split"
BATCH_SIZE = 30
WEIGHTS_FILE = "DEFAULT_WEIGHTS"
DEFAULT_ERROR_PRINT = 10

class ModelManager:
    def loadData(self, numberClasses, args = [], kwargs = []):
        """
        Creates the dataset loaders
        :param args, console argument names
        :param kwargs, data arguments
        :return training and test dataset loaders
        """
        self.numberClasses = numberClasses
        batchSize = BATCH_SIZE
        print("args!!!")
        print(args)
        if(args != []):
            batchSize = args.batch_size

        """
        Data augmentation
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
        INPUT IMAGE DIMENSIONS IN DENSENET?
        """

        # Normalize the dataset and transform to grayscale
        transformations = transforms.Compose(
            [transforms.transforms.Resize((DEFAULT_SIZE , DEFAULT_SIZE )), transforms.ToTensor(), transforms.Normalize([0.485], [0.229])])
        #load the datasets
        dataSetTraining = DatasetCarsActivities(DEFAULT_PATH, transformations)
        dataSetValidation = DatasetCarsActivities(DEFAULT_PATH, transformations)
        #creates the dataset loaders
        loaderTrainingSamples = torch.utils.data.DataLoader(dataset = dataSetTraining, batch_size= batchSize, shuffle=True)
        loaderTestSamples = torch.utils.data.DataLoader(dataset = dataSetValidation, batch_size = batchSize, shuffle = True)
        #returns both of them
        return (loaderTrainingSamples, loaderTestSamples)

    def createModel(self, args, device):
        """
        Creates the model, its optimizer with a specific criterion
        :param args:
        :param device:
        :return:
        """
        #model creation
        model = ConvolutionalNeuralNetwork(self.numberClasses).to(device)
        #loss function
        criterion = nn.NLLLoss()
        #train all the parameters
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
        return (model, optimizer, criterion)

    def trainModel(self, loadWeights, device, args, loaderTraining, loaderTest):
        """
        Train the model
        :param loadWeights:
        :param device:
        :param args:
        :param loaderTraining:
        :param loaderTest:
        :return:
        """
        # se tratan de cargar los pesos
        #loadWeights = args.loadWeights

        epochs = args.epochs
        # create the model with the optimizer and the loss function
        (model, optimizer, criterion) = self.createModel(args, device)
        if (loadWeights):
            try:
                model.load_state_dict(torch.load(WEIGHTS_FILE))
            except:
                print("Cannot load the weights")
        #for each epoch...
        for epoch in range(0, epochs):
            #iterate through different batches from the dataset
            for idLote, (images, labels) in enumerate(loaderTraining):
                images = Variable(images)
                labels = Variable(labels)
                labels = labels.long()
                # save labels and samples to the device
                inputs, labels = images.to(device), labels.to(device)
                # put weights to zero
                optimizer.zero_grad()
                # estimated output by the model
                estimatedOutput = model.forward(inputs)
                # calculate model loss
                loss = criterion(estimatedOutput, labels)
                # back propagate error
                loss.backward()
                # update weigths
                optimizer.step()
                # train mode
                model.train()
                if idLote % 100 == 0:
                    print('Epoch de entrenamiento: {} [{}/{} ({:.0f}%)]\tPerdida: {:.6f}'.format(
                        epochs, idLote * len(images), len(loaderTraining.dataset),
                                100. * idLote / len(loaderTraining), loss.item()))
                    # guarda el modelo
                    torch.save(model.state_dict(), WEIGHTS_FILE)

    def createArgumentParser(self):
        """
        :return: the argument parser
        """
        useCUDA = torch.cuda.is_available()

        # Configuracion del entrenamiento
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        # Tamanio del lote de entrenamiento
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        # Tamanio del lote de pruebas
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        # Cantidad de epochs o iteraciones
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        # Coeficiente de aprendizaje
        parser.add_argument('--lr', type=float, default=1000, metavar='LR',
                            help='learning rate (default: 0.001)')
        # Momentum
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        # CUDA
        parser.add_argument('--no-cuda', action='store_true', default=True,
                            help='disables CUDA training')
        # Semilla?
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # Intervalo en volcar a la bitacora
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

        # Carga de pesos
        parser.add_argument('--loadWeights', type=int, default=False, metavar='W',
                            help='Load the weights from file')

        args = parser.parse_args(args=[])
        kwargs = {'num_workers': 1, 'pin_memory': True} if useCUDA else {}

        return args, kwargs



def testModelManagerDataLoader():
    modelManager = ModelManager()
    (args,kwargs) = modelManager.createArgumentParser()


    (loaderTrainingSamples, loaderTestSamples) = modelManager.loadData(args)
    
    for i, (images, labels) in enumerate(loaderTrainingSamples):
        #takes only one batch of 12 observations
        images = Variable(images)
        labels = Variable(labels)
        print("Dimensions of IMAGES and LABELS")
        print(images.shape)
        print(labels.shape)
        break

def testRunModel():
    #useCUDA = torch.cuda.is_available()
    numberClasses = 11

    modelManager = ModelManager()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    (args, kwargs) = modelManager.createArgumentParser()
    (loaderTrainingSamples, loaderTestSamples) = modelManager.loadData(args)
    modelManager.trainModel(device, 0, args, loaderTrainingSamples, loaderTrainingSamples)

testRunModel()