
import torch, os, torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from model_trainer import ModelTrainerCIFAR10
from model_trainer import LeNet
from ood_detectors import OutOfDistributionDetectors


DEFAULT_IMAGENET_TINY = "/media/Data/saul/Datasets/ImagenetTinyValidation"
DEFAULT_SIZE = 32
DEFAULT_BATCH_SIZE = 1

class OOD_Tester:
    def __init__(self):
        self.model_trainer_cifar10 = ModelTrainerCIFAR10()
        self.ood_detectors = OutOfDistributionDetectors()


    def create_dataset_loader_imagenet_tiny_resize(self, data_dir = DEFAULT_IMAGENET_TINY):
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.transforms.Resize((DEFAULT_SIZE, DEFAULT_SIZE)), transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

        num_workers = { 'train': 100, 'val': 0, 'test': 0 }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
        dataloadersImagenetTiny = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=num_workers[x]) for x in ['val']}

        print(dataloadersImagenetTiny['val'])
        return dataloadersImagenetTiny['val']

    def create_dataset_loader_cifar10(self):
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=20)
        return trainloader

    def create_dataset_MINST(self):
        #images have to be converted to RGB usually
        transformations = transforms.Compose(
            [transforms.transforms.Resize((DEFAULT_SIZE, DEFAULT_SIZE)),  transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])
        mnistTestset = datasets.MNIST(root='./data', train=False, download=True, transform=transformations)
        datasetLoader = torch.utils.data.DataLoader(mnistTestset, batch_size=1, shuffle=True, num_workers=1)
        return datasetLoader

    def test1_cifar10_tiny_imagenet_resize(self):
        leNet = LeNet()
        self.model_trainer_cifar10.load_net(leNet, "./model_weights/CIFAR_10_LeNet_10_epochs.pth")
        dataset_loader_imagenet = self.create_dataset_loader_imagenet_tiny_resize()
        (histogram_imagenet, meanEntropy_imagenet, stdEntropy_imagenet) = self.ood_detectors.testUncertaintyEntropy(leNet, dataset_loader_imagenet, datasetName="Imagenet")
        dataset_loader_cifar10 = self.create_dataset_loader_cifar10()
        (histogram_cifar10, meanEntropy_cifar10, stdEntropy_cifar10) = self.ood_detectors.testUncertaintyEntropy(leNet, dataset_loader_cifar10, datasetName="CIFAR 10")
        dataset_loader_minst = self.create_dataset_MINST()
        (histogram_mnist, meanEntropy_mnist, stdEntropy_mnist) = self.ood_detectors.testUncertaintyEntropy(leNet, dataset_loader_minst, datasetName="MNIST")




if __name__ == '__main__':
    ood_tester = OOD_Tester()
    ood_tester.test1_cifar10_tiny_imagenet_resize()
