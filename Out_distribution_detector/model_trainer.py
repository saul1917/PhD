import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

#Test and training loaders


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = torch.nn.Softmax(x)
        #CORREGIR! NO ES LOG SOFTMAX, ES SOFTMAX!
        output = F.log_softmax(x, dim=1)
        #for Softmax output
        output = output.exp()
        return output

class ModelTrainerCIFAR10:

    def __init__(self, net = None):
        self.lr = 0.0001
        self.momentum = 0.9
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
                                                  shuffle=True, num_workers=20)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                 shuffle=False, num_workers=20)

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if(net != None):
            self.net = net
            self.net.to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)





    def load_net(self, net_to_load, path):
        net_to_load.load_state_dict(torch.load(path))
        self.net = net_to_load
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)

        print("Network loaded! ", path)

    def save_net(self, path = './cifar_net.pth'):
        torch.save(self.net.state_dict(), path)
        print("Network saved! ", path)



    def train_model(self, epochs):
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def test_model(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                accuracy))
        return accuracy

if __name__ == '__main__':
    lenet = LeNet()
    model_trainer_lenet = ModelTrainerCIFAR10(lenet)
    model_trainer_lenet.train_model(epochs = 100)
    model_trainer_lenet.test_model()
    model_trainer_lenet.save_net("CIFAR_10_LeNet_100_epochs.pth")

