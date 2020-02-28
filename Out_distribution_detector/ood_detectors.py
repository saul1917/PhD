import torch
import torchvision
import numpy as np
from torch.distributions import Categorical

class OutOfDistributionDetectors:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def testUncertaintyEntropy(self, cnnNet, datasetLoader, datasetName="MINST"):
        # if input dimensions are wrong, it will explode at the top model input
        sumEntropy = []
        # Just load one batch
        print("Processing the observations to calculate its entropy... ")
        for (images, labels) in datasetLoader:
            # takes only one batch of 12 observations
            images, labels = images.to(self.device), labels.to(self.device)
            output = cnnNet(images)
            entropy = self.getEntropy(output)
            sumEntropy += [float(entropy)]
        entropies = np.array(sumEntropy)
        meanEntropy = np.mean(entropies)
        stdEntropy = np.std(entropies)
        print("Mean entropy: ", datasetName)
        print(meanEntropy)
        print("Std entropy", datasetName)
        print(stdEntropy)
        (histogram, buckets) = np.histogram(entropies, range=(0, 1.5), bins = 2000)
        print("histogram")
        print(histogram)
        print("buckets")
        print(buckets)
        return (histogram, meanEntropy, stdEntropy)

    def getEntropy(self, px):
        entropy = Categorical(probs=px).entropy()
        return entropy