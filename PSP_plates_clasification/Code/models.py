import torch
import torch.nn as nn
from ResNetUSM import *
from torchvision import models

def get_cnn(num_classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("args.pretrained: " + str(args.pretrained))
    if(args.arch.startswith("resnet") or args.arch.startswith("inception")):
        if(args.arch == "resnet18"):
            model_ft = models.resnet18(pretrained=args.pretrained)              # Load the pretrained model from pytorch
        elif(args.arch == "resnet50"):
            model_ft = models.resnet50(pretrained=args.pretrained)
        elif(args.arch == "resnetusm"):
            model_ft = resnet18_usm(pretrained=args.pretrained, cuda=args.cuda)

        elif(args.arch == "inception_v3"):
            print("Using Inception v3")
            model_ft = models.inception_v3(pretrained=args.pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif(args.arch == "vgg19_bn"):
        model_ft =  models.vgg19_bn(pretrained=args.pretrained)                 # Load the pretrained model from pytorch
        num_features = model_ft.classifier[6].in_features                       # Newly created modules have require_grad=True by default
        features = list(model_ft.classifier.children())[:-1]                    # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)])                 # Add our layer with 4 outputs
        model_ft.classifier = nn.Sequential(*features)                          # Replace the model classifier

    model_ft = model_ft.to(device)
    return model_ft


def get_optimizer(model_ft, args):
    if(args.arch == "resnetusm"):
        optimizer_ft = torch.optim.SGD([{'params':model_ft.filter.parameters(),
                                'lr':args.lr*10},
                            {'params':model_ft.filter_conv1.parameters(),
                                'lr':args.lr*10},
                            {'params': model_ft.conv1.parameters()},
                            {'params': model_ft.layer1.parameters()},
                            {'params': model_ft.layer2.parameters()},
                            {'params': model_ft.layer3.parameters()},
                            {'params': model_ft.layer4.parameters()},
                            {'params': model_ft.fc.parameters()}],
                              lr = args.lr,
                              momentum = args.momentum)
    else:
        optimizer_ft = torch.optim.SGD(model_ft.parameters(),
                            lr = args.lr, momentum = args.momentum)
    return optimizer_ft
