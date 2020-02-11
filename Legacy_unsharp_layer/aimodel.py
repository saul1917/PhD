
import torch.nn as nn
import torchvision.models as models

from resnet_h import *

def predefined_model(args):
    # create model
    '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    '''
    #model = resnet18_usm(pretrained=True)
    model = inception_v3_usm(pretrained=True)
    return model


def create_model(model, num_classes, use_cuda):

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if use_cuda:
        model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    return model
