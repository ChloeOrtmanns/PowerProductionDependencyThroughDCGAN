# Before we continue lets set our inputs and configure the device for our model code
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import datasets, transforms, utils

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from datetime import datetime
import os
from math import sqrt

from Code.Models.generator import Generator
from Code.Models.discriminator import Discriminator

# We can make use of a GPU if you have one on your computer. This works for Nvidia and M series GPU's
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # These 2 lines assign some data on the memory of the device and output it. The output confirms
    # if we have set the intended device
    x = torch.ones(1, device=device)
    print (x)
elif torch.backends.cuda.is_built():
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print (x)
else:
    device = ("cpu")
    x = torch.ones(1, device=device)
    print (x)

# Initialize with He's initializer
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # You see nn.LayerNorm here, this will come in handy in the D model
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# We init a function to calculate the EMA for our parameters.
# This is implemented using two model, one the training G and the other
# a G we do not train but only use to hold the weights after perfoming the EMA
def EMA(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

###Generator
g = Generator().to(device)
# g.apply(init_weights)

###Testing
g_out = g(batch_size=1, layer_num=1, alpha=0.5)

###Discriminator
d = Gulrajani_Discriminator().to(device)
d.apply(init_weights)

###Testing
d_in = torch.randn((1, 2, 64, 64), device=device)
d_out = d(d_in)


