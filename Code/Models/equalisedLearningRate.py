import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# This is the key class
# It applies EqLR not just at intialisation but dynamically throughout training
# This is done through a forward pre-hook, which essentially is a function
# which runs before the forward method
class EqualLR:
    # We init the EqualLR class with the name of the weight parameter
    def __init__(self, name):
        self.name = name
    
    def compute_weight(self, module):
        # getattr = get attribute, in weight we stored the original
        # in the apply method
        weight = getattr(module, self.name + '_orig')
        # weight.data.size(1) is the number of channels (for a conv layer)
        # or the number of input features (for a linear layer)
        # weight.data[0][0].numel() for a linear layer is 1 and for a conv layer
        # it is the kernel size*kernel_size.  
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / (fan_in))

    @staticmethod
    def apply(module, name):
        # We create an instance of EqualLR 
        fn = EqualLR(name)
        
        # The original weight is retrived from the layer
        weight = getattr(module, name)
        # The weight is deleted from the layer
        del module._parameters[name]
        # We register a new parameter with the name _orig
        # saving the original parameters
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        # We call the pre-hook, which runs before the forward pass
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        # We call compute weight before the forward pass
        # When registering the pre_hook this __call__ is what will be called
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    # This is just a simple function to apply the EqualLR regime
    EqualLR.apply(module, name)

    return module

# We redefine the layers simply, initialising the weight with a normal 
# distribution and the biases as 0. 
# By using  *args and **kwargs we allow for full flexibility in these layers
# and we can pass the usual arguments we would to the regular PyTorch versions.
class EqualLRConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLRConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLRLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
    