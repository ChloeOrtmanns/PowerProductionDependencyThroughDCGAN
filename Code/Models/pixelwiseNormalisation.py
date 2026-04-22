import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    # the term inside the sqrt is a summation over all feature maps
    # divided by all feature maps, a.k.a taking the mean!
    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8
        )