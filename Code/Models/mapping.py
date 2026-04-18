import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.pixelwiseNormalisation import PixelNorm
from Models.equalisedLearningRate import EqualLRLinear

class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # We normalise the input with PixelNorm
        self.norm = PixelNorm()
        
        # I set output to 256 to match the shortened G case
        # As we just want to pass the input through the whole network, I use a PyTorch Sequential
        # it allows us to pass our input through multiple layers with a single line in the forward method.
        self.layers = nn.Sequential(
            # I set the dimensionality of each layer to be the same. Perhaps you could change the hidden layer dimensions
            # and see what happens.
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        # Normalise input latent
        x = self.norm(x)

        # Using the Sequential pass the normalised input through the network
        out = self.layers(x)
        
        return out