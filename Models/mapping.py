import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.pixelwiseNormalisation import PixelNorm
from Models.equalisedLearningRate import EqualLRLinear

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        self.norm = PixelNorm()
        self.layers = nn.Sequential(
            EqualLRLinear(z_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualLRLinear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.norm(x)
        return self.layers(x)