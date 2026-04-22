import torch
import torch.nn.functional as F
import torch.nn as nn

# We make use of PyTorch modules again, it's a very useful class and we make use of it a lot.
class NoiseLayer(nn.Module):
    # The only input is the number of channels we current have, we need this to ensure our weight parameter matches number of channels
    # in current layer
    def __init__(self, channels):
        super().__init__()

        # This weight is for the "learned" aspect of the scaling
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, gen_image, noise=None):
        # If noise is not initialised for this layer
        if noise is None:
            # We want to match the dims of current model stage
            N, _, H, W = gen_image.shape
            # generate the noise
            noise = torch.randn(N, 1, H, W, device=gen_image.device)

        # The noise is added with a summation
        return gen_image + (noise * self.weight)