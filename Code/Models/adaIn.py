import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.equalisedLearningRate import EqualLRLinear

# We define AdaIN as a PyTorch module
class AdaIN(nn.Module):
    # As input we take the number of dimensions we have in our latent space, the number  and the current number of dimensions
    # the current number will change based on the layer. Remember as we grow our G we decrease the number of channels with it starting
    # at 256 and ending at 64
    def __init__(self, latent_dim, current_c):
        super().__init__()

        # Our map layer is a simple Linear layer, we make use of the EqLR linear layer we implemented in the previous post
        self.map_layer = EqualLRLinear(latent_dim, current_c*2)  # Style is 2x current dim, this is stated in the paper
        # Look ahead for why it is 2x current dim.
        # IN constitutes the middle part of Figure 4 (without the y's), the default for InstanceNorm2d is affine=False (gamma and beta
        # are not used when affine=False)
        self.IN = nn.InstanceNorm2d(current_c)

    # As input to our forward pass we have w - the output of the mapping network 
    # and the image (the object being passed through the network)
    def forward(self, w, image):
        # To get out style we pass w through the affine transformation
        style = self.map_layer(w)
        # We need to create y_style and y_bias from the style output. To do this we use the chunk method from PyTorch
        # chunk splits the tensor in 2 down the middle, so we will end up with y_s and y_b having half the number of channels that style 
        # has, this is why we do current_c*2 in the map_layer. 
        y_s, y_b = style.chunk(2, dim=1)
        
        # Reshape y_s and y_b to match image's dimensions, currently the dims of y_s and y_b are (N, C) and image is (N, C, H, W)
        # after the unsqueezes the dims of y_s and y_b will be (N, C, 1, 1)
        y_s = y_s.unsqueeze(2).unsqueeze(3)
        y_b = y_b.unsqueeze(2).unsqueeze(3)

        # Implement the full formula
        return (y_s * self.IN(image)) + y_b