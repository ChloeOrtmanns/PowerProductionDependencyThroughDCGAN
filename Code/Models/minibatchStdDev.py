import torch
import torch.nn as nn

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        # Group size is from the github repo I linked to
        # It isn't discussed at all in the paper AFAIK
        self.group_size = group_size
    
    def forward(self, x):
        N, C, H, W = x.shape  # N = num feature maps, C = num channels, H = height, W = width
        G = min(self.group_size, N)  # the minibatch must be divisible by group_size
        
        # Here we split up X into groups
        # This line may be a little weird, expand the next code box to explore this line!
        y = x.view(G, -1, C, H, W)
        
        # The 3 following lines see us implement number 1 from the list above.
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(torch.square(y), dim=0)
        y = torch.sqrt(y + 1e-8)
        
        # This is number 2
        y = torch.mean(y, dim=[1,2,3], keepdim=True)
        
        # Finally this is number 3
        y = y.repeat(G, 1, H, W)
        
        # We return the input x with an additional feature map
        return torch.cat([x,y], dim=1)