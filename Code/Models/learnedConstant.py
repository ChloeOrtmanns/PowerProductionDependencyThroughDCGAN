import torch
import torch.nn.functional as F
import torch.nn as nn

class LearnedConstant(nn.Module):
    # in_c is the number of channels we start with
    def __init__(self, in_c=512):
        super().__init__()

        # We create a PyTorch parameter this is trainable. It has batch size 1
        # The LearnedConstant is initialised to 1 as set out in the paper.
        self.constant = nn.Parameter(torch.ones(1, in_c, 4, 7))

    # When we init the learned constant we pass in the batch size
    def forward(self, batch_size):
        # It is expanded the match the batch_size currently in network, the rest of dims stay the same
        return self.constant.expand(batch_size, -1, -1, -1)