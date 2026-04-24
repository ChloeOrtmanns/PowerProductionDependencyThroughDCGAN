import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.equalisedLearningRate import EqualLRConv2d, EqualLRLinear
from Models.minibatchStdDev import MiniBatchStdDev

class Discriminator(nn.Module):
    def __init__(self, out_c=256):
        super().__init__()

        # Mirrors the Generator's 6 stages, but in reverse order.
        # D goes from high resolution down to 4×7.
        # Index 0 is always where we start (highest resolution currently active),
        # index 6 is always where we finish (4×7 + minibatch stddev).
        self.blocks = nn.ModuleList([
            D_ConvBlock(out_c//8, out_c//8, 3, 1),           # 6 - 128×208
            D_ConvBlock(out_c//8, out_c//4, 3, 1),           # 5 - 64×104
            D_ConvBlock(out_c//4, out_c//2, 3, 1),           # 4 - 32×52
            D_ConvBlock(out_c//2, out_c,   3, 1),            # 3 - 16×26
            D_ConvBlock(out_c,    out_c,   3, 1),            # 2 - 8×13
            D_ConvBlock(out_c,    out_c,   3, 1),            # 1 - 4×7
            D_ConvBlock(out_c+1,  out_c,   3, 1, 4, 0, mbatch=True),  # 0 - final
        ])

        # output channels match the block each stage feeds into
        self.from_out = nn.ModuleList([
            EqualLRConv2d(2, out_c//8, 1),   # for 128×208
            EqualLRConv2d(2, out_c//8, 1),   # for 64×104
            EqualLRConv2d(2, out_c//4, 1),   # for 32×52
            EqualLRConv2d(2, out_c//2, 1),   # for 16×26
            EqualLRConv2d(2, out_c,    1),   # for 8×13
            EqualLRConv2d(2, out_c,    1),   # for 4×7
            EqualLRConv2d(2, out_c,    1),   # unused but keeps indexing consistent
        ])

        self.num_layers = len(self.blocks)
        self.linear = EqualLRLinear(out_c, 1)


    def forward(self, x, layer_num, alpha):
        # x arrives at the current active resolution, e.g. at layer_num=6
        # x is (B, 2, 121, 201) — pad to 128×208 first
        if layer_num == 6:
            x = F.pad(x, (0, 7, 0, 7), mode='reflect')
            # → (B, 2, 128, 208)

        for i in reversed(range(layer_num)):
            idx = self.num_layers - i - 1

            if i + 1 == layer_num:
                out = self.from_out[idx](x)

            out = self.blocks[idx](out)

            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i + 1 == layer_num and 0 <= alpha < 1:
                    skip = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip = self.from_out[idx + 1](skip)
                    out = (1 - alpha) * skip + alpha * out

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class D_ConvBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        ksize1, 
        padding, 
        ksize2=None, 
        padding2=None,
        stride=None,   
        mbatch=None,
    ):
        super().__init__()
        
        layers_list = []
        
        if ksize2 is None:
            ksize2 = ksize1
        if padding2 is None:
            padding2 = padding
        
        if mbatch:
            layers_list.extend([
                MiniBatchStdDev(),
            ])
            
        layers_list.extend([
            EqualLRConv2d(in_c, out_c, ksize1, padding=padding),
            nn.LeakyReLU(0.2),
            EqualLRConv2d(out_c, out_c, ksize2, padding=padding2),
            nn.LeakyReLU(0.2),
        ])
        
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x