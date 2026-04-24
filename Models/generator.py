import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from Models.equalisedLearningRate import EqualLRConv2d
from Models.pixelwiseNormalisation import PixelNorm
from Models.mapping import MappingNetwork
from Models.learnedConstant import LearnedConstant
from Models.noise import NoiseLayer
from Models.adaIn import AdaIN

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.g_mapping = MappingNetwork()  # z_dim=512, w_dim=512

        self.block_4x7     = G_ConvBlock(512, 512, 3, 1, upsample=False)
        self.block_8x13    = G_ConvBlock(512, 512, 3, 1, upsample_size=(8,   13))
        self.block_16x26   = G_ConvBlock(512, 256, 3, 1, upsample_size=(16,  26))
        self.block_32x52   = G_ConvBlock(256, 128, 3, 1, upsample_size=(32,  52))
        self.block_64x104  = G_ConvBlock(128, 64,  3, 1, upsample_size=(64,  104))
        self.block_128x208 = G_ConvBlock(64,  32,  3, 1, upsample_size=(128, 208))

        self.to_out_4   = EqualLRConv2d(512, 2, 1)
        self.to_out_8   = EqualLRConv2d(512, 2, 1)
        self.to_out_16  = EqualLRConv2d(256, 2, 1)
        self.to_out_32  = EqualLRConv2d(128, 2, 1)
        self.to_out_64  = EqualLRConv2d(64,  2, 1)
        self.to_out_128 = EqualLRConv2d(32,  2, 1)

        self.tanh = nn.Tanh()

    def forward(self, z, layer_num, alpha):
        w = self.g_mapping(z)

        # style mixing — same as tutorial
        if torch.rand(1).item() < 0.9:
            z2 = torch.randn_like(z)
            w2 = self.g_mapping(z2)
            crossover_point = random.randint(1, layer_num)
        else:
            crossover_point = None

        if crossover_point and 1 >= crossover_point:
            w = w2
        out_4 = self.block_4x7(w)
        if layer_num == 1:
            return self.tanh(self.to_out_4(out_4))

        if crossover_point and 2 >= crossover_point:
            w = w2
        out_8 = self.block_8x13(w, out_4)
        if layer_num == 2:
            skip = F.interpolate(self.to_out_4(out_4), size=(8, 13), mode='bilinear', align_corners=False)
            return self.tanh((1 - alpha) * skip + alpha * self.to_out_8(out_8))

        if crossover_point and 3 >= crossover_point:
            w = w2
        out_16 = self.block_16x26(w, out_8)
        if layer_num == 3:
            skip = F.interpolate(self.to_out_8(out_8), size=(16, 26), mode='bilinear', align_corners=False)
            return self.tanh((1 - alpha) * skip + alpha * self.to_out_16(out_16))

        if crossover_point and 4 >= crossover_point:
            w = w2
        out_32 = self.block_32x52(w, out_16)
        if layer_num == 4:
            skip = F.interpolate(self.to_out_16(out_16), size=(32, 52), mode='bilinear', align_corners=False)
            return self.tanh((1 - alpha) * skip + alpha * self.to_out_32(out_32))

        if crossover_point and 5 >= crossover_point:
            w = w2
        out_64 = self.block_64x104(w, out_32)
        if layer_num == 5:
            skip = F.interpolate(self.to_out_32(out_32), size=(64, 104), mode='bilinear', align_corners=False)
            return self.tanh((1 - alpha) * skip + alpha * self.to_out_64(out_64))

        if crossover_point and 6 >= crossover_point:
            w = w2
        out_128 = self.block_128x208(w, out_64)
        if layer_num == 6:
            skip = F.interpolate(self.to_out_64(out_64), size=(128, 208), mode='bilinear', align_corners=False)
            out  = self.tanh((1 - alpha) * skip + alpha * self.to_out_128(out_128))
            return out[:, :, :121, :201]

        # print(f"x:          {x.shape}")
        # print(f"out_4x7:    {out_4x7.shape}")
        # print(f"out_8x13:   {out_8x13.shape}")
        # print(f"out_16x26:  {out_16x26.shape}")
        # print(f"out_32x52:  {out_32x52.shape}")
        # print(f"out_64x104: {out_64x104.shape}")
        # print(f"out_128x208:{out_128x208.shape}")

class G_ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ksize1, padding,
                 ksize2=None, padding2=None, upsample=True, upsample_size=None):
        super().__init__()

        if ksize2 is None:   ksize2 = ksize1
        if padding2 is None: padding2 = padding

        self.upsample      = upsample
        self.upsample_size = upsample_size
        layers_list = []

        if upsample:
            layers_list.extend([
                # explicit size instead of scale_factor=2
                nn.Upsample(size=upsample_size, mode='bilinear'),
                EqualLRConv2d(in_c, out_c, ksize1, padding=padding),
                NoiseLayer(out_c),
                AdaIN(512, out_c),  # 512 instead of 256
            ])
        else:
            # first block — learned constant, no conv before AdaIN
            self.learned_constant = LearnedConstant(in_c)  # 4×7 handled inside LearnedConstant
            layers_list.extend([
                NoiseLayer(in_c),
                AdaIN(512, in_c),   # 512 instead of 256
            ])

        layers_list.extend([
            nn.LeakyReLU(0.2),
            EqualLRConv2d(out_c, out_c, ksize2, padding=padding2),
            NoiseLayer(out_c),
            AdaIN(512, out_c),      # 512 instead of 256
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.ModuleList(layers_list)

    def forward(self, w, x=None):
        if not self.upsample:
            x = self.learned_constant(w.size(0))

        for layer in self.layers:
            if isinstance(layer, AdaIN):
                x = layer(w, x)
            else:
                x = layer(x)

        return x