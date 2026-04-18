import torch
import torch.nn.functional as F
import torch.nn as nn

from Models.equalisedLearningRate import EqualLRConv2d
from Models.pixelwiseNormalisation import PixelNorm

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.const = nn.Parameter(torch.randn(1, 512, 4, 7))

        self.block_4x7     = G_ConvBlock(512, 512, 3, 1, upsample=False)
        self.block_8x13    = G_ConvBlock(512, 512, 3, 1, upsample_size=(8,   13))
        self.block_16x26   = G_ConvBlock(512, 256, 3, 1, upsample_size=(16,  26))
        self.block_32x52   = G_ConvBlock(256, 128, 3, 1, upsample_size=(32,  52))
        self.block_64x104  = G_ConvBlock(128, 64,  3, 1, upsample_size=(64,  104))
        self.block_128x208 = G_ConvBlock(64,  32,  3, 1, upsample_size=(128, 208))

        # The to_out layer outputs 2 feature maps, i.e. solar & wind energy
        self.to_out_4 = EqualLRConv2d(512, 2, 1)
        self.to_out_8 = EqualLRConv2d(512, 2, 1)
        self.to_out_16 = EqualLRConv2d(256, 2, 1)
        self.to_out_32 = EqualLRConv2d(128, 2, 1)
        self.to_out_64 = EqualLRConv2d(64, 2, 1)
        self.to_out_128 = EqualLRConv2d(32, 2, 1)
        
        # We use tanh for the output activation, this bounds our pixels values between [-1,1]
        self.tanh = nn.Tanh()

    def forward(self, batch_size, layer_num, alpha):
        x = self.const.expand(batch_size, -1, -1, -1)

        # The first layer is simple, we just pass it through the 4x4 block 
        # and if we are currently on layer_1 we pass it through to_out and call it a day
        out_4 = self.block_4x7(x)
        if layer_num == 1:
            out = self.to_out_4(out_4)
            out = self.tanh(out)
            return out
        
        # Being the second layer we have a previous layer available and it must be used in our progressive growing
        # We pass the out_4 through out_8 
        out_8 = self.block_8x13(out_4)
        if layer_num == 2:
            skip = F.interpolate(self.to_out_4(out_4), size=(8, 13), mode='bilinear', align_corners=False)
            out  = self.to_out_8(out_8)
            return self.tanh((1 - alpha) * skip + alpha * out)

        out_16 = self.block_16x26(out_8)
        if layer_num == 3:
            skip = F.interpolate(self.to_out_8(out_8), size=(16, 26), mode='bilinear', align_corners=False)
            out  = self.to_out_16(out_16)
            return self.tanh((1 - alpha) * skip + alpha * out)

        out_32 = self.block_32x52(out_16)
        if layer_num == 4:
            skip = F.interpolate(self.to_out_16(out_16), size=(32, 52), mode='bilinear', align_corners=False)
            out  = self.to_out_32(out_32)
            return self.tanh((1 - alpha) * skip + alpha * out)

        out_64 = self.block_64x104(out_32)
        if layer_num == 5:
            skip = F.interpolate(self.to_out_32(out_32), size=(64, 104), mode='bilinear', align_corners=False)
            out  = self.to_out_64(out_64)
            return self.tanh((1 - alpha) * skip + alpha * out)

        out_128 = self.block_128x208(out_64)
        if layer_num == 6:
            skip = F.interpolate(self.to_out_64(out_64), size=(128, 208), mode='bilinear', align_corners=False)
            out  = self.to_out_128(out_128)
            out  = self.tanh((1 - alpha) * skip + alpha * out)
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

        self.conv1 = EqualLRConv2d(in_c,  out_c, ksize1, padding=padding)
        self.conv2 = EqualLRConv2d(out_c, out_c, ksize2, padding=padding2)
        self.pnorm = PixelNorm()
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, size=self.upsample_size,
                              mode='bilinear', align_corners=False)
        x = self.act(self.pnorm(self.conv1(x)))
        x = self.act(self.pnorm(self.conv2(x)))
        return x