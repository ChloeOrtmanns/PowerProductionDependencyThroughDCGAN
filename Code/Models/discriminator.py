
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: (2, 121, 201)

            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # ~60x100
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),  # ~30x50
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),  # ~15x25
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),  # ~7x12
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final collapse
            nn.Conv2d(ndf*8, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):
        for i, layer in enumerate(self.main):
            x = layer(x)
            if debug:
                print(f"Layer {i} ({type(layer).__name__}): {x.shape}")
        return x