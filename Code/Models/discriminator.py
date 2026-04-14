
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: (2, 121, 201)

            spectral_norm(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)),  # ~60x100
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False)),  # ~30x50
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False)),  # ~15x25
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False)),  # ~7x12
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Final collapse

            spectral_norm(nn.Conv2d(ndf*8, 1, kernel_size=(3, 6), stride=1, padding=0, bias=False)),
            nn.Flatten(),
        )

    def forward(self, x, debug=False):
        for i, layer in enumerate(self.main):
            x = layer(x)
            if debug:
                print(f"Layer {i} ({type(layer).__name__}): {x.shape}")
        return x