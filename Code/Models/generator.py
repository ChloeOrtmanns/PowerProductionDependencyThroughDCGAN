import torch.nn.functional as F
import torch.nn as nn

# The generator is designed to map the latent space vector to data-space.
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, lat, lon):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.lat = lat
        self.lon = lon
        self.main = nn.Sequential(
            #---------------------------------CODE WEBSITE----------------------------------------------
            # input z: (nz, 1, 1)

            nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False), # 24x24
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False), # 120x120
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # final layer → 
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_tensor, debug=False):
        if debug:
            print(f"Input: {input_tensor.shape}")

        for i, layer in enumerate(self.main):
            input_tensor = layer(input_tensor)
            if debug:
                print(f"Layer {i} ({type(layer).__name__}): {input_tensor.shape}")
        
        # input_tensor = F.interpolate(input_tensor, size=(lat,lon), mode='bilinear', align_corners=False)
        input_tensor = F.interpolate(input_tensor, size=(self.lat, self.lon), mode='bilinear')
        # input_tensor = input_tensor[:, :, :lat, :lon]
        if debug:
            print(f"After interpolation: {input_tensor.shape}")

        return input_tensor