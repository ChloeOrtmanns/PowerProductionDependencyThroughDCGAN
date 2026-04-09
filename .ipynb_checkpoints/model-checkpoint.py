# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# 1. Load data
# 2. Split into train/test
# 3. Compute stats on train ONLY
# 4. Pass stats into dataset

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

# ---VARIABLES-----------------------------------------------------------------------------------------------------------------
statsPath = Path("Data/Processed/stats.pt")
splitsPath = Path("Data/Processed/splits.pt")

lat = 121
lon = 201

latentDim = 10      # latent dimension
numEpochs = 3       # number of training epochs. You may increase it to gain better results but it will take more time.
batchSize = 32      # -> that means it'll group them by 32 in one batch, so 11x32+13=365 for one year
ngpu = 0
nz = 100            # Size of z latent vector (underlying degrees of freedom)
# |---------When nz is too small => outputs look too similar, when too big => may learn noise instead of structure
ngf = 64            # 32 is trauning is unstable, 128 for more detail
# |---------ngf ↑ → more capacity → better detail → harder training
# |---------ngf ↓ → simpler model → more stable → less expressive
nc = 2
ndf = 64


print(f"Using Pytorch {torch.__version__}.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")

# ---DATASET+LOADER------------------------------------------------------------------------------------------------------------
# ---NORMALIZATION-------------------------------------------------------------------------------------------------------------
# neural networks work best when inputs are roughly: mean ~ 0, std ~1. That's why it's important to normalize. If you wouldn't,
# the model could prioritize larger-scale features = unstable/slow training = poorly behaving gradients. 
# IMPORTANT to note is you should compute the stats on training data only and reuse same values for validation/test
class EnergyDataset(Dataset):
    def __init__(self, solarPaths, windPaths, stats=None):                      # Compute constants
        self.dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
        self.dsWind  = xr.open_mfdataset(windPaths , combine="by_coords")

        # Align datasets
        self.dsSolar, self.dsWind = xr.align(self.dsSolar, self.dsWind)

        self.solar = self.dsSolar["Solar Energy Potential"]   
        self.wind = self.dsWind["Wind Energy Potential"]

        self.stats = stats

    def __len__(self):
        return self.solar.sizes["time"]

    def __getitem__(self, idx):                                     # Apply transformation per sample
        solarSample = self.solar.isel(time=idx).values
        windSample = self.wind.isel(time=idx).values

        # Convert to tensors
        solarTensor = torch.tensor(solarSample, dtype=torch.float32)
        windTensor = torch.tensor(windSample, dtype=torch.float32)

        if self.stats is not None:
            solarTensor = (solarTensor - self.stats["solarMean"]) / self.stats["solarStd"]
            windTensor  = (windTensor  - self.stats["windMean"])  / self.stats["windStd"]

        # Concatenate along channel dimension
        x = torch.stack([solarTensor, windTensor], dim=0)

        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu):
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

# The generator is designed to map the latent space vector to data-space.
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
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
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(self, input_tensor, debug=False):
        if debug:
            print(f"Input: {input_tensor.shape}")

        for i, layer in enumerate(self.main):
            input_tensor = layer(input_tensor)
            if debug:
                print(f"Layer {i} ({type(layer).__name__}): {input_tensor.shape}")
        
        input_tensor = F.interpolate(input_tensor, size=(lat,lon), mode='bilinear', align_corners=False)
        if debug:
            print(f"After interpolation: {input_tensor.shape}")

        return input_tensor

# ---FUNCTIONS-----------------------------------------------------------------------------------------------------------------    
def weights_init(m):                                
    # custom weights initialization called on ``netG`` and ``netD``
    # the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
    # The weights_init function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose,
    # and batch normalization layers to meet this criteria. This function is applied to the models immediately after initialization.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def savePathsLists():
    solarPaths = list(Path("Data/Power/Solar").glob("*.nc"))
    windPaths  = list(Path("Data/Power/Wind").glob("*.nc"))

    torch.save({
    "solarPaths": solarPaths,
    "windPaths": windPaths
}, "paths.pt")
    
def pathsLists():
    paths = torch.load("paths.pt", weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

def calculateSaveStatsSplits():
    solarPaths, windPaths = pathsLists()
    fullDataset = EnergyDataset(
        solarPaths,
        windPaths,
        stats=None
    )
    trainSize = int(0.8 * len(fullDataset))
    trainIndices = list(range(0, trainSize))
    testIndices  = list(range(trainSize, len(fullDataset)))
    torch.save({
        "train": trainIndices,
        "test": testIndices
        }, splitsPath)

    solarMean = fullDataset.solar.isel(time=trainIndices).mean().compute().item()
    solarStd  = fullDataset.solar.isel(time=trainIndices).std().compute().item()
    windMean  = fullDataset.wind.isel(time=trainIndices).mean().compute().item()
    windStd   = fullDataset.wind.isel(time=trainIndices).std().compute().item()
    torch.save({
        "solarMean": solarMean,
        "solarStd": solarStd,
        "windMean": windMean,
        "windStd": windStd
        }, statsPath)


def main():
    # only run this function once
    # savePathsLists()
    # calculateSaveStatsSplits()

    # ---NORMALIZED-DATASET--------------------------------------------------------------------------------------------------------
    solarPaths, windPaths = pathsLists()
    normalizedDs = EnergyDataset(
        solarPaths,
        windPaths,
        torch.load(statsPath)
    )

    # ---DATASET-SPLITTING---------------------------------------------------------------------------------------------------------
    splits = torch.load(splitsPath)
    trainIndices = splits["train"]
    testIndices  = splits["test"]
    
    trainDataset = Subset(normalizedDs, trainIndices)
    testDataset = Subset(normalizedDs, testIndices)

    # ---LOADER--------------------------------------------------------------------------------------------------------------------
    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)    
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    # ---GENERATOR-----------------------------------------------------------------------------------------------------------------
    netG = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # ---Testing---
    # z = torch.randn(1, nz, 1, 1)  # batch_size=1
    # out = netG(z, debug=True)
    # should be torch.Size([1, 2, 121, 201])

    # ---DISCRIMINATOR------------------------------------------------------------------------------------------------------------
    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    # ---Testing---
    x = torch.randn(1, 2, 121, 201).to(device)
    out = netD(x)

    print(out.shape)

    # # ---TRAININGLOOP-------------------------------------------------------------------------------------------------------------
    # for epoch in range(numEpochs):
    #     for i, realData in enumerate(trainLoader):

    #         realData = realData.to(device)   # shape: [B, 2, 121, 201]
    #         bSize = realData.size(0)

    #         # Labels
    #         real_labels = torch.ones(bsize, 1, device=device)
    #         fake_labels = torch.zeros(bSize, 1, device=device)

    #         # train
    #         netD.zero_grad()

    #         # Real data
    #         output_real = netD(real_data).view(-1, 1)
    #         loss_real = criterion(output_real, real_labels)

    #         # Fake data
    #         noise = torch.randn(b_size, nz, 1, 1, device=device)
    #         fake_data = netG(noise)

    #         output_fake = netD(fake_data.detach()).view(-1, 1)
    #         loss_fake = criterion(output_fake, fake_labels)

    #         # Total loss
    #         loss_D = loss_real + loss_fake
    #         loss_D.backward()
    #         optimizerD.step()

    #         # ====================================================
    #         # 2️⃣ Train Generator
    #         # ====================================================
    #         netG.zero_grad()

    #         output_fake = netD(fake_data).view(-1, 1)
    #         loss_G = criterion(output_fake, real_labels)  # trick: wants D to think fake is real

    #         loss_G.backward()
    #         optimizerG.step()

    #         # ====================================================
    #         # Logging
    #         # ====================================================
    #         if i % 50 == 0:
    #             print(f"[{epoch}/{numEpochs}] [{i}/{len(trainLoader)}] "
    #                 f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")


if __name__ == "__main__":
    main()


