# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# 1. Load data
# 2. Split into train/test
# 3. Compute stats on train ONLY
# 4. Pass stats into dataset

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

statsPath = Path("Data/Processed/stats.pt")
splitsPath = Path("Data/Processed/splits.pt")


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
    
# ---FUNCTIONS-----------------------------------------------------------------------------------------------------------------    
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
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)   # -> that means it'll group them by 32 in one batch, so 11x32+13=365 



if __name__ == "__main__":
    main()


# latent_dim = 10  # latent dimension
# num_epochs = 3  # number of training epochs. You may increase it to gain better results but it will take more time.
# batch_size = 512  # batch size (you may increase it to gain time, but check not to exceed your GPU memory limit)

# train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.main = nn.Sequential(
#             # Convolution 1
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=True),
#             nn.LeakyReLU(),
#             nn.Dropout2d(0.3),

#             # Convolution 2
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
#             nn.LeakyReLU(),
#             nn.Dropout2d(0.3),

#             # Flatten and Linear layer
#             nn.Flatten(),
#             nn.Linear(128 * 7 * 7, 1, bias=True),

#             # Output layer
#             nn.Sigmoid()
#         )

#     def forward(self, input_tensor):
#         return self.main(input_tensor)


# # Example:
# discriminator = Discriminator()
# x = discriminator(torch.randn(batch_size, 1, 28, 28))
# print(x.shape)  # torch.Size([batch_size, 1])

