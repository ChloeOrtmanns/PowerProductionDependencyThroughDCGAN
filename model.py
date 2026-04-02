# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

print(f"Using Pytorch {torch.__version__}.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")

# ---RANDOM-FUNCTIONS----------------------------------------------------------------------------------------------------------
def createListOfPaths(folderPath):
    folder = Path(folderPath)
    filePaths = [p for p in folder.iterdir() if p.is_file()]
    return filePaths

# ---DATASET+LOADER------------------------------------------------------------------------------------------------------------
# ---NORMALIZATION-------------------------------------------------------------------------------------------------------------
# neural networks work best when inputs are roughly: mean ~ 0, std ~1. That's why it's important to normalize. If you wouldn't,
# the model could prioritize larger-scale features = unstable/slow training = poorly behaving gradients. 
# IMPORTANT to note is you should compute the stats on training data only and reuse same values for validation/test

class EnergyDataset(Dataset):
    def __init__(self, solar_paths, wind_paths):                      # Compute constants
        self.ds_solar = xr.open_mfdataset(solar_paths, combine="by_coords")
        self.ds_wind  = xr.open_mfdataset(wind_paths , combine="by_coords")

        # Align datasets
        self.ds_solar, self.ds_wind = xr.align(self.ds_solar, self.ds_wind)

        self.solar = self.ds_solar["Solar Energy Potential"]   
        self.wind = self.ds_wind["Wind Energy Potential"]

        # Normalization init
        self.solar_mean = self.solar.mean().compute().item()
        self.solar_std = self.solar.std().compute().item()

        self.wind_mean = self.wind.mean().compute().item()
        self.wind_std = self.wind.std().compute().item()

    def __len__(self):
        return self.solar.sizes["time"]

    def __getitem__(self, idx):                                     # Apply transformation per sample
        solar_sample = self.solar.isel(time=idx).values
        wind_sample = self.wind.isel(time=idx).values

        # Convert to tensors
        solar_tensor = torch.tensor(solar_sample, dtype=torch.float32)
        wind_tensor = torch.tensor(wind_sample, dtype=torch.float32)

        # Normalization
        solar_tensor = (solar_tensor - self.solar_mean) / self.solar_std
        wind_tensor = (wind_tensor - self.wind_mean) / self.wind_std

        # Concatenate along channel dimension
        x = torch.stack([solar_tensor, wind_tensor], dim=0)

        return x

dataset = EnergyDataset(createListOfPaths('Data/Power/Solar')[0:2], createListOfPaths('Data/Power/Wind')[0:2])

loader = DataLoader(dataset, batch_size=32, shuffle=True)   # -> that means it'll group them by 32 in one batch, so 11x32+13=365 

for batch in loader:
    print(batch.shape)

# -----------------------------------------------------------------------------------------------------------------------------


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

