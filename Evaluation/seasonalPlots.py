"""
Makes an averaged map of all seasons throughout the years
"""

import os
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

pathsPath = Path("Resources/paths.pt")

def pathsLists():
    paths = torch.load(pathsPath, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

solarPaths, windPaths = pathsLists()

seasonal_per_year = []

for path in solarPaths:
    ds = xr.open_dataset(path)
    solar = ds["Solar Energy Potential"]
    
    # Compute seasonal means for this year
    seasonal = solar.groupby("time.season").mean(dim="time")
    
    # Add year info (important!)
    year = solar["time.year"][0].item()
    seasonal = seasonal.expand_dims(year=[year])
    
    seasonal_per_year.append(seasonal)

# Combine all years
result = xr.concat(seasonal_per_year, dim="year")

# Average over years AND space
result_mean = result.mean(dim="year")

solar_vmin, solar_vmax = result_mean.min(), result_mean.max()

for season in result_mean.season.values:
    result_mean.sel(season=season).plot(
        vmin=solar_vmin,
        vmax=solar_vmax
    )
    plt.title(f"Solar - {season}")
    plt.savefig(f"{season}_solar.png", dpi=300)
    plt.close()

#--------------------------------------------

seasonal_per_year = []

for path in windPaths:
    ds = xr.open_dataset(path)
    wind = ds["Wind Energy Potential"]
    
    # Compute seasonal means for this year
    seasonal = wind.groupby("time.season").mean(dim="time")
    
    # Add year info (important!)
    year = wind["time.year"][0].item()
    seasonal = seasonal.expand_dims(year=[year])
    
    seasonal_per_year.append(seasonal)

# Combine all years
result = xr.concat(seasonal_per_year, dim="year")

# Average over years AND space
result_mean = result.mean(dim="year")

wind_vmin, wind_vmax = result_mean.min(), result_mean.max()

for season in result_mean.season.values:
    result_mean.sel(season=season).plot(
        vmin=wind_vmin,
        vmax=wind_vmax
    )
    plt.title(f"Wind - {season}")
    plt.savefig(f"{season}_wind.png", dpi=300)
    plt.close()