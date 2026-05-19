import os
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# import geopandas as gpd
# from shapely.geometry import box

def plot(datasetValue, day):
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(8, 6))
    datasetValue[day].plot(ax=ax, cmap='RdBu_r')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Load data
    gdf = gpd.read_file("Data/geo/export.geojson")
    gdf["geometry"] = gdf["geometry"].buffer(0)
    # Define bounding box
    bbox = box(2, 49, 7, 52)
    # Clip
    clipped = gdf.clip(bbox)
    clipped.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.show()

def plotTwo(data1, data2, day):
    # Create figure + axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Load data
    gdf = gpd.read_file("Data/geo/export.geojson")
    gdf["geometry"] = gdf["geometry"].buffer(0)
    # Define bounding box
    bbox = box(2, 49, 7, 52)
    # Clip
    clipped = gdf.clip(bbox)

    # data1
    data1[day].plot(ax=ax[0], cmap='RdBu_r')
    clipped.boundary.plot(ax=ax[0], color='black', linewidth=1)

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # data2
    data2[day].plot(ax=ax[1], cmap='RdBu_r')
    clipped.boundary.plot(ax=ax[1], color='black', linewidth=1)

    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.show()


def getData(filepath, var):
    # filepath = string, var = string
    dataSet = xr.open_dataset(filepath)
    return dataSet[var]
    # return dataSet[var].isel(time=day-1)

def processUasVasToWindspeed(year):
    
    uasOneDay = getData(f"../../../../mnt/mvbc-gan/Data/2006-2025/uas/uas_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc",
                              'uas')
    vasOneDay = getData(f"../../../../mnt/mvbc-gan/Data/2006-2025/vas/vas_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc",
                              'vas')
    return np.sqrt(uasOneDay**2+vasOneDay**2)

def windEnergyConversion(windspeed, vci, vr, vco, hhub, href, alpha):
    windspeed = windspeed.where(windspeed >= vci, 0)
    windspeed = windspeed.where(~((windspeed >= vci) & (windspeed <= vr)), ((windspeed * (hhub/href)**alpha)**3-vci**3)/(vr**3-vci**3))
    windspeed = windspeed.where(~((windspeed >= vr) & (windspeed < vco)), 1)
    windspeed = windspeed.where(vco > windspeed, 0)
    return windspeed

def windEnergyProduction(year):
    
    ### variables for wind production energy onshore!!!
    vci = 3.5
    vr = 13
    vco = 25
    hhub = 80
    href = 10
    alpha = 0.143

    windspeed = processUasVasToWindspeed(year)

    ### convert windspeed to energy power
    ws = windspeed
    windEnergy = windEnergyConversion(ws, vci, vr, vco, hhub, href, alpha)
    windEnergy.attrs.pop("units", None)
    windEnergy.attrs["long_name"] = "Wind Energy Potential"
    windEnergy.name = "Wind Energy Potential"

    ### saving files
    outputPath = Path("/mnt/mvbc-gan/Data/Power/Wind") / f"wind_{year}.nc"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    windEnergy.to_netcdf(outputPath)

    ### visual verification
    # plot(windEnergy, 31)

def solarEnergyProduction(year):
    ### variables
    toCelsius = -273.15
    G = getData(f"../../../../mnt/mvbc-gan/Data/2006-2025/rsds/rsds_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc", "rsds")
    Gstc = 1000                                 #[W/m²]
    gamma = -0.005

    V = processUasVasToWindspeed(year)

    ### Deze zijn in K!!!!
    Tmean = getData(f"../../../../mnt/mvbc-gan/Data/2006-2025/tas/tas_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc", "tas")
    Tmax = getData(f"../../../../mnt/mvbc-gan/Data/2006-2025/tasmax/tasmax_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc", "tasmax")

    Tref = 25                                   #[°C]
    Taday = (Tmean+toCelsius + Tmax+toCelsius) / 2                  #[°C]

    c1 = 4.3                                    #[°C]
    c2 = 0.943
    c3 = 0.028                                  #[°C m²/W]
    c4 = -1.528                                 #[°C s/m]

    Tcell = c1 + c2 * Taday + c3*G + c4 * V     #[°C]
    Pr = 1 + gamma *(Tcell-Tref)

    solarEnergy = Pr * G / Gstc

    solarEnergy.attrs.pop("units", None)
    solarEnergy.attrs["long_name"] = "Solar Energy Potential"
    solarEnergy.name = "Solar Energy Potential"
    # plot(solarEnergy, 3)

    # ### saving files
    outputPath = Path("/mnt/mvbc-gan/Data/Power/Solar") / f"solar_{year}.nc"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    solarEnergy.to_netcdf(outputPath) 

def analyseVariabilityBoxplot():
    years = range(2006, 2026)
    solar_all, wind_all = [], []

    for year in years:
        solar_all.append(getData(
            f"Data/2006-2025/rsds/rsds_be-04_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_RMIB-UGent-ALARO-0_v1_day_{year}0101-{year}1231.nc",
            "rsds"
        ))
        wind_all.append(processUasVasToWindspeed(year))

    solar_cat = xr.concat(solar_all, dim="time")
    wind_cat  = xr.concat(wind_all,  dim="time")

    # Flatten lat/lon per day → shape (n_days, 121*201)
    solar_flat = solar_cat.values.reshape(len(solar_cat.time), -1)
    wind_flat  = wind_cat.values.reshape(len(wind_cat.time), -1)

    wind_vals = wind_flat.ravel()

    below_ci  = (wind_vals < 3.5)
    partial   = (wind_vals >= 3.5) & (wind_vals <= 6)
    partial2   = (wind_vals >= 6) & (wind_vals <= 13)
    rated     = (wind_vals > 13)  & (wind_vals < 25)
    cutout    = (wind_vals >= 25)

    print(f"Below cut-in  (<3.5):    {below_ci.sum():,}")
    print(f"Partial power (3.5–6):  {partial.sum():,}")
    print(f"Partial power (6–13):  {partial2.sum():,}")
    print(f"Rated power   (13–25):   {rated.sum():,}")
    print(f"Cut-out       (>=25):    {cutout.sum():,}")

    wind_filtered = wind_flat.ravel()
    wind_filtered = wind_filtered[wind_filtered >= 3.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(solar_flat.ravel(), bins=100, color="orange", edgecolor="none")
    axes[0].set_title("Solar (rsds) — distribution of all values")
    axes[0].set_xlabel("W/m²")
    axes[0].set_ylabel("Count")

    axes[1].hist(wind_filtered.ravel(), bins=100, color="steelblue", edgecolor="none")
    axes[1].set_title("Windspeed — distribution of all values")
    axes[1].set_xlabel("m/s")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()   

def main():
    for year in range(2006,2026):
        solarEnergyProduction(year)
        windEnergyProduction(year)

    # plot(xr.open_dataset("Data/Power/Solar/solar_2003.nc")["Solar Energy Potential"], 3)
    # plot(xr.open_dataset("Data/Power/Wind/wind_2003.nc")["Wind Energy Potential"], 3)

    # plotTwo(xr.open_dataset("Data/Power/Solar/solar_2003.nc")["Solar Energy Potential"], xr.open_dataset("Data/Power/Wind/wind_2003.nc")["Wind Energy Potential"], 2)    

if __name__ == "__main__":
    main()