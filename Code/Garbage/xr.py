import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import seaborn as sns

#----------------------------------------------Functions
def windspeed(u, v):
    return np.sqrt(u**2 + v**2)

#----------------------------------------------Inits
# specify the filepath
file_path_uas = 'Data/1975-2005/uas/uas_be-04_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_RMIB-UGent-ALARO-0_v1_day_19750101-19751231.nc'
file_path_vas = 'Data/1975-2005/vas/vas_be-04_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_RMIB-UGent-ALARO-0_v1_day_19750101-19751231.nc'

# Open the NetCDF file using xarray
ds_uas = xr.open_dataset(file_path_uas)
ds_vas = xr.open_dataset(file_path_vas)

#-----------------------------------------------Prints
# Print the dataset details
# print(ds)

# Access a specific variable
uas = ds_uas['uas']
vas = ds_vas['vas']
lon = ds_uas['lon']
lat = ds_uas['lat']

# You can also access dimensions and coordinates
# print(ds.dims)
# print(ds.coords)
# print(ds_uas['lon'].values.min())

u = uas.values[0][0][0]
v = vas.values[0][0][0]

# print("The uas value of 01/01/1975, lat=49.0, lon=2.0: ", u, "m/s")
# print("The vas value of 01/01/1975, lat=49.0, lon=2.0: ", v, "m/s")

# print("Combined windspeed would be:")
# print(windspeed(u, v))

print('Number of elements in the array:')
print(np.size(uas.values[0][0]))

# print('Length of one array element in bytes:')
# print(np.itemsize(uas.values[0]))

# print('Number of array dimensions:')
# print(int(uas.values[0]))

print('Tuple of array dimensions:')
print(np.shape(uas.values[0][0][0:5]))


#-----------------------------------------------Plots

# bounding_box = (lon.values.min(), lon.values.max(),
#                 lat.values.min(), lat.values.max())

# print(bounding_box)
# norm = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)

# plt.subplot(1,2,1)
# uas[2].plot(cmap='RdBu_r', norm=norm)

# plt.subplot(1,2,2)
# vas[2].plot(cmap='RdBu_r', norm=norm)
# plt.show()

print(type(uas[2]))

#-----------------------------------------------Close
# Close the dataset (optional, handled automatically with xarray)
ds_uas.close()
ds_vas.close()