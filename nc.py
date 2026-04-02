import netCDF4 as nc

# specify the filepath
file_path = 'Data/1975-2005/rsds/rsds_be-04_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_RMIB-UGent-ALARO-0_v1_day_19490101-19491231.nc'

# open the file
ds = nc.Dataset(file_path)

# print the file details
print(ds)


# Close the dataset after using it
ds.close()