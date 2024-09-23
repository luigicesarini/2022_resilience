import os
import sys
import argparse
import subprocess
import numpy as np 
import xarray as xr 
# import xarray.ufuncs as xu
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature


#create parser object
parser = argparse.ArgumentParser(description='Remap SPHERA data')
#add argument
parser.add_argument('--year', type=int, help='Year to remap.',default=2000)
parser.add_argument('--long_name', type=str, help='Long name of the variable.',default="pr")
parser.add_argument('--short_name', type=str, help='Short name of the variable.', default="tp")
#parse the arguments
args = parser.parse_args()

NAME_VAR=args.long_name
SHORT_VAR=args.short_name
YEAR=args.year
# NAME_VAR='tp'
# SHORT_VAR='pr'
# YEAR=1995
#PATH TO THE GRID YOU WANT TO REMAP THE DATA IN THIS CASE I?LL USE AN EXAMPLE
PATH_GRID="/mnt/beegfs/lcesarini/SPHERA/newcommongrid.txt"
PATH_DATA="/mnt/beegfs/lcesarini/SPHERA/original"

patterns_original = [
    f'{PATH_DATA}/{SHORT_VAR}/*{YEAR}**zoom*', 
            ]

# Use glob to find files matching any of the specified patterns
matching_files = []
for pattern in patterns_original:
    matching_files.extend(glob(pattern))
print(matching_files)

for file_grib in tqdm(matching_files):
    subprocess.run(f"cdo remapycon,{PATH_GRID} {file_grib} {file_grib.replace('zoom','remapped')}",shell=True)
    subprocess.run(f"grib_to_netcdf -o {file_grib.replace('zoom','remapped').replace('grb2','nc')} {file_grib.replace('zoom','remapped')}",shell=True)




# xr.open_dataset("/mnt/data/commonData/SPHERA/test/200012_ten3_Tdeep_2t_remapped.nc").isel(time=1).t2m.plot()
# plt.show()

