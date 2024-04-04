#! /home/lcesarini/miniconda3/envs/colorbar/bin/python

import os
import re
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
# import xarray.ufuncs as xu 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
# from sklearn.metrics import (mean_absolute_error,mean_squared_error,
#                              r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

from utils import *

if __name__=="__main__":

    path_uas = f"/mnt/data/lcesarini/SPHERA/u/"
    path_vas = f"/mnt/data/lcesarini/SPHERA/v/"

    # uas=xr.open_mfdataset(glob(path_uas+"*")[0]) 
    # print(uas.time[0:2].values)

    for year in np.arange(2000,2010):
        

        path_mw = "/mnt/data/lcesarini/SPHERA/mw"
        uas = xr.open_mfdataset(glob(f"{path_uas}*{year}**remapped*"),engine="cfgrib")
        vas = xr.open_mfdataset(glob(f"{path_vas}*{year}**remapped*"),engine="cfgrib")

        name_file = f"mw_{year}_remapped_SPHERA.nc"

        # cropped_uas = crop_to_extent(uas)
        # cropped_vas = crop_to_extent(vas)

        mw = np.sqrt(np.power(uas.u10.values,2) + np.power(vas.v10.values,2))

        mw_d = xr.DataArray(mw, 
                    coords={'time':uas.time,
                            'lon': uas.longitude.values, 
                            'lat': uas.latitude.values
                            },
                    dims={'time':uas.time.shape[0],
                          'lat':uas.latitude.shape[0],
                          'lon':uas.longitude.shape[0]},
                    attrs = uas.attrs
        ) 
                    # dict(description = "mw stands for 'Module Wind",
                    #             unit = '[m*s-1]'))

        mw_ds=mw_d.to_dataset(name = 'mw', promote_attrs = True)        
    
        mw_ds.to_netcdf(f'/mnt/data/lcesarini/SPHERA/mw/{name_file}',
                        encoding = {"mw": {"dtype": "float32"}})
