#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                            r2_score,mean_absolute_percentage_error)

from utils import *

import warnings
warnings.filterwarnings('ignore')

os.chdir("2022_resilience/")

gripho = f"/mnt/data/commonData/OBSERVATIONS/ITALY/gripho-v1_1h_TSmin30pct_2001-2016.nc"
gripho = f"/home/lcesarini/2022_resilience/data/regrid/gripho-v1_1h_TSmin30pct_2001-2016_remap.nc"

ds = xr.open_dataset(gripho).load()

if hasattr(ds,"crs"):
    if any(hasattr(ds,name_attr) for name_attr in ["laea","lambert_azimuthal_equal_area"]):
        
        rot = ccrs.LambertAzimuthalEqualArea(central_longitude = ds['crs'].latitude_of_projection_origin, 
                                            central_latitude   = ds['crs'].longitude_of_central_meridian, 
                                            false_easting      = ds['crs'].false_easting,
                                            false_northing     = ds['crs'].false_northing, 
                                            globe=None)

    else:
        rot = ccrs.PlateCarree()

fig,ax = plt.subplots(nrows=1,
                      ncols=2,
                      figsize=(12,3),constrained_layout=True, squeeze=True,
                      subplot_kw={"projection":ccrs.PlateCarree()})

ax=ax.flatten()
# Ref to palettes https://matplotlib.org/stable/tutorials/colors/colormaps.html
cmap = plt.cm.hsv

for i,metric in enumerate([ds]):

    slice = metric.isel(time=180).pr
    bounds =  np.arange(1,20,4)
    print(bounds)

    # slice=xr.where(np.isnan(slice),0,slice)
    norm = mpl.colors.BoundaryNorm(bounds, bounds.shape[0]+1, extend='both')
    pcm=slice.plot.pcolormesh(ax=ax[i],alpha=0.75,transform=rot,
                              cmap=cmap, norm=norm, add_colorbar=True, 
                              cbar_kwargs={"shrink":0.8,
                                           "orientation":"horizontal",
                                           "alpha":0.15}
                            )
    ax[i].coastlines()
    # gl = ax[i].gridlines(
    #     draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--'
    # )
    ax[i].add_feature(cfeature.BORDERS, linestyle='--')
    # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
    # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
    # ax.add_feature(cfeature.STATES)
    ax[i].set_title("Bitch")
plt.savefig("figures/slice_gripho.png")
plt.close()

