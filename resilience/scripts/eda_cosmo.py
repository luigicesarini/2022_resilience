#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import rioxarray
import subprocess
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
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

from resilience.utils import *

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

os.chdir("/home/lcesarini/2022_resilience/")

gripho=xr.open_dataset("/mnt/data/lcesarini/GRIPHO/gripho_2003.nc")

proj=ccrs.LambertAzimuthalEqualArea(central_latitude=52,
                                    central_longitude=10,
                                    false_easting=4321000,
                                    false_northing=3210000)

tran=ccrs.PlateCarree()
x=gripho.pr.quantile(dim='time',q=0.999)

ax=plt.axes(projection=tran)
x.plot(ax=ax,transform=proj,cmap=cmap_q,levels=np.arange(2,18,2))
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
plt.savefig(f"output/GRIPHO_2003.png",dpi=300)  
plt.close()

mask=xr.open_dataset("data/mask_stations_nan_common.nc")
sta_val=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 
PATH_COSMO=f"/mnt/data/lcesarini/COSMO/REA_2"

EV= "TOT_PRECIP"

ll_sphera=glob("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*")


ll_ds=[xr.open_dataset(pth) for pth in ll_sphera[34:]]

ds_concat=xr.concat(ll_ds,dim='time')


hp_sph_all_djf=get_season(ds_concat.sel(time=ds_concat.pr["time.year"].isin(np.arange(2000,2010))),"DJF")
hp_sph_all_mam=get_season(ds_concat,"MAM")
hp_sph_all_jja=get_season(ds_concat,"JJA")
hp_sph_all_son=get_season(ds_concat,"SON")

sph_all_djf=hp_sph_all_djf.quantile(dim='time',q=0.999)
sph_all_mam=hp_sph_all_mam.quantile(dim='time',q=0.999)
sph_all_jja=hp_sph_all_jja.quantile(dim='time',q=0.999)
sph_all_son=hp_sph_all_son.quantile(dim='time',q=0.999)

gripho_djf=xr.open_dataset("/home/lcesarini/2022_resilience/output/DJF/GRIPHO_ORIGINAL_q.nc")
sph_old_djf=xr.open_dataset("/home/lcesarini/2022_resilience/output/DJF/SPHERA_q.nc")
station_djf=xr.open_dataset("/home/lcesarini/2022_resilience/output/DJF/STATIONS_q.nc")

SEASON = ['DJF','MAM','JJA','SON']
plot_panel_rotated(
    figsize=(14,10),ncol=3,nrow=1,
    list_to_plot=[sph_all_djf.pr,
                  sph_djf.pr,station_djf.pr.isel(quantile=0)],
    name_fig="",
    list_titles=[f'Heavy Prec for {seas}' for seas in SEASON],
    levels=[lvl_q for _ in range(3)],
    suptitle="SPHERA 1996-2020",
    name_metric=["Heavy Prec" for _ in range(3)],
    SET_EXTENT=0,
    cmap=[cmap_q for _ in range(3)],
    proj=ccrs.PlateCarree(),
    transform=ccrs.PlateCarree(),
    SAVE=False
)

sph_djf=get_triveneto(sph_old_djf,sta_val)


sphera=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*1995*")
# stringa=f"{PATH_COSMO}/{EV}/bz2/*bz2*"

# list_files=glob(stringa)

# for file in tqdm(list_files,total=len(list_files)):
#     subprocess.run(f"bzip2 -dk {file}",shell=True)

ds=xr.open_dataset(glob(f"{PATH_COSMO}/{EV}/*.nc")[0])
ds=xr.open_dataset("/mnt/data/lcesarini/COSMO/REA_2/TOT_PRECIP/TOT_PRECIP.2D.201806.grb")


ds.to_netcdf("/mnt/data/lcesarini/COSMO/REA_2/TOT_PRECIP/TOT_PRECIP.2D.201806.nc")


ds.tp.quantile(dim='time',q=0.99).plot()
plt.show()

ds_repro=xr.open_dataset("/mnt/data/lcesarini/tmp/TOT_PRECIP.2D.201806_repro.grb",engine='pynio')

ds_repro['TOT_PREC_GDS0_SFC'.ini]


ds.quantile(q=0.99,dim='time').pr.plot()

ds.attrs.keys()

ds.attrs['Conventions']

ds
ds2=xr.open_dataset("/mnt/data/lcesarini/tmp/prova_cosmo.grb",engine='pynio')
ds2['TOT_PREC_GDS0_SFC']

ax=plt.axes(projection=ccrs.PlateCarree())
(ds2.pr.quantile(q=0.999,dim='time') * 3600).\
    plot(ax=ax,
         cmap='RdYlGn',
         levels=np.arange(2,18,2))
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
plt.show()



ds1=xr.open_dataset(f"{PATH_COSMO}/TOT_PRECIP/TOT_PRECIP.2D.201809.grb",engine="cfgrib")
ds2=xr.open_dataset(f"{PATH_COSMO}/TOT_PRECIP/TOT_PRECIP.2D.201809.grb",engine="pynio")


ds1.drop(["step","surface","valid_time"]).to_netcdf("/mnt/ssd/lcesarini/TOT_PRECIP.2D.201809.nc")

q_99=ds1.tp.quantile(dim='time',q=0.99)
q_99.plot()
plt.show()

# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/COSMO/REA_2/TOT_PRECIP/TOT_PRECIP.2D.201809.grb /mnt/ssd/lcesarini/TOT_PRECIP.2D.201809_reamp.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/ssd/lcesarini/TOT_PRECIP.2D.201809.nc /mnt/ssd/lcesarini/TOT_PRECIP.2D.201809_reamp.nc

# cdo griddes /mnt/ssd/lcesarini/TOT_PRECIP.2D.201809.nc > target_grid_description.txt




# xx=xr.open_dataset("/mnt/ssd/lcesarini/TOT_PRECIP.2D.201809_reamp.nc",engine='cfgrib')