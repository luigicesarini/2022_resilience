#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
"""
0. Copy the original file adding the extension grb2
1. Remap the original
2. Decumulate the precipitation 
3. Save to the common folder

"""
import os
import sys 
sys.path.append("/home/lcesarini/2022_resilience")
import argparse
import subprocess
# import rioxarray
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

from resilience.utils import *

PATH_GRID="/home/lcesarini/2022_resilience/scripts/newcommongrid.txt"
PATH_SPHERA="/mnt/data/RESTRICTED/SPHERA"
PATH_SPHERA_OUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr"
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"

os.chdir("/home/lcesarini/2022_resilience")


gripho=xr.open_dataset("/mnt/data/commonData/OBSERVATIONS/ITALY/gripho-v1_1h_TSmin30pct_2001-2016_cut3km.nc",
                       chunks={"time":8760})

year=2004
gripho_slice = gripho.sel(time=gripho["time.year"].isin(year))
gripho_slice.to_netcdf(f"/mnt/data/lcesarini/gripho_{year}.nc")


from utils import *
from cfgrib.xarray_to_grib import to_grib

list_sphera=\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2000*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2001*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2002*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2003*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2004*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2005*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2006*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2007*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2008*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2009*')

# subprocess.run("ls /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{2000,2001}*",shell=True)


ds_sphera=xr.open_mfdataset(list_sphera)
cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()


sphera_in_cnrm=[t_sphera in cnrm_rg.time_bnds[:,1] for t_sphera in ds_sphera.time.values]
cnrm_in_sphera=[t_cnrm in ds_sphera.time for t_cnrm in cnrm_rg.time_bnds[:,1].values]

print(cnrm_rg.time.values[np.argwhere(np.array(cnrm_in_sphera)==0)])


ds_f=xr.open_dataset(f"output/{SEAS}/{nm}_f_{np.int8(WH)}.nc")
ds_i=xr.open_dataset(f"output/{SEAS}/{nm}_i_{np.int8(WH)}.nc")
ds_q=xr.open_dataset(f"output/{SEAS}/{nm}_q_{np.int8(WH)}.nc")
if nm == "STATIONS":
    ds_q=ds_q.isel(quantile=0)

plot_panel_rotated(
    nrow=1,ncol=3,
    list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
    name_fig=f"PANEL_{nm}_{SEAS}_WH",
    list_titles=["Frequency","Intensity","Heavy Prec."],
    levels=[lvl_f,lvl_i,lvl_q],
    suptitle=f"Ensemble's metrics for {SEAS}",
    name_metric=["[fraction]","[mm/h]","[mm/h]"],
    SET_EXTENT=False,
    cmap=[cmap_f,cmap_i,cmap_q]
)




lon,lat=bias_cpm_sta.lon,bias