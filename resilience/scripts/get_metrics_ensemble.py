#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
"""
Compute the metrics for the ensemble

"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import rasterio
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
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature

os.chdir("/home/lcesarini/2022_resilience/")

from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-s','--season', 
                    required=False,default='JJA',
                    choices=['SON','DJF','MAM','JJA'], 
                    help='season to analyse')
parser.add_argument('-m','--metrics', 
                    required=False,default='q',
                    choices=['f','i','q'],
                    help='metrics to analyse')

args = parser.parse_args()

PATH_OUTPUT="/home/lcesarini/2022_resilience/output"
 
lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI','CMCC','KIT']

for SEAS in tqdm(['SON','DJF','MAM','JJA'],total=4):
    list_f = [xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{nm}_f.nc") for nm in name_models]
    list_i = [xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{nm}_i.nc") for nm in name_models]
    list_q = [xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{nm}_q.nc") for nm in name_models]
    
    ens_mean_f=xr.concat(list_f,'model').mean(dim='model')
    ens_mean_i=xr.concat(list_i,'model').mean(dim='model')
    ens_mean_q=xr.concat(list_q,'model').mean(dim='model')
    
    ens_spre_f=xr.concat(list_f,'model').std(dim='model')
    ens_spre_i=xr.concat(list_i,'model').std(dim='model')
    ens_spre_q=xr.concat(list_q,'model').std(dim='model')

    ens_mean_f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_f.nc")
    # ens_spre_f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_f.nc")

    ens_mean_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_i.nc")
    # ens_spre_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_i.nc")

    ens_mean_q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_q.nc")
    # ens_spre_q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_q.nc")
    
    plot_panel_rotated(
    figsize=(12,6),
    nrow=1,ncol=2,
    list_to_plot=[ens_mean_q.pr,ens_spre_q.pr],
    name_fig=f"PANEL_ENSEMBLE_MEAN_SPREAD_{SEAS}",
    list_titles=["Mean","Spread"],
    levels=[lvl_q,np.arange(0,6,0.8)],
    suptitle=f"Ensemble mean and spread for the 8 CPM mdoels over 2000-2009 for {SEAS}\nHeavu Precipitation",
    # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
    name_metric=["[mm/hr]","[mm/hr]"],
    SET_EXTENT=False,
    cmap=[cmap_q,cmap_q]
    )