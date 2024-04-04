#! /home/lcesarini/miniconda3/envs/colorbar/bin/python

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
                    required=True,default='JJA',
                    choices=['SON','DJF','MAM','JJA'], 
                    help='season to analyse')
parser.add_argument('-m','--metrics', 
                    required=True,default='q',
                    choices=['f','i','q'],
                    help='metrics to analyse')

args = parser.parse_args()

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
ENV_VAR='mw'
eth__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ETH/CPM/{ENV_VAR}/*" ).load()
# mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/{ENV_VAR}/*").load()
ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/{ENV_VAR}/*").load()
hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/{ENV_VAR}/*").load()
cnrm_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CNRM/CPM/{ENV_VAR}/*" ).load()
knmi_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KNMI/CPM/{ENV_VAR}/*" ).load()
kit__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KIT/CPM/{ENV_VAR}/*" ).load()
cmcc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CMCC/CPM/{ENV_VAR}/*" ).load()

sphe_rg = xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*" ).load()

name_models=['ETH','ICTP','HCLIMcom','CNRM','KNMI','KIT','CMCC','SPHERA']
ds_models=[eth__rg,ictp_rg,hcli_rg,cnrm_rg,knmi_rg,kit__rg,cmcc_rg,sphe_rg]

dict_q99_hourly={}
for NAME,DS in tqdm(zip(name_models,ds_models),total=len(name_models)):
    for S in ["DJF"]:
        dict_0={f"{NAME}_{S}":compute_quantiles_by_hour_wind(DS,0.999,S)}
        dict_q99_hourly.update(dict_0)

ax=plt.axes()
[dict_q99_hourly[list(dict_q99_hourly.keys())[j]].\
 plot(label=name_models[i],ax=ax,alpha=0.5,linestyle='-.') for i,j in enumerate(range(5))]
xr.concat([dict_q99_hourly[list(dict_q99_hourly.keys())[j]] for j in range(5)],'model').\
    mean(dim='model').plot(label="Ensemble",ax=ax, linewidth=4,color='red')
dict_q99_hourly[f"SPHERA_{S}"].plot(label="SPHERA",ax=ax,linestyle='-', linewidth=3,color='blue')
# dict_q99_hourly[f"VHR_CMCC_{S}"].plot(label="VHR CMCC",ax=ax,linestyle='-', linewidth=3,color='magenta')
# dict_q99_hourly[f"STATIONS_{S}"].plot(label="STATIONS",ax=ax,marker='*', linewidth=3,color='green')
ax.set_title("Heavy Winds (99.9th percentile) by hour JJA")
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Winds (m/s)")
plt.legend()
plt.savefig(f"figures/q999_hourly_winds_{S}.png",dpi=300,bbox_inches="tight")
plt.close()
