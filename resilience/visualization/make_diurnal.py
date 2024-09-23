#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")
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

os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")

from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-s','--season',nargs='+', 
                    required=False,default='JJA',
                    # choices=['SON','DJF','MAM','JJA'], 
                    help='seasons to analyse')
parser.add_argument('-m','--metrics', 
                    required=False,default='q',
                    choices=['f','i','q'],
                    help='metrics to analyse')
parser.add_argument('-ev','--env_var', 
                    required=False,default='pr',
                    choices=['pr','mw'],
                    help='environmental variable to analyse')
parser.add_argument('-q','--quantile', 
                    required=False,default=0.999,
                    nargs='+',type=float,
                    help='Quantile to compute')
parser.add_argument('-nm','--name_model', 
                    required=False,default="ETH",
                    help='Name of the model')
parser.add_argument('-p','--plot',action="store_true",
                    help='Plot the output')

parser.add_argument('-c','--compute',action='store_true',
                    help='Compute the output')

args = parser.parse_args()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT"
PATH_OUT_DIURNAL="/mnt/beegfs/lcesarini/2022_resilience/output/diurnal"

ENV_VAR=args.env_var
PLOT=args.plot
COMPUTE=args.compute
Q=args.quantile
SEASONS=args.season
NAME=args.name_model
print(SEASONS)
print(Q)
print(NAME)
if COMPUTE:
    if ENV_VAR=='pr':
        # name_models=['MOHC', 'ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom','SPHERA']
        # for NAME in name_models:
        if NAME == "ETH":
            ds_rg=fix_eth()
        elif NAME == "SPHERA":
            ll_sphera=[glob(f"/mnt/beegfs/lcesarini/DATA_FPS/reanalysis/{NAME}/{ENV_VAR}/*{year}*") for year in np.arange(2000,2010)]
            ds_rg=xr.open_mfdataset(item for list in ll_sphera for item in list).load()
        else:
            ds_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/{NAME}/CPM/{ENV_VAR}/*").load()

        for S in tqdm(SEASONS,total=4):
        
            compute_quantiles_by_hour(ds_rg,Q,S).to_netcdf(f'/mnt/beegfs/lcesarini/2022_resilience/output/diurnal/{ENV_VAR}/{S}/diurnal_{NAME}_{ENV_VAR}_{"_".join(np.array(Q).astype(str))}.nc')
        

    elif ENV_VAR=='mw':
        # name_models=['ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom','SPHERA']
        # name_models=['SPHERA']
        # for NAME in name_models:
        if NAME == "SPHERA": 
            ds_rg=xr.open_mfdataset(f"/mnt/beegfs/lcesarini/DATA_FPS/reanalysis/{NAME}/{ENV_VAR}/*" ).load()
        else:
            ds_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/{NAME}/CPM/{ENV_VAR}/*").load()
            
        for S in tqdm(SEASONS,total=4):
    
            compute_quantiles_by_hour_wind(ds_rg,Q,S).to_netcdf(f'/mnt/beegfs/lcesarini/2022_resilience/output/diurnal/{ENV_VAR}/{S}/diurnal_{NAME}_{ENV_VAR}_{"_".join(np.array(Q).astype(str))}.nc')


if PLOT:
    if ENV_VAR == 'pr':
        for S in SEASONS:

            name_models=['MOHC', 'ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom','SPHERA']

            ax=plt.axes()

            ll_pr=[xr.load_dataset(f'{PATH_OUT_DIURNAL}/{ENV_VAR}/{S}/diurnal_{NAME}_{ENV_VAR}_{Q}.nc') for NAME in name_models]
            [ll_pr[i].pr.plot(label=name_models[i],ax=ax,alpha=0.5,linestyle='-.') for i in range(len(name_models)-1)]
            xr.concat(ll_pr[:-1],'model').mean(dim='model').pr.plot(label="Ensemble",ax=ax,marker='^', linewidth=4,color='red')
            ll_pr[len(name_models)-1].pr.plot(label="SPHERA",ax=ax,marker='+',linestyle='-', linewidth=3,color='blue')
            ax.set_title(f"Heavy Precipitation ({'9.9' if Q == 0.999 else '99'}th percentile) by hour {S}")
            ax.set_xlabel("Hour of the day")
            ax.set_ylabel("Precipitation (mm/hr)")
            plt.legend()
            # plt.show()
            # plt.close()
            plt.savefig(f"figures/q{'999' if Q == 0.999 else '99'}_hourly_{ENV_VAR}_{S}.png",dpi=300,bbox_inches="tight")
            plt.close()

    elif ENV_VAR=='mw':
        for S in SEASONS:

            name_models=['ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom','SPHERA']

            ax=plt.axes()

            ll_mw=[xr.load_dataset(f'{PATH_OUT_DIURNAL}/{ENV_VAR}/{S}/diurnal_{NAME}_{ENV_VAR}_{Q}.nc') for NAME in name_models]
            [ll_mw[i].mw.plot(label=name_models[i],ax=ax,alpha=0.5,linestyle='-.') for i in range(len(name_models)-1)]
            xr.concat(ll_mw[:-1],'model').mean(dim='model').mw.plot(label="Ensemble",ax=ax,marker='^', linewidth=4,color='red')
            ll_mw[len(name_models)-1].mw.plot(label="SPHERA",ax=ax,marker='+',linestyle='-', linewidth=3,color='blue')
            ax.set_title(f"Heavy Winds ({'99.9' if Q == 0.999 else '99'}th percentile) by hour {S}")
            ax.set_xlabel("Hour of the day")
            ax.set_ylabel("Wind (m/s)")
            plt.legend()
            # plt.show()
            # plt.close()
            plt.savefig(f"figures/q{'999' if Q == 0.999 else '99'}_hourly_{ENV_VAR}_{S}.png",dpi=300,bbox_inches="tight")
            plt.close()
