#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from glob import glob
from tqdm import tqdm
from datetime import datetime, timedelta
# from utils import *

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

parser.add_argument('-wh','--wethours',action='store_true',
                    help='For wethours?')

args = parser.parse_args()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS/"
PATH_OUT="/mnt/beegfs/lcesarini/2022_resilience/output/metrics"

if not os.path.exists(PATH_OUT): os.makedirs(PATH_OUT)


ENV_VAR=args.env_var
PLOT=args.plot
COMPUTE=args.compute
Q=args.quantile
SEASONS=args.season
NAME=args.name_model
WH=args.wethours
print(SEASONS)
print(Q)
print(NAME)



if NAME=="SPHERA":
    # ds=xr.open_mfdataset(f"{PATH_COMMON_DATA}/reanalysis/{NAME}/{ENV_VAR}/*.nc") #minimum -0.015625
    ds=xr.open_mfdataset(f"/mnt/beegfs/lcesarini/{NAME}/decumulated/new/*.nc").load()
elif NAME=="GRIPHO":
    ds=xr.open_mfdataset(f"{PATH_COMMON_DATA}/gripho_ori_clipped.nc") #minimum -0.015625
    ds=ds.isel(time=ds['time.year'].isin(np.arange(2000,2010)))
elif NAME=="ETH":
    ds21=fix_eth()
else:
    ds2=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()


assert len(SEASONS) < 2, f"TOO many seasons: {len(SEASONS)}"

dict_metrics={}

if not os.path.exists(f"{PATH_OUT}/{SEASONS[0]}"): os.makedirs(f"{PATH_OUT}/{SEASONS[0]}")


if WH:
    dict_0={NAME:compute_metrics(get_season(ds,season=SEASONS[0]),meters=True,quantile=Q,wethours=WH)}
    dict_metrics.update(dict_0)
    dict_metrics[NAME][0].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_f_{np.int8(WH)}.nc")
    dict_metrics[NAME][1].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_i_{np.int8(WH)}.nc")
    dict_metrics[NAME][2].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_v_{np.int8(WH)}.nc")
    dict_metrics[NAME][3].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_q_{np.int8(WH)}.nc")
else:
    dict_0={NAME:compute_metrics(get_season(ds,season=SEASONS[0]),meters=True,quantile=Q,wethours=False)}
    dict_metrics.update(dict_0)
    dict_metrics[NAME][0].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_f.nc")
    dict_metrics[NAME][1].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_i.nc")
    dict_metrics[NAME][2].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_v.nc")
    dict_metrics[NAME][3].to_netcdf(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_q.nc")




lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

ss=xr.load_dataset(f"{PATH_OUT}/{SEASONS[0]}/{NAME}_q.nc")
ss.pr.plot(levels=np.arange(2,27,3),cmap=cmap_q)
plt.savefig("aaaa.png")
plt.close()

for name,mdl in tqdm(zip(name_models,array_model), total=len(array_model)):
    
    if WH:
        dict_0={name:compute_metrics(get_season(mdl,season=SEAS),meters=True,quantile=0.999,wethours=WH)}
        
        dict_metrics.update(dict_0)
    else:
        dict_0={name:compute_metrics(get_season(mdl,season=SEAS),meters=True,quantile=0.999)}
        
        dict_metrics.update(dict_0)

