#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
"""



"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/") 
import time
import argparse
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

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 
PATH_COSMO=f"/mnt/data/lcesarini/COSMO/REA_2"

"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-s","--season",
                    help="Season over which the bootstrap is evaluated",
                    required=True,default="JJA",
                    choices=["MAM","JJA","SON","DJF"]  
                    )
parser.add_argument("-ref","--reference",
                    help="Reference dataset for bias correction",
                    required=True,default="STATIONS",
                    choices=["SPHERA","STATIONS"]  
                    )
parser.add_argument("-csv","--to_csv",
                    help="Save the output in csv format",
                    action="store_true",
                    )
args = parser.parse_args()

seas=args.season    
REF=args.reference
if REF == "SPHERA":
    ll_bias_eqm=[xr.load_dataset(file) for file in glob(f"output/bootstrap/EQM/{seas}/*{REF}**nc")]
    ll_bias_qdm=[xr.load_dataset(file) for file in glob(f"output/bootstrap/QDM/{seas}/*{REF}**nc")]
    ll_bias_ori=[xr.load_dataset(file) for file in glob(f"output/bootstrap/ORI/{seas}/*{REF}**nc")]
else:
    ll_bias_eqm=[xr.load_dataset(file) for file in glob(f"output/bootstrap/EQM/{seas}/bias_heavy_prec_EQM_{seas}_*")]
    ll_bias_qdm=[xr.load_dataset(file) for file in glob(f"output/bootstrap/QDM/{seas}/bias_heavy_prec_QDM_{seas}_*")]
    ll_bias_ori=[xr.load_dataset(file) for file in glob(f"output/bootstrap/ORI/{seas}/bias_heavy_prec_{seas}_*")]

print(len(ll_bias_eqm),len(ll_bias_qdm))

ds_eqm=xr.concat(ll_bias_eqm,np.arange(len(ll_bias_eqm))).rename({"concat_dim":"iteration"})
ds_qdm=xr.concat(ll_bias_qdm,np.arange(len(ll_bias_qdm))).rename({"concat_dim":"iteration"})
ds_ori=xr.concat(ll_bias_ori,np.arange(len(ll_bias_ori))).rename({"concat_dim":"iteration"})

x1,x2=np.where(~np.isnan(ds_ori.pr.values[0,:,:]))

list_ori_ss=[]
list_eqm_ss=[]
list_qdm_ss=[]
for i,j in zip(x1,x2):
    if ((np.quantile(ds_eqm.pr.values[:,i,j],q=0.025) > 0) | (np.quantile(ds_eqm.pr.values[:,i,j],q=0.975) < 0)):
        lon1,lat1=ds_eqm.pr.isel(lon=j,lat=i).lon.item(),ds_eqm.pr.isel(lon=j,lat=i).lat.item()
        list_eqm_ss.append([lon1,lat1])
    if ((np.quantile(ds_qdm.pr.values[:,i,j],q=0.025) > 0) | (np.quantile(ds_qdm.pr.values[:,i,j],q=0.975) < 0)):
        lon1,lat1=ds_qdm.pr.isel(lon=j,lat=i).lon.item(),ds_qdm.pr.isel(lon=j,lat=i).lat.item()
        list_qdm_ss.append([lon1,lat1])
    if ((np.quantile(ds_ori.pr.values[:,i,j],q=0.025) > 0) | (np.quantile(ds_ori.pr.values[:,i,j],q=0.975) < 0)):
        lon2,lat2=ds_ori.pr.isel(lon=j,lat=i).lon.item(),ds_ori.pr.isel(lon=j,lat=i).lat.item()
        list_ori_ss.append([lon2,lat2])

# for i,j in zip(x1,x2):
#     lon1,lat1=ds_eqm.pr.isel(lon=j,lat=i).lon.item(),ds_eqm.pr.isel(lon=j,lat=i).lat.item()
#     list_eqm_ss.append([lon1,lat1])

#     lon2,lat2=ds_ori.pr.isel(lon=j,lat=i).lon.item(),ds_ori.pr.isel(lon=j,lat=i).lat.item()
#     list_ori_ss.append([lon2,lat2])

print(f"{REF} {seas}\nRAW:{len(list_ori_ss) / x1.shape[0]:.4f}\nEQM:{len(list_eqm_ss) / x1.shape[0]:.4f}\nQDM:{len(list_qdm_ss) / x1.shape[0]:.4f}")

# fig,ax=plt.subplots(1,3,figsize=(15,6))
# [sns.kdeplot(ds_eqm.pr.sel(lon=ln,lat=lt),ax=ax[0]) for ln,lt in list_eqm_ss] 
# [sns.kdeplot(ds_qdm.pr.sel(lon=ln,lat=lt),ax=ax[1]) for ln,lt in list_qdm_ss] 
# [sns.kdeplot(ds_ori.pr.sel(lon=ln,lat=lt),ax=ax[2]) for ln,lt in list_ori_ss]
# [ax[_].set_title(f"{name}") for _,name in enumerate(["EQM","QDM","RAW"])]
# [ax[_].vlines(x=0,ymin=0,ymax=0.2,color="green",linewidth=5) for _ in range(3)]
# plt.suptitle(f"{seas}")
# plt.show()
# time.sleep(3)
# plt.close()
# plt.savefig("../pistolina.png")
# plt.close()

ds_band_ori=ds_ori.quantile(dim='iteration',q=[0.025,0.975])
ds_band_eqm=ds_eqm.quantile(dim='iteration',q=[0.025,0.975])
ds_band_qdm=ds_qdm.quantile(dim='iteration',q=[0.025,0.975])
x_ori=xr.where((ds_band_ori.pr.isel(quantile=0) > 0) | (ds_band_ori.pr.isel(quantile=1) < 0), 1,0)
x_eqm=xr.where((ds_band_eqm.pr.isel(quantile=0) > 0) | (ds_band_eqm.pr.isel(quantile=1) < 0), 1,0)
x_qdm=xr.where((ds_band_qdm.pr.isel(quantile=0) > 0) | (ds_band_qdm.pr.isel(quantile=1) < 0), 1,0)

to_csv=args.to_csv
if to_csv:
    nc_to_csv(x_ori,f"bias_boot_ori_{seas}_{REF}",M="SS",csv=True)
    nc_to_csv(x_eqm,f"bias_boot_eqm_{seas}_{REF}",M="SS",csv=True)
    nc_to_csv(x_qdm,f"bias_boot_qdm_{seas}_{REF}",M="SS",csv=True)


# nc_to_csv(x_ori,f"bias_boot_ori_{seas}_{REF}",M="SS",csv=False)

# pd.read_csv("csv/bias_boot_ori_DJF_SPHERA.csv").SS.sum()


