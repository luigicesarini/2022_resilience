#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience")
import pickle
import argparse
# import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import seaborn as sns
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature

from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth

os.chdir("/home/lcesarini/2022_resilience/")

"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-thr","--threshold",
                    help="Quantile to use as threshold",
                    required=True,default=0.99  
                    )

args = parser.parse_args()

q_mw = float(args.threshold)
q_pr = float(args.threshold)

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
#missing ETH, HCLIMcom
# MDL="CNRM"
SEAS="JJA"

"""
WRITE THE CODE TO COMPUTE THE EVENTS FOR CO-OCCURRENCE ANALYSIS OF EVENTS

THRESHOLDS referred to the single cell and season
90th percentile
pr: 3.6 mm
mw: 6 m/s

99th percentile
pr: 11.4 mm
mw: 12 m/s

"""

# arr_pr=model_pr.pr.values[:,i,j]
# arr_mw=model_mw.mw.values[:,i,j]
# thr_pr=THR_PR
# thr_mw=THR_MW

def concurrent_events(arr_pr,arr_mw,thr_pr,thr_mw):
    above_threshold_periods = []
    current_period = []
    for ts in np.arange(arr_pr.shape[0]):
        if (arr_pr[ts] > thr_pr) and (arr_mw[ts] > thr_mw):
            current_period.append([arr_pr[ts],arr_mw[ts]])
        else:
            if current_period:
                above_threshold_periods.append(current_period)
                current_period = []
    
    # Check if the last period extends beyond the end of the array
    if current_period:
        above_threshold_periods.append(current_period)
  
    n_event=[len(x) for x in above_threshold_periods]
    max_int=[np.max(x,axis=0) for x in above_threshold_periods]
    mean_int=[np.mean(x,axis=0) for x in above_threshold_periods]
    
    return n_event,max_int,mean_int
"""
READ THE DATA BASED ON THE MODEL
"""

for MDL in ["ETH","ICTP","HCLIMcom","KNMI","CMCC","CNRM","KIT","SPHERA"]: #


    if MDL == "ICTP":
        ll_files_pr=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/pr/*")[:-1]    
        ll_files_mw=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/mw/*")
    else:
        ll_files_pr=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/pr/*")
        ll_files_mw=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/mw/*")


    if MDL == "ETH":
        model_pr=fix_eth()
        model_mw=xr.open_mfdataset(ll_files_mw).load()
    elif MDL == "SPHERA":
        sphera_pr = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{yr}*") for yr in np.arange(2000,2010)]
        sphera_ds_pr=xr.concat(sphera_pr,dim="time")
        sphera_ds_pr=sphera_ds_pr.rename({'longitude':'lon','latitude':'lat'})
        model_pr=sphera_ds_pr.load()
        
        sphera_mw = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*{yr}*") for yr in np.arange(2000,2010)]
        sphera_ds_mw=xr.concat(sphera_mw,dim="time")
        # sphera_ds=sphera_ds.rename({'longitude':'lon','latitude':'lat'})
        model_mw=sphera_ds_mw.load()
        del sphera_pr,sphera_ds_pr,sphera_mw,sphera_ds_mw
    elif MDL == "CNRM":
        model_pr=xr.open_mfdataset(ll_files_pr[:10]).load()
        model_mw=xr.open_mfdataset(ll_files_mw).load()
    else:
        model_pr=xr.open_mfdataset(ll_files_pr).load()
        model_mw=xr.open_mfdataset(ll_files_mw).load()

    """
    RUN THE FUNCTION AND SAVE TO A PICKLE
    """
    if MDL=="SPHERA":
        model_mw=model_mw.sel(time=model_mw.time.isin(model_pr.time))
        model_pr=model_pr.sel(time=model_pr.time.isin(model_mw.time))

    print(f"Running {MDL}")

    model_pr=model_pr.sel(time=model_pr["time.season"].isin(SEAS))
    model_mw=model_mw.sel(time=model_mw["time.season"].isin(SEAS))

    len_per_above_threshold = []
    max_per_periods         = []
    mean_per_periods        = []

    for i in tqdm(range(model_pr.lat.shape[0])):
        for j in range(model_pr.lon.shape[0]):
            
            THR_PR=np.nanquantile(model_pr.pr.values[:,i,j][model_pr.pr.values[:,i,j] > 0.1],q=q_pr)
            THR_MW=np.nanquantile(model_mw.mw.values[:,i,j],q=q_mw)

            x,y,z=concurrent_events(model_pr.pr.values[:,i,j],model_mw.mw.values[:,i,j],THR_PR,THR_MW)

            len_per_above_threshold.append(x)
            max_per_periods.append(y)
            mean_per_periods.append(z)


    # Saving the object
    with open(f'/mnt/data/lcesarini/EVENTS/combined/{MDL}_len_events_{int(q_pr * 100)}_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(len_per_above_threshold, file)

    with open(f'/mnt/data/lcesarini/EVENTS/combined/{MDL}_max_events_{int(q_pr * 100)}_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(max_per_periods, file)

    with open(f'/mnt/data/lcesarini/EVENTS/combined/{MDL}_mean_events_{int(q_pr * 100)}_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(mean_per_periods, file)
