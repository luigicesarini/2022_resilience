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


PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
sftlf=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KNMI/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc")
SEAS="JJA"
#missing ETH, HCLIMcom
for MDL in ["CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom","KNMI","SPHERA"]: #
# MDL="KNMI"


    if MDL == "ICTP":
        ll_files=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/mw/*")[:-1]    
    else:
        ll_files=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/mw/*")


    # if MDL == "ETH":
    #     model=fix_eth()
    if MDL == "SPHERA":
        sphera = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*{yr}*") for yr in np.arange(2000,2010)]
        sphera_ds=xr.concat(sphera,dim="time")
        # sphera_ds=sphera_ds.rename({'longitude':'lon','latitude':'lat'})
        model=sphera_ds.load()
        del sphera,sphera_ds
    # elif MDL == "STATIONS":
    #     stations = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/*{yr}*") for yr in np.arange(2000,2010)]
    #     stati_ds = xr.concat(stations,dim="time")
    #     model=stati_ds.load()
    #     del stations,stati_ds
    else:
        model=xr.open_mfdataset(ll_files).load()

    hig_int=model.sel(lat=45.161545, lon=8.210204,method='nearest').mw.load()
    low_int=model.sel(lat=44.712540, lon=7.263958,method='nearest').mw.load()

    np.quantile(hig_int,q=0.9)
    np.quantile(low_int,q=0.9)

    plt.plot(hig_int)
    plt.plot(low_int)
    plt.show()
    print(f"Running {MDL}")

    # if MDL == "STATION":
    #     THR = 0.2
    # else:
    #     THR = 0.1

    # stat1=model.mw.quantile(q=[0.90,0.99])
    # stat2=(model * xr.where(sftlf.sftlf > 50,1,0)).mw.quantile(q=[0.90,0.99])
    #12 and 6 m/s 99th and 90th percentile of sphera's wind speed, sea included 
    #4.67 and 8.7 m/s 99th and 90th percentile of sphera's wind speed, sea NOT included 
    # THR = 8.7
    def _count_threshold_periods(arr,thr):
        above_threshold_periods = []
        current_period = []
        for element in arr:
            if element > thr:
                current_period.append(element)
            else:
                if current_period:
                    above_threshold_periods.append(current_period)
                    current_period = []
        
        # Check if the last period extends beyond the end of the array
        if current_period:
            above_threshold_periods.append(current_period)

        n_event=[len(x) for x in above_threshold_periods]
        max_int=[max(x) for x in above_threshold_periods]
        mean_int=[np.mean(x) for x in above_threshold_periods]
        
        return n_event,max_int,mean_int

    len_per_above_threshold = []
    max_per_periods         = []
    mean_per_periods        = []

    model=model.sel(time=model["time.season"].isin(SEAS))

    for i in tqdm(range(model.lat.shape[0])):
        for j in range(model.lon.shape[0]):
            # if MDL == "STATIONS":
            #     x,y,z=_count_threshold_periods(np.moveaxis(model.mw.values,2,0)[:,i,j])
            # else:
            THR=np.nanquantile(model.mw.values[:,i,j],q=q_mw)
            x,y,z=_count_threshold_periods(model.mw.values[:,i,j],THR)

            len_per_above_threshold.append(x)
            max_per_periods.append(y)
            mean_per_periods.append(z)

    # Saving the object
    with open(f'/mnt/data/lcesarini/EVENTS/mw/{MDL}_len_events_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(len_per_above_threshold, file)

    with open(f'/mnt/data/lcesarini/EVENTS/mw/{MDL}_mw_max_events_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(max_per_periods, file)

    with open(f'/mnt/data/lcesarini/EVENTS/mw/{MDL}_mw_mean_events_{int(q_mw * 100)}_{SEAS}.pkl', 'wb') as file:
        pickle.dump(mean_per_periods, file)


