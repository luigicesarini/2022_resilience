#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                            r2_score,mean_absolute_percentage_error)

from utils import *

import warnings
warnings.filterwarnings('ignore')
path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"

os.chdir("/home/lcesarini/2022_resilience/")
from scripts.utils import *


ij=pd.read_table("griglia_ele.txt",header=None)
ij.columns=["name","i","j"]
ij.i=ij.i-1
ij.j=ij.j-1
ij
#VE0011
i,j=197,142
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
ele=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/datiDallan/CPM_ETH_Italy_2000-2009_pr_hour.nc")
eth__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ETH/CPM/pr/ETH_ECMWF-ERAINT_*.nc").load()
mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/pr/MOHC_ECMWF-ERAINT_*.nc").load()
ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/pr/ICTP_ECMWF-ERAINT_*.nc").load()
hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_200*.nc").load()
cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()


for idx in tqdm(np.arange(0,ij.shape[0]),total=ij.shape[0]):
    idx=np.random.choice(np.arange(0,ij.shape[0]))
    B=ele.isel(rlon=ij.i[idx],rlat=ij.j[idx]).pr
    A=eth__rg.sel(lon=B.lon.item(),lat=B.lat.item(),method='nearest').pr
    metrics=[ij.name[idx],
             r2_score(A.values,B.values),
             mean_absolute_percentage_error(A.values,B.values) * 100,
             np.sqrt(mean_squared_error(A.values,B.values))
    ]
    # 1 degree : 111.1 km = diff° : x
    print(
    f"""
    {np.abs((111.139 * (A.lon.item()-11.20880581)) * 1000):.2f}m
    {np.abs((111.139 * (A.lat.item()-45.62689319)) * 1000):.2f}m
    """
    )
    
    
    pd.DataFrame(pd.to_datetime(A.time.values)).\
        to_csv(f"/home/lcesarini/2022_resilience/data/dates_{ij.name[idx]}.csv",
               index=0,
               header = ["date"]
                )
    pd.DataFrame(A.values).\
        to_csv(f"/home/lcesarini/2022_resilience/data/pr_{ij.name[idx]}.csv",
               index=0,
               header = ["pr"]
                )


    A0=xr.where(A > 0.1, A,0)
    B0=xr.where(B > 0.1, B,0)
    
    bias_ts=(A0 - B0) / B0 * 100    
    bias_ecdf=(np.sort(A0) - np.sort(B0)) / np.sort(B0) * 100    

    np.nanmax(bias_ecdf[np.isfinite(bias_ecdf)])
    np.nanmax(bias_ts[np.isfinite(bias_ts)])

    bias_ts.isel(time=np.isfinite(bias_ts))[12413]

    pd.DataFrame(metrics).transpose().to_csv("/home/lcesarini/2022_resilience/data/metriche_check_stazioni.csv",index=0,
                                            mode="a" if os.path.exists("/home/lcesarini/2022_resilience/data/metriche_check_stazioni.csv") else "w",
                                            header = None if os.path.exists("/home/lcesarini/2022_resilience/data/metriche_check_stazioni.csv") else["r2","mape","rmse"])


    orig_0 = xr.where(eth__rg.pr > 0.1, eth__rg.pr, 0)
    eleo_0 = xr.where(ele.pr > 0.1, ele.pr, 0)

    bias_all=(orig_0.values - eleo_0.values) / eleo_0.shape * 100

    (A.sel(time="2008-07-02T18:33:00")-B.sel(time="2008-07-02T18:33:00")) / B.sel(time="2008-07-02T18:33:00") * 100
    (A[4] - B[4]) / B[4] * 100    

    (np.quantile(A0,q=[0.8375,0.9,0.95,0.99])-np.quantile(B0,q=[0.8375,0.9,0.95,0.99])) / np.quantile(B0,q=[0.8375,0.9,0.95,0.99]) * 100
    

    eth_orig=xr.open_dataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200001010030_200012312330.nc").load()
    eth_orig.lon[152,150]-eth_orig.lon[152,151]
    ele.lon[152,150]-ele.lon[152,151]
    
    eth_orig.isel(rlon=ij.i[idx],rlat=ij.j[idx]).pr.quantile(q=0.95).item()


    # https://nominatim.openstreetmap.org/search?<params>
    # https://nominatim.openstreetmap.org/search?q=4+alboino+via,+pavia&format=json&addressdetails=0