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
from resilience.utils import ComputeMetrics

from resilience.utils.fix_year_eth import fix_eth

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-ev','--EV', 
                    required=True,default='pr',
                    choices=['pr','mw'], 
                    help='Environmental variable to use')

args = parser.parse_args()

EV=args.EV

def compute_and_print(DS,SEAS,Q,EV,NAME):
    if EV == 'pr':
        f,i,v,q=ComputeMetrics(get_season(DS,season=SEAS)).compute_tp(meters=True,quantile=Q)
        f.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_f.nc")
        i.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_i.nc")
        v.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_v.nc")
        q.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_q.nc")
    elif EV == 'mw':
        obj_metric=ComputeMetrics(get_season(DS,season=SEAS)).compute_wind(thr=6,quantile=Q)
        obj_metric[0].to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_m.nc")
        obj_metric[1].to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_f.nc")
        obj_metric[2].to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_{Q}_q.nc")

max_vento=sta.max(dim='time')

xr.where(np.isnan(max_vento.mw),np.nan,1).to_netcdf("/mnt/data/lcesarini/na_stations_wind.nc")

if EV == "pr":
    print("fix path to stations's precipitation")
    # sta=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/{EV}/*.nc").load()
else:
    sta=[xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/{EV}/wind_{year}.nc").load() for year in np.arange(2000,2010)]
    sta_concat=xr.concat(sta,dim='time')
    sta=sta_concat.rename({"__xarray_dataarray_variable__":"mw"})
compute_and_print(sta,"JJA",0.99,EV, "STATIONS")
compute_and_print(sta,"JJA",0.999,EV,"STATIONS")
compute_and_print(sta,"DJF",0.99,EV, "STATIONS")
compute_and_print(sta,"DJF",0.999,EV,"STATIONS")

if EV == "pr":
    print("fix path to sphera's precipitation")
    # sph=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/{EV}/*.nc").load()
else:
    sph=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/{EV}/*.nc").load()
compute_and_print(sph,"JJA",0.99,EV, "SPHERA")
compute_and_print(sph,"JJA",0.999,EV,"SPHERA")
compute_and_print(sph,"DJF",0.99,EV, "SPHERA")
compute_and_print(sph,"DJF",0.999,EV,"SPHERA")



if EV == "pr":
    eth=fix_eth()
else:
    eth=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ETH/CPM/{EV}/*.nc").load()
compute_and_print(eth,"JJA",0.99,EV, "ETH")
compute_and_print(eth,"JJA",0.999,EV,"ETH")
compute_and_print(eth,"DJF",0.99,EV, "ETH")
compute_and_print(eth,"DJF",0.999,EV,"ETH")

del eth

print("ETH done")

if EV == "pr":
    mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/{EV}/MOHC_ECMWF-ERAINT_*.nc").load()
    mohc_rg=mohc_rg.sel(time=mohc_rg['time.year'].isin(np.arange(2000,2010)))
    compute_and_print(mohc_rg,"JJA",0.99,EV, "MOHC")
    compute_and_print(mohc_rg,"JJA",0.999,EV,"MOHC")
    compute_and_print(mohc_rg,"DJF",0.99,EV, "MOHC")
    compute_and_print(mohc_rg,"DJF",0.999,EV,"MOHC")

    del mohc_rg
    print("MOHC done")

if EV == 'pr':
    ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/{EV}/ICTP_ECMWF-ERAINT_*.nc").load()
    ictp_rg=ictp_rg.sel(time=ictp_rg['time.year'].isin(np.arange(2000,2010)))
else:
    ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/{EV}/*.nc").load()
    ictp_rg=ictp_rg.sel(time=ictp_rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(ictp_rg,"JJA",0.99,EV, "ICTP")
compute_and_print(ictp_rg,"JJA",0.999,EV,"ICTP")
compute_and_print(ictp_rg,"DJF",0.99,EV, "ICTP")
compute_and_print(ictp_rg,"DJF",0.999,EV,"ICTP")

del ictp_rg
print("ICTP done")

if EV == 'pr':
    hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/{EV}/HCLIMcom_ECMWF-ERAINT_*.nc").load()
    hcli_rg=hcli_rg.sel(time=hcli_rg['time.year'].isin(np.arange(2000,2010)))
else:
    hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/{EV}/*.nc").load()
    hcli_rg=hcli_rg.sel(time=hcli_rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(hcli_rg,"JJA",0.99,EV, "HCLIMcom")
compute_and_print(hcli_rg,"JJA",0.999,EV,"HCLIMcom")
compute_and_print(hcli_rg,"DJF",0.99,EV, "HCLIMcom")
compute_and_print(hcli_rg,"DJF",0.999,EV,"HCLIMcom")

del hcli_rg

if EV == 'pr':
    cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/{EV}/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
    cnrm_rg=cnrm_rg.sel(time=cnrm_rg['time.year'].isin(np.arange(2000,2010)))
else:
    cnrm_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CNRM/CPM/{EV}/*.nc").load()
    cnrm_rg=cnrm_rg.sel(time=cnrm_rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(cnrm_rg,"JJA",0.99,EV, "CNRM")
compute_and_print(cnrm_rg,"JJA",0.999,EV,"CNRM")
compute_and_print(cnrm_rg,"DJF",0.99,EV, "CNRM")
compute_and_print(cnrm_rg,"DJF",0.999,EV,"CNRM")

del cnrm_rg

if EV == 'pr':
    knmi_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/{EV}/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
    knmi_rg=knmi_rg.sel(time=knmi_rg['time.year'].isin(np.arange(2000,2010)))
else:
    knmi_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KNMI/CPM/{EV}/*.nc").load()
    knmi_rg=knmi_rg.sel(time=knmi_rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(knmi_rg,"JJA",0.99,EV, "KNMI")
compute_and_print(knmi_rg,"JJA",0.999,EV,"KNMI")
compute_and_print(knmi_rg,"DJF",0.99,EV, "KNMI")
compute_and_print(knmi_rg,"DJF",0.999,EV,"KNMI")

del knmi_rg
print("KNMI done")

if EV == 'pr':
    cmcc_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CMCC/CPM/{EV}/CMCC_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
    cmcc_rg=cmcc_rg.sel(time=cmcc_rg['time.year'].isin(np.arange(2000,2010)))
else:
    cmcc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CMCC/CPM/{EV}/*.nc").load()
    cmcc_rg=cmcc_rg.sel(time=cmcc_rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(cmcc_rg,"JJA",0.99,EV, "CMCC")
compute_and_print(cmcc_rg,"JJA",0.999,EV,"CMCC")
compute_and_print(cmcc_rg,"DJF",0.99,EV, "CMCC")
compute_and_print(cmcc_rg,"DJF",0.999,EV,"CMCC")

del cmcc_rg

if EV == 'pr':
    kit__rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KIT/CPM/{EV}/KIT_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
    kit__rg=kit__rg.sel(time=kit__rg['time.year'].isin(np.arange(2000,2010)))
else:
    kit__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KIT/CPM/{EV}/*.nc").load()
    kit__rg=kit__rg.sel(time=kit__rg['time.year'].isin(np.arange(2000,2010)))

compute_and_print(kit__rg,"JJA",0.99,EV, "KIT")
compute_and_print(kit__rg,"JJA",0.999,EV,"KIT")
compute_and_print(kit__rg,"DJF",0.99,EV, "KIT")
compute_and_print(kit__rg,"DJF",0.999,EV,"KIT")

del kit__rg

print("KIT done")

# name_models=['CMCC','KIT']
# ds_models=[cmcc_rg,kit__rg]

# for SEAS in tqdm(['SON','DJF','MAM','JJA'],total=4):
#         for NAME,DS in zip(name_models,ds_models):
#             cmcc_f,cmcc_i,cmcc_v,cmcc_q=compute_metrics(get_season(DS,season=SEAS),meters=True,quantile=0.999)
            
#             cmcc_f.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_f.nc")
#             cmcc_i.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_i.nc")
#             cmcc_v.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_v.nc")
#             cmcc_q.to_netcdf(f"/mnt/data/lcesarini/eval/{EV}/{SEAS}/{NAME}_q.nc")
