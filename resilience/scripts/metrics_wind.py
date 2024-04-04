#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
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


os.chdir("/home/lcesarini/2022_resilience/")
seasons=['SON','DJF','MAM','JJA']
list_ms=['Frequency','Intensity','Heavy Prec.']
abbr_ms=['f','i','q']
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
ENV_VAR='mw'

PLOT_BOXPLOTS=False

if __name__ == "__main__":
    # eth__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ETH/CPM/{ENV_VAR}/*" ).load()
    # # mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/{ENV_VAR}/*").load()
    # ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/{ENV_VAR}/*").load()
    # hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/{ENV_VAR}/*").load()
    # cnrm_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CNRM/CPM/{ENV_VAR}/*" ).load()
    # knmi_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KNMI/CPM/{ENV_VAR}/*" ).load()
    # kit__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/KIT/CPM/{ENV_VAR}/*" ).load()
    # cmcc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/CMCC/CPM/{ENV_VAR}/*" ).load()

    # name_models=['ETH','ICTP','HCLIMcom','CNRM','KNMI','KIT','CMCC']
    # array_model=[eth__rg,ictp_rg,hcli_rg,cnrm_rg,knmi_rg,kit__rg,cmcc_rg]
    
    THR=6
    # for SEAS in ['MAM','SON','JJA','DJF']:
    #     dict_metrics={}

    #     for name,mdl in tqdm(zip(name_models,array_model), total=len(array_model)):
    #         #INITIALIZE COMPUTE METRICS OBJECT
    #         obj_metric=ComputeMetrics(get_season(mdl,season=SEAS))
            
    #         dict_0={name:obj_metric.compute_wind(thr=THR)}
            
    #         dict_metrics.update(dict_0)


    #     mean_speed=xr.concat([dict_metrics[name][0] for name in name_models],dim='model').mean(dim='model')
    #     mean_f_thr=xr.concat([dict_metrics[name][1] for name in name_models],dim='model').mean(dim='model')
    #     mean_q__99=xr.concat([dict_metrics[name][2] for name in name_models],dim='model').mean(dim='model')

    #     for idx,metrics in enumerate(['m','f','q']):
    #         for mdl in name_models:
    #                 dict_metrics[mdl][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{ENV_VAR}/{mdl}_{THR}_{metrics}.nc")

    #     for idx,(mdl,metrics) in enumerate(zip([mean_speed,mean_f_thr,mean_q__99],['m','f','q'])):
    #             mdl.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{ENV_VAR}/ENSEMBLE_{THR}_{metrics}.nc")


    sphe_rg = xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*" ).load()

    for SEAS in ['MAM','SON','JJA','DJF']:#
        dict_metrics={}

        for name,mdl in tqdm(zip(['SPHERA'],[sphe_rg])):
            #INITIALIZE COMPUTE METRICS OBJECT
            obj_metric=ComputeMetrics(get_season(mdl,season=SEAS))
            
            dict_0={name:obj_metric.compute_wind(thr=THR)}
            
            dict_metrics.update(dict_0)

        for idx,metr in enumerate(['m','f','q']):
            for mdl in ['SPHERA']:
                    dict_metrics[mdl][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{ENV_VAR}/{mdl}_{THR}_{metr}.nc")



    
