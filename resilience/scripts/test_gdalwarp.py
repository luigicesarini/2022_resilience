#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import sys
sys.path.append("/home/lcesarini/2022_resilience/")

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

from resilience.utils import *


os.chdir("/home/lcesarini/2022_resilience/")
# from scripts.utils import *

seasons=[
    # 'SON',
    # 'DJF',
    # 'MAM',
    'JJA'
    ]
if __name__ == "__main__":


    cmap_f,cmap_i,cmap_q=get_palettes()
    lvl_f,lvl_i,lvl_q=get_levels()
    
    name_models=[
        # 'ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI',
        # "CMCC_VHR",
        "ENSEMBLE","STATIONS","SPHERA"]
    
    WH=False
    for nm in name_models:
        print(nm)
        for idx,SEAS in enumerate(seasons):
            if WH:
                ds_f=xr.open_dataset(f"output/{SEAS}/{nm}_f_{np.int8(WH)}.nc")
                ds_i=xr.open_dataset(f"output/{SEAS}/{nm}_i_{np.int8(WH)}.nc")
                ds_q=xr.open_dataset(f"output/{SEAS}/{nm}_q_{np.int8(WH)}.nc")
                if nm == "STATIONS":
                    ds_q=ds_q.isel(quantile=0)
                
                plot_panel_rotated(
                    figsize=(8,8),
                    nrow=1,ncol=3,
                    list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
                    name_fig=f"PANEL_{nm}_{SEAS}_WH",
                    list_titles=["Frequency","Intensity","Heavy Prec."],
                    levels=[lvl_f,lvl_i,lvl_q],
                    suptitle=f"{nm}'s metrics for {SEAS}",
                    name_metric=["[fraction]","[mm/h]","[mm/h]"],
                    SET_EXTENT=False,
                    cmap=[cmap_f,cmap_i,cmap_q],
                    SAVE=False
                )

            else:
                ds_f=xr.open_dataset(f"output/{SEAS}/{nm}_f.nc")
                #Put all zeros to np.nan
                ds_f=xr.where(ds_f.pr == 0, np.nan,ds_f.pr).to_dataset(name='pr')
                ds_i=xr.open_dataset(f"output/{SEAS}/{nm}_i.nc")
                ds_q=xr.open_dataset(f"output/{SEAS}/{nm}_q.nc")
                if nm == "STATIONS":
                    ds_q=ds_q.isel(quantile=0)

                plot_panel_rotated(
                    figsize=(13,3.5),
                    nrow=1,ncol=3,
                    list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
                    name_fig=f"PANEL_{nm}_{SEAS}",
                    list_titles=["Frequency","Intensity","Heavy Prec."],
                    levels=[lvl_f,lvl_i,lvl_q],
                    suptitle=f"{nm}'s metrics for {SEAS}",
                    name_metric=["[fraction]","[mm/h]","[mm/h]"],
                    SET_EXTENT=False,
                    cmap=[cmap_f,cmap_i,cmap_q],
                    SAVE=False

                )

            # st_al=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/*.nc")
            # st_al=st_al.isel(time=st_al['time.year'].isin(np.arange(2000,2009))).load()
            
            # st_fre,st_int,st_q99=compute_metrics_stat(get_season(st_al,"JJA"))
            
            # ens_f_tr=ens_f.sel(lon=st_fre.lon,lat=st_fre.lat)
            # ens_i_tr=ens_i.sel(lon=st_int.lon,lat=st_int.lat)
            # ens_q_tr=ens_q.sel(lon=st_q99.lon,lat=st_q99.lat)

            # bias_f = (ens_f_tr - st_fre) / st_fre * 100
            # bias_i = (ens_i_tr - st_int) / st_int * 100
            # bias_q = (ens_q_tr - st_q99) / st_q99 * 100
            
            # np.nanmean(bias_f.freq)
            # np.nanmean(bias_i.int)
            # np.nanmean(bias_q.q)

            # lvl_q=np.arange(2,19,2)
            # lvl_q=9
            # plot_panel_rotated(
            #     nrow=1,ncol=3,
            #     list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
            #     name_fig=f"PANEL_{nm}_{SEAS}",
            #     list_titles=["Frequency","Intensity","Heavy Prec."],
            #     levels=[lvl_f,lvl_i,lvl_q],
            #     suptitle=f"Ensemble's metrics for {SEAS}",
            #     name_metric=["[fraction]","[mm/h]","[mm/h]"],
            #     SET_EXTENT=False,
            #     cmap=[cmap_f,cmap_i,cmap_q]
            # )

            # plot_panel_rotated(
            #     nrow=1,ncol=1,
            #     list_to_plot=[ds_q.isel(quantile=0).pr],
            #     name_fig=f"PANEL_{SEAS}",
            #     list_titles=["Heavy Prec."],
            #     levels=[lvl_f,lvl_i,lvl_q],
            #     suptitle=f"Station {SEAS}",
            #     name_metric=["[mm/h]"],
            #     SET_EXTENT=False,
            #     #cmap=[cmap_q]
            # )
            # print(ds_f)



