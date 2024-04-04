#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python

import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")

sys.path.append("/mnt/beegfs/lcesarini/SkillMetrics/")
import skill_metrics as sm

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
from collections import OrderedDict
from shapely.geometry import mapping
from scipy.signal import correlate2d
from cartopy import feature as cfeature

os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")

from resilience.utils import *
from resilience.utils import return_obj_taylor

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-s','--season', 
                    required=False,default="JJA",
                    choices=["SON","DJF","MAM","JJA"], 
                    help='season to analyse')
parser.add_argument('-m','--metrics', 
                    required=True,default="q",
                    choices=["f","i","q"],
                    help='metrics to analyse')
parser.add_argument('-ev','--env_var', 
                    required=True,default="pr",
                    choices=["pr_sta","mw","pr"],
                    help='Envoronmental variable to analyse')

args = parser.parse_args()


# SAVE=False
PATH_SPHERA_OUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr"
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
PATH_OUTPUT=f"/mnt/beegfs/lcesarini/2022_resilience/output/"

pred=xr.open_dataset(f"{PATH_OUTPUT}/JJA/ENSEMBLE_q.nc").pr.values
ref=xr.open_dataset(f"{PATH_OUTPUT}/JJA/SPHERA_q.nc").pr.values

#plot pred e ref on two subplots (1 row, 2 columns)
# fig,axs=plt.subplots(1,2,figsize=(12,4))
# xr.open_dataset(f"{PATH_OUTPUT}/JJA/ENSEMBLE_q.nc").pr.plot.pcolormesh(ax=axs[0])
# xr.open_dataset(f"{PATH_OUTPUT}/JJA/SPHERA_q.nc").pr.plot.pcolormesh(ax=axs[1])
# if SAVE:
#     plt.savefig()
# plt.show()



shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

shp_ita=gpd.read_file("/mnt/beegfs/lcesarini/2022_resilience/gadm36_ITA.gpkg", layer="gadm36_ITA_0")
mask=xr.open_dataset("data/mask_stations_nan_common.nc")

SEAS=args.season
M=args.metrics
EV=args.env_var
# M='q'
# SEAS='JJA'
if EV == 'pr_sta':
    na_stat=xr.load_dataset(f"{PATH_OUTPUT}/JJA/STATIONS_q.nc").isel(quantile=0)

    ens=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_{M}.nc")
    rea=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/SPHERA_{M}.nc")
    if M=='q':
        sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc").isel(quantile=0)
    else:
        sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc")



    # pattern_correlation(ens.isel(lon=range(6),lat=range(6)),
    #                     rea.isel(lon=range(6),lat=range(6)),type="uncentred")
    xmin,xmax=get_range(na_stat.lon)
    ymin,ymax=get_range(na_stat.lat)

    ens_sta = clip_ds(ens * mask.mask,xmin,xmax,ymin,ymax)
    rea_sta = clip_ds(rea * mask.mask,xmin,xmax,ymin,ymax)

    # M='q'

    labels,markers = return_obj_taylor(LABELS=True,MARKERS=True)


    fig,axs=plt.subplots(figsize=(30,6),nrows=1,ncols=3)
    axs=axs.flatten()
    for j,M in enumerate( ["f","i","q"] ):
        sdev=[]
        crmsd=[]
        ccoef=[]
        for idx,SEAS in enumerate(["DJF","MAM","JJA","SON"]):
            pred=xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_{M}.nc").pr.values
            ref=xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/SPHERA_{M}.nc").pr.values

            if M=='q':
                sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc").isel(quantile=0).pr.values
            else:
                sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc").pr.values

                
                

            
            if idx ==0:
                sdev.append(np.nanstd(ref))
                crmsd.append(sm.centered_rms_dev(ref,ref))
                ccoef.append(pattern_correlation(ref,ref))
            sdev.append(np.nanstd(pred))
            crmsd.append(sm.centered_rms_dev(pred,ref))
            ccoef.append(pattern_correlation(pred,ref))

            ens_sta = clip_ds(xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_{M}.nc") * mask.mask,xmin,xmax,ymin,ymax)
            sph_sta = clip_ds(xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/SPHERA_{M}.nc") * mask.mask,xmin,xmax,ymin,ymax)
            # sph_sta = clip_ds(rea * mask.mask,xmin,xmax,ymin,ymax)
            if M == 'f':
            #     print(f"""
            #         ENSEMBLE {SEAS}
            #         {np.nanmean(ens_sta.pr.values):.2f}
            #         SPHERA {SEAS}
            #         {np.nanmean(sph_sta.pr.values):.2f}
            #         STATIONS {SEAS}
            #         {np.nanmean(sta):.2f}
            #           """)
            #     print(ens_sta.pr.values.reshape(-1)[~np.isnan(ens_sta.pr.values.reshape(-1))].shape)
            #     print(sph_sta.pr.values.reshape(-1)[~np.isnan(sph_sta.pr.values.reshape(-1))].shape)
                sta=xr.where(sta == 0, np.nan,sta)
                # print(yy.reshape(-1)[~np.isnan(yy.reshape(-1))].shape)
                
            sdev.append(np.nanstd(ens_sta.pr.values))
            crmsd.append(sm.centered_rms_dev(ens_sta.pr.values,sta))
            ccoef.append(pattern_correlation(ens_sta.pr.values,sta))

            sdev.append(np.nanstd(sph_sta.pr.values))
            crmsd.append(sm.centered_rms_dev(sph_sta.pr.values,sta))
            ccoef.append(pattern_correlation(sph_sta.pr.values,sta))


        # print(labels)

        # plt.figure(figsize=(16,6))
        sm.taylor_diagram(axs[j],
                        np.array(sdev),
                        np.array(crmsd),
                        np.array(ccoef), 
                        markers=markers,
                        markerLabel = labels, 
                        markerLabelColor = 'r', 
                        markerLegend = 'on',# if M =='q' else 'off', 
                        markerColor = 'r',
                        styleOBS = '-', 
                        colOBS = 'r', 
                        #   markerobs = 'o',
                        markerLayout=[6,2],
                        markerSize = 10, 
                        tickRMS = [0.0],
                        tickCOR = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                        tickRMSangle = 115,
                        showlabelsRMS = 'off',
                        titleRMS = 'off', titleOBS = 'Ref')
        axs[j].set_title(f"{'Heavy Precipitation' if M=='q' else 'Intensity' if M=='i' else 'Frequency'}",fontsize=18,pad=30)

    plt.suptitle(f"{'Precipitation with comparison on station' if M=='q' else 'Intensity' if M=='i' else 'Frequency'}",fontsize=30,y=1.05)

    plt.savefig(f"/mnt/beegfs/lcesarini/taylor_{EV}.png",dpi=300,bbox_inches="tight")

if EV == 'pr':

    labels,markers = return_obj_taylor_sph(LABELS=True,MARKERS=True)


    fig,axs=plt.subplots(figsize=(16,4.5),nrows=1,ncols=3)
    axs=axs.flatten()
    for j,M in enumerate( ["f","i","q"] ):
        sdev=[]
        crmsd=[]
        ccoef=[]
        for idx,SEAS in enumerate(["DJF","MAM","JJA","SON"]):
            pred=xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_{M}.nc").pr.values
            ref=xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/SPHERA_{M}.nc").pr.values

            if idx ==0:
                sdev.append(np.nanstd(ref))
                crmsd.append(sm.centered_rms_dev(ref,ref))
                ccoef.append(pattern_correlation(ref,ref))
            sdev.append(np.nanstd(pred))
            crmsd.append(sm.centered_rms_dev(pred,ref))
            ccoef.append(pattern_correlation(pred,ref))

                
        # print(labels)

        # plt.figure(figsize=(16,6))
        sm.taylor_diagram(axs[j],
                        np.array(sdev),
                        np.array(crmsd),
                        np.array(ccoef), 
                        markers=markers,
                        markerLabel = labels, 
                        markerLabelColor = 'r', 
                        markerLegend = 'on',# if M =='q' else 'off', 
                        markerColor = 'r',
                        styleOBS = '-', 
                        colOBS = 'r', 
                        markerLayout=[2,2],
                        markerobs = 'o',
                        markerSize = 10, 
                        tickRMS = [0.0],
                        tickCOR = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                        tickRMSangle = 115,
                        showlabelsRMS = 'off',
                        titleRMS = 'off', titleOBS = 'Ref')
        axs[j].set_title(f"{'Heavy Precipitation' if M=='q' else 'Intensity' if M=='i' else 'Frequency'}",fontsize=18,pad=30)

    plt.suptitle(f'')

    plt.savefig(f"/mnt/beegfs/lcesarini/taylor_{EV}.png",dpi=300,bbox_inches="tight")

if EV == 'mw':
    THR=6
    ens=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{EV}/ENSEMBLE_{THR}_{M}.nc")
    rea=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{EV}/SPHERA_{THR}_{M}.nc")

    ll_metrics=["m","f","q"]
    labels,markers = return_obj_taylor_sph(LABELS=True,MARKERS=True)


    fig,axs=plt.subplots(figsize=(16,4.5),nrows=1,ncols=3)
    axs=axs.flatten()
    for j,M in enumerate(ll_metrics):
        sdev=[]
        crmsd=[]
        ccoef=[]
        for idx,SEAS in enumerate(["DJF","MAM","JJA","SON"]):
            pred=xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/{EV}/ENSEMBLE_{THR}_{M}.nc")[EV].values
            ref =xr.open_dataset(f"{PATH_OUTPUT}/{SEAS}/{EV}/SPHERA_{THR}_{M}.nc"  )[EV].values

            
            if idx ==0:
                sdev.append(np.nanstd(ref))
                crmsd.append(sm.centered_rms_dev(ref,ref))
                ccoef.append(pattern_correlation(ref,ref))
            sdev.append(np.nanstd(pred))
            crmsd.append(sm.centered_rms_dev(pred,ref))
            ccoef.append(pattern_correlation(pred,ref))
                

        sm.taylor_diagram(axs[j],
                        np.array(sdev),
                        np.array(crmsd),
                        np.array(ccoef), 
                        markers=markers,
                        markerLabel = labels, 
                        markerLabelColor = 'r', 
                        markerLegend = 'on',# if M =='q' else 'off', 
                        markerColor = 'r',
                        styleOBS = '-', 
                        colOBS = 'r', 
                        markerLayout=[1,2],
                        markerobs = 'o',
                        markerSize = 10, 
                        tickRMS = [0.0],
                        tickCOR = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                        tickRMSangle = 115,
                        showlabelsRMS = 'off',
                        titleRMS = 'off', titleOBS = 'Ref')

        axs[j].set_title(f"{'Heavy Wind' if M=='q' else 'Above Threshold' if M=='f' else 'Mean Speed'}",fontsize=18,pad=30)

    plt.suptitle(f"")

    plt.savefig(f"/mnt/beegfs/lcesarini/taylor_{EV}.png",dpi=300,bbox_inches="tight")

