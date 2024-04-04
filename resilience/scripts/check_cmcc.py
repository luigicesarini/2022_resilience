#! /home/lcesarini/miniconda3/envs/colorbar/bin/python

import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
#import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
#from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
#from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

from resilience.utils import *

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

PATH_ORIGINAL_DATA="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT"
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/"

#Loqd FIXED PALETTES
cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()



cmcc_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}ECMWF-ERAINT/CMCC/CPM/pr/CMCC_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
# cmcc_rg=xr.where(cmcc_rg.pr < 0.1, 0, cmcc_rg.pr)
ll1=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{yr}*") for yr in np.arange(2000,2010)]
ll2=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/CMCC/pr/*{yr}*") for yr in np.arange(2000,2010)]

vhr__rg=xr.open_mfdataset(get_unlist(ll2)).load()
# vhr__rg=xr.where(vhr__rg.pr < 0.1, 0, vhr__rg.pr)
sphe_rg=xr.open_mfdataset(get_unlist(ll1)).load()

"""
CHECK ON:
1) Total annual precipiation
2) Mean annual precipitation
3) Average Annula Maxima
"""

# 1) TAP

tap_cmcc = cmcc_rg.groupby(cmcc_rg['time.year']).sum().mean(dim='year')
tap__vhr = vhr__rg.groupby(vhr__rg['time.year']).sum().mean(dim='year')
tap_sphe = sphe_rg.groupby(sphe_rg['time.year']).sum().mean(dim='year')

bias_tap = (tap_cmcc.pr - tap__vhr.pr) / tap__vhr.pr * 100


plot_panel_rotated(
    figsize=(12,12),
    nrow=2,ncol=2,
    list_to_plot=[tap_cmcc.pr,tap__vhr.pr,
                  tap_sphe.pr,bias_tap],
    name_fig=f"panel_comparison_cmccs",
    list_titles=["CPM","VHR","SPHERA","BIAS"],
    levels=[np.arange(100,3500,250),np.arange(100,3500,250),np.arange(100,3500,250),np.arange(-5,25,5)],
    suptitle=f"Panel comparing CMCC CPM model to VHR CMCC on Total annual precipitation",
    # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
    name_metric=["[mm]","[mm]","[mm]","[%]"],
    SET_EXTENT=False,
    cmap=["RdBu","RdBu","RdBu","PuOr"]
)

# 2) MAP 
# map_cmcc = cmcc_rg.groupby(cmcc_rg['time.year']).mean().mean(dim='year')
# map__vhr = vhr__rg.groupby(vhr__rg['time.year']).mean().mean(dim='year')

# bias_map = (map_cmcc.pr - map__vhr.pr) / map__vhr.pr * 100

# plot_panel_rotated(
#     figsize=(24,6),
#     nrow=1,ncol=3,
#     list_to_plot=[map_cmcc.pr,map__vhr.pr,bias_map],
#     name_fig=f"panel_comparison_cmccs_map",
#     list_titles=["CPM","VHR","BIAS"],
#     levels=[np.arange(0,0.5,0.05),np.arange(0,0.5,0.05),np.arange(-5,25,5)],
#     suptitle=f"Panel comparing CMCC CPM model to VHR CMCC on Mean annual precipitation",
#     # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
#     name_metric=["[mm]","[mm]","[%]"],
#     SET_EXTENT=False,
#     cmap=["RdBu","RdBu","RdBu"]
# )


# 3) AMS
ams_cmcc = cmcc_rg.groupby(cmcc_rg['time.year']).max().mean(dim='year')
ams__vhr = vhr__rg.groupby(vhr__rg['time.year']).max().mean(dim='year')
ams_sphe = sphe_rg.groupby(sphe_rg['time.year']).max().mean(dim='year')

bias_ams = (ams_cmcc.pr - ams__vhr.pr) / ams__vhr.pr * 100

plot_panel_rotated(
    figsize=(12,12),
    nrow=2,ncol=2,
    list_to_plot=[ams_cmcc.pr,ams__vhr.pr,
                  ams_sphe.pr,bias_ams],
    name_fig=f"panel_comparison_cmccs_ams",
    list_titles=["CPM","VHR","SPHERA","BIAS"],
    levels=[np.arange(0,41,5),np.arange(0,41,5),np.arange(0,41,5),np.arange(-5,25,5)],
    suptitle=f"Panel comparing CMCC CPM model to VHR CMCC of mean AMS during 2000-2009",
    # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
    name_metric=["[mm]","[mm]","[mm]","[%]"],
    SET_EXTENT=False,
    cmap=["RdBu","RdBu","RdBu","RdBu"]
)
# #CPM-Models
# kit_or=xr.open_dataset(f"{PATH_ORIGINAL_DATA}/KIT/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-KIT-CCLM5-0-14_fpsconv-x2yn2-v1_1hr_200901010030-200912312330.nc").isel(time=12).load()
# #REANALYSIS
# cmcc_pr=xr.open_mfdataset([f"{PATH_CMCC}era5-downscaled-over-italy_VHR-REA_IT_1989_2020_hourly_{year}{month}_hourly.nc" for month in np.arange(1,13) for year in np.arange(2000,2010)]).load()
# cmcc_remap=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/cmcc/remap/reanalysis/pr/*.nc").load()
# #STATIONS
# station=xr.open_mfdataset([f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
# #ENSEMBLE
# ens_f=xr.open_dataset("output/Ensemble_2000_2008_freq.nc")
# ens_i=xr.open_dataset("output/Ensemble_2000_2008_int.nc")
# ens_q=xr.open_dataset("output/Ensemble_2000_2008_q.nc")



# assert cmcc_pr.time.shape == cmcc_remap.time.shape == station.time.shape,\
#       "Time length between CMCC and STATIONS are different"

# cmcc_or_f,cmcc_or_i,cmcc_or_q = compute_metrics_cmcc(get_season(cmcc_pr))
# cmcc_f,cmcc_i,cmcc_q          = compute_metrics(get_season(cmcc_remap))


# plot_panel_rotated(
#     1,3,list_to_plot=[cmcc_or_f,cmcc_or_i,cmcc_or_q.isel(quantile=0)],
#     name_fig="CMCC_original",
#     list_titles=["Freq","Int","Heavy"],
#     levels=[lvl_f,lvl_i,lvl_q],
#     cmap=[cmap_f,cmap_i,cmap_q],
#     SET_EXTENT=False,
#     transform=ccrs.RotatedPole(-168,47)
# )

# plot_panel_rotated(
#     nrow=1,
#     ncol=3,
#     list_to_plot=[cmcc_f,cmcc_i,cmcc_q.isel(quantile=0)],
#     name_fig="CMCC_remapped",
#     list_titles=["Freq","Int","Heavy"],
#     levels=[lvl_f,lvl_i,lvl_q],
#     cmap=[cmap_f,cmap_i,cmap_q],
#     SET_EXTENT=False,
# )

# """ROUTINE TO FIND xval,yvals,xbounds,ybounds for conservative remapping"""
# model="EURO4M-APGD"
# ds=xr.open_dataset(f"/mnt/data/commonData/OBSERVATIONS/{model}/EURO4M-APGD-1971-2008.nc")

# ds=kit_or

# longitude,latitude=[],[]

# for lon in tqdm(np.arange(ds.rlon.shape[0]),total=ds.rlon.shape[0]):
#     for lat in np.arange(ds.rlat.shape[0]):
#         longitude.append(ds.sel(rlon=ds.rlon[lon].item(),rlat=ds.rlat[lat].item()).lon.item())
#         latitude.append( ds.sel(rlon=ds.rlon[lon].item(),rlat=ds.rlat[lat].item()).lat.item())


# list_xy=create_list_coords(longitude,latitude)
# res=ds.rlon[1].lon.item()-ds.X[0].item()
# rot_proj=ccrs.LambertAzimuthalEqualArea(central_longitude=ds['ETRS89-LAEA'].longitude_of_projection_origin,
#                                         central_latitude=ds['ETRS89-LAEA'].latitude_of_projection_origin,
#                                         false_easting=ds['ETRS89-LAEA'].false_easting,
#                                         false_northing=ds['ETRS89-LAEA'].false_northing
#                                         )

# fig,ax=plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})


# ds.pr.isel(time=12)

# for idx in tqdm(np.arange(list_xy.shape[0]), total=list_xy.shape[0]):
#     ur_corner_x,ur_corner_y=convert_coords(list_xy[idx,0]-res/2,list_xy[idx,1]+res/2)
#     ul_corner_x,ul_corner_y=convert_coords(list_xy[idx,0]+res/2,list_xy[idx,1]+res/2)
#     lr_corner_x,lr_corner_y=convert_coords(list_xy[idx,0]-res/2,list_xy[idx,1]-res/2)
#     ll_corner_x,ll_corner_y=convert_coords(list_xy[idx,0]+res/2,list_xy[idx,1]-res/2)

#     with open(f'data/{model}/x_bounds.txt','a') as writer:
#         writer.write(" ".join([ur_corner_x.astype('str'),ul_corner_x.astype('str'),lr_corner_x.astype('str'),ll_corner_x.astype('str')]) + "\n")

#     with open(f'data/{model}/y_bounds.txt','a') as writer:
#         writer.write(" ".join([ur_corner_y.astype('str'),ul_corner_y.astype('str'),lr_corner_y.astype('str'),ll_corner_y.astype('str')]) + "\n")

# list_wgs84=create_list_coords(longitude,latitude)

# idx=0
# for lon,lat in tqdm(zip(longitude,latitude), total=61152):

#     if (idx % 4 == 0) & (idx != 0):
#         with open('data/x_vals.txt','a') as writer:
#             writer.write(f"{lon} " + "\n")
#         with open('data/y_vals.txt','a') as writer:
#             writer.write(f"{lat} " + "\n")
    
#     else:
#         with open('data/x_vals.txt','a') as writer:
#             writer.write(f"{lon} ")
#         with open('data/y_vals.txt','a') as writer:
#             writer.write(f"{lat} ")

#     idx+=1

# os.getcwd()

# rea_cmcc_q=xr.open_dataset("output/JJA/CMCC_VHR_q.nc")

# cmcc_or = xr.open_mfdataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CMCC/CPM/pr/*nc")

# cmcc_jja=get_season(cmcc_or,"JJA")
# cmcc_jja=cmcc_jja.load()
# cmcc_jja_q=cmcc_jja.pr.quantile(q=0.999,dim='time',keep_attrs=True)

# min_lon,max_lon=rea_cmcc_q.lon.min().item(),rea_cmcc_q.lon.max().item()
# min_lat,max_lat=rea_cmcc_q.lat.min().item(),rea_cmcc_q.lat.max().item()

# cmcc_jja_q=(cmcc_jja_q * 3600).expand_dims({"rlon":cmcc_or.rlon,"rlat":cmcc_or.rlat}) 

# cmcc_jja_q.swap_dims({'x':'rlon','y':'rlat'})
# cmcc_jja_q_north=cmcc_jja_q.where((cmcc_jja_q.lon >= min_lon) & (cmcc_jja_q.lon <= max_lon) &
#                  (cmcc_jja_q.lat >= min_lat) & (cmcc_jja_q.lat <= max_lat) ,drop=False)




# ax=plt.axes(projection=ccrs.RotatedPole(pole_longitude=-162,pole_latitude=39.25))

# (cmcc_jja_q_north * 3600).plot.pcolormesh(
#     ax=ax,cmap=cmap_q,levels=np.arange(2,19,2)
# )
# plt.savefig(f"/home/lcesarini/2022_resilience/figures/quantile_cpm_cmcc.png")

# plot_panel_rotated(
#     figsize=(8,8),
#     nrow=1,ncol=1,
#     list_to_plot=[cmcc_jja_q_north * 3600],
#     name_fig=f"quantile_cpm_cmcc",
#     list_titles=["Heavy prec CMCC CPM"],
#     levels=[np.arange(2,19,2)],
#     suptitle=f"JJA",
#     name_metric=["[m/h]"],
#     SET_EXTENT=True,
#     cmap=[cmap_q],
#     proj=ccrs.RotatedPole(pole_longitude=-162,pole_latitude=39.25)
# )            