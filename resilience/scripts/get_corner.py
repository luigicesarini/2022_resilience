#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience")

import argparse
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature

from utils import *

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

if os.getcwd() == "/home/lcesarini/2022_resilience":
    from resilience.utils import get_unlist,get_palettes,get_levels,create_list_coords


shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

PATH_ORIGINAL_DATA="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT"
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
PATH_CMCC="/mnt/data/RESTRICTED/CARIPARO/cmcc/reanalysis/precipitation_amount/"

#Loqd FIXED PALETTES
cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

#CPM-Models
kit_or=xr.open_dataset(f"{PATH_ORIGINAL_DATA}/KIT/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CLMcom-KIT-CCLM5-0-14_fpsconv-x2yn2-v1_1hr_200901010030-200912312330.nc").isel(time=12).load()
#REANALYSIS
cmcc_pr=xr.open_mfdataset([f"{PATH_CMCC}era5-downscaled-over-italy_VHR-REA_IT_1989_2020_hourly_{year}{month}_hourly.nc" for month in np.arange(1,13) for year in np.arange(2000,2010)]).load()
cmcc_remap=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/cmcc/remap/reanalysis/pr/*.nc").load()
#STATIONS
station=xr.open_mfdataset([f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
#ENSEMBLE
ens_f=xr.open_dataset("output/Ensemble_2000_2008_freq.nc")
ens_i=xr.open_dataset("output/Ensemble_2000_2008_int.nc")
ens_q=xr.open_dataset("output/Ensemble_2000_2008_q.nc")



assert cmcc_pr.time.shape == cmcc_remap.time.shape == station.time.shape,\
      "Time length between CMCC and STATIONS are different"

cmcc_or_f,cmcc_or_i,cmcc_or_q = compute_metrics_cmcc(get_season(cmcc_pr))
cmcc_f,cmcc_i,cmcc_q          = compute_metrics(get_season(cmcc_remap))


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

"""ROUTINE TO FIND xval,yvals,xbounds,ybounds for conservative remapping"""
model="EURO4M-APGD"
ds=xr.open_dataset(f"/mnt/data/lcesarini/gripho-v1_1h_TSmin30pct_2001-2016_cut3km.nc")

ds=kit_or

longitude,latitude=[],[]

name_rot_lon="x" if "x" in list(ds.coords) else "rlon"
name_rot_lat="y" if "y" in list(ds.coords) else "rlat"


if "x" in list(ds.coords):
    for lon in tqdm(np.arange(ds[name_rot_lon].shape[0]),total=ds[name_rot_lon].shape[0]):
        for lat in np.arange(ds[name_rot_lat].shape[0]):
            #We take the meters in the LAEA CRS
            longitude.append(ds.sel(x=ds[name_rot_lon][lon].item(),
                                    y=ds[name_rot_lat][lat].item()).x.item())
            latitude.append( ds.sel(x=ds[name_rot_lon][lon].item(),
                                    y=ds[name_rot_lat][lat].item()).y.item())
else:    
    for lon in tqdm(np.arange(ds[name_rot_lon].shape[0]),total=ds[name_rot_lon].shape[0]):
        for lat in np.arange(ds[name_rot_lat].shape[0]):
            longitude.append(ds.sel(rlon=ds[name_rot_lon][lon].item(),
                                    rlat=ds[name_rot_lat][lat].item()).lon.item())
            latitude.append( ds.sel(rlon=ds[name_rot_lon][lon].item(),
                                    rlat=ds[name_rot_lat][lat].item()).lat.item())


list_xy=create_list_coords(longitude,latitude)

if "x" in list(ds.coords):
    res=np.abs(latitude[1]-latitude[0]).item()#np.abs(list_xy[1,0]-list_xy[0,0]).item()
    rot_proj=ccrs.LambertAzimuthalEqualArea(central_longitude=ds.crs.longitude_of_central_meridian,
                                        central_latitude=ds.crs.latitude_of_projection_origin,
                                        false_easting=ds.crs.false_easting,
                                        false_northing=ds.crs.false_northing
                                        )
else:
    res=ds.rlon[1].lon.item()-ds.X[0].item()
    rot_proj=ccrs.LambertAzimuthalEqualArea(central_longitude=ds['ETRS89-LAEA'].longitude_of_projection_origin,
                                            central_latitude=ds['ETRS89-LAEA'].latitude_of_projection_origin,
                                            false_easting=ds['ETRS89-LAEA'].false_easting,
                                            false_northing=ds['ETRS89-LAEA'].false_northing
                                            )
degree_to_meters(res)
fig,ax=plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})

res=3000
ds.pr.isel(time=12)

model="GRIPHO"
list_xy=np.concatenate([np.array(longitude).reshape(-1,1),np.array(latitude).reshape(-1,1)],axis=1)
for idx in tqdm(np.arange(list_xy.shape[0]), total=list_xy.shape[0]):
    ur_corner_x,ur_corner_y=convert_coords(list_xy[idx,0]-res/2,list_xy[idx,1]+res/2,rot=rot_proj)
    ul_corner_x,ul_corner_y=convert_coords(list_xy[idx,0]+res/2,list_xy[idx,1]+res/2,rot=rot_proj)
    lr_corner_x,lr_corner_y=convert_coords(list_xy[idx,0]-res/2,list_xy[idx,1]-res/2,rot=rot_proj)
    ll_corner_x,ll_corner_y=convert_coords(list_xy[idx,0]+res/2,list_xy[idx,1]-res/2,rot=rot_proj)

    with open(f'data/{model}/x_bounds.txt','a') as writer:
        writer.write(" ".join([ur_corner_x.astype('str'),ul_corner_x.astype('str'),lr_corner_x.astype('str'),ll_corner_x.astype('str')]) + "\n")

    with open(f'data/{model}/y_bounds.txt','a') as writer:
        writer.write(" ".join([ur_corner_y.astype('str'),ul_corner_y.astype('str'),lr_corner_y.astype('str'),ll_corner_y.astype('str')]) + "\n")

list_wgs84=create_list_coords(longitude,latitude)

idx=0
for lon,lat in tqdm(zip(longitude,latitude), total=61152):

    if (idx % 4 == 0) & (idx != 0):
        with open(f'data/{model}/x_vals.txt','a') as writer:
            writer.write(f"{lon} " + "\n")
        with open(f'data/{model}/y_vals.txt','a') as writer:
            writer.write(f"{lat} " + "\n")
    
    else:
        with open(f'data/{model}/x_vals.txt','a') as writer:
            writer.write(f"{lon} ")
        with open(f'data/{model}/y_vals.txt','a') as writer:
            writer.write(f"{lat} ")

    idx+=1

