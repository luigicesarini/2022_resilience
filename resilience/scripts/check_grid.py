#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import metview as mv
import seaborn as sns
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from shapely.geometry import mapping
from cartopy import feature as cfeature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

# from scripts.utils import *


PATH_ORIGINAL="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT"
PATH_REMAP="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
import sys

model=sys.argv[1]
# model="HCLIMcom"

mdl_pr_or=xr.open_dataset(glob(f"{PATH_ORIGINAL}/{model}/CPM/pr/*2000*")[0]).isel(time=12).load()
mdl_pr_re=xr.open_dataset(glob(f"{PATH_REMAP}/{model}/CPM/pr/*2000*")[0]).isel(time=12).load()
mdl_u_ori=xr.open_dataset(glob(f"{PATH_ORIGINAL}/{model}/CPM/uas/*2000*")[0]).isel(time=12).load()
mdl_v_ori=xr.open_dataset(glob(f"{PATH_ORIGINAL}/{model}/CPM/vas/*2000*")[0]).isel(time=12).load()

xr.open_dataset(glob(f"{PATH_ORIGINAL}/HCLIMcom/CPM/vas/*2000*")[0]).isel(time=12).Lambert_Conformal.attrs['proj4']  

name_lon_rot="rlon" if "rlon" in list(mdl_pr_or.coords) else "x"
name_lat_rot="rlat" if "rlat" in list(mdl_pr_or.coords) else "y"

name_lon="lon" if "lon" in list(mdl_pr_or.coords) else "longitude"
name_lat="lat" if "lat" in list(mdl_pr_or.coords) else "latitude"

if hasattr(mdl_pr_or,"rotated_pole"):   
    CRS_ORIG_PR=ccrs.RotatedPole(pole_longitude=mdl_pr_or.rotated_pole.attrs['grid_north_pole_longitude'],
                                 pole_latitude =mdl_pr_or.rotated_pole.attrs['grid_north_pole_latitude'])
elif hasattr(mdl_pr_or,"Lambert_Conformal"):
    CRS_ORIG_PR=ccrs.LambertConformal(central_longitude = mdl_pr_or.Lambert_Conformal.attrs["longitude_of_central_meridian"],
                                      central_latitude  = mdl_pr_or.Lambert_Conformal.attrs["latitude_of_projection_origin"],
                                      false_easting     = mdl_pr_or.Lambert_Conformal.attrs["false_easting"],
                                      false_northing    = mdl_pr_or.Lambert_Conformal.attrs["false_northing"])
elif hasattr(mdl_pr_or,"rotated_latitude_longitude"):
    CRS_ORIG_PR=ccrs.RotatedPole(pole_longitude=mdl_pr_or.rotated_latitude_longitude.attrs['grid_north_pole_longitude'],
                                 pole_latitude =mdl_pr_or.rotated_latitude_longitude.attrs['grid_north_pole_latitude'])
elif hasattr(mdl_pr_or,"crs"):
    CRS_ORIG_PR=ccrs.LambertConformal(central_longitude = mdl_pr_or.crs.attrs["longitude_of_central_meridian"],
                                    central_latitude    = mdl_pr_or.crs.attrs["latitude_of_projection_origin"],
                                    false_easting       = mdl_pr_or.crs.attrs["false_easting"],
                                    false_northing      = mdl_pr_or.crs.attrs["false_northing"])


if hasattr(mdl_u_ori,"rotated_pole"):   
    CRS_ORIG_U=ccrs.RotatedPole(pole_longitude=mdl_u_ori.rotated_pole.attrs['grid_north_pole_longitude'],
                                pole_latitude =mdl_u_ori.rotated_pole.attrs['grid_north_pole_latitude'])
elif hasattr(mdl_u_ori,"Lambert_Conformal"):
    CRS_ORIG_U=ccrs.LambertConformal(central_longitude  = mdl_u_ori.Lambert_Conformal.attrs["longitude_of_central_meridian"],
                                      central_latitude  = mdl_u_ori.Lambert_Conformal.attrs["latitude_of_projection_origin"],
                                      false_easting     = mdl_u_ori.Lambert_Conformal.attrs["false_easting"],
                                      false_northing    = mdl_u_ori.Lambert_Conformal.attrs["false_northing"])
elif hasattr(mdl_u_ori,"rotated_latitude_longitude"):
    CRS_ORIG_U=ccrs.RotatedPole(pole_longitude=mdl_u_ori.rotated_latitude_longitude.attrs['grid_north_pole_longitude'],
                                 pole_latitude =mdl_u_ori.rotated_latitude_longitude.attrs['grid_north_pole_latitude'])
elif hasattr(mdl_u_ori,"crs"):
    CRS_ORIG_U=ccrs.LambertConformal(central_longitude = mdl_u_ori.crs.attrs["longitude_of_central_meridian"],
                                     central_latitude  = mdl_u_ori.crs.attrs["latitude_of_projection_origin"],
                                     false_easting     = mdl_u_ori.crs.attrs["false_easting"],
                                     false_northing    = mdl_u_ori.crs.attrs["false_northing"])

if CRS_ORIG_U == CRS_ORIG_PR: print(f"{model} has SAME CRS")

if np.all(mdl_pr_or[name_lon_rot].values != mdl_u_ori[name_lon_rot].values): 
    print(f"{model} uses different grids for pr and u")
else:
    print(f"{model} uses same grids for pr and u")
if np.all(mdl_v_ori[name_lon_rot].values != mdl_u_ori[name_lon_rot].values): 
    print(f"{model} uses different grids for v and u")
else:
    print(f"{model} uses same grids for v and u")

mdl_pr_or=mdl_pr_or.where((mdl_pr_or[name_lon] > 10.38) & (mdl_pr_or[name_lon] < 13.1) & (mdl_pr_or[name_lat] > 44.7) & (mdl_pr_or[name_lat] < 47.1), drop=True)
mdl_pr_re=mdl_pr_re.where((mdl_pr_re['lon'] > 10.38) & (mdl_pr_re['lon'] < 13.1) & (mdl_pr_re['lat'] > 44.7) & (mdl_pr_re['lat'] < 47.1), drop=True)
mdl_u_ori=mdl_u_ori.where((mdl_u_ori[name_lon] > 10.38) & (mdl_u_ori[name_lon] < 13.1) & (mdl_u_ori[name_lat] > 44.7) & (mdl_u_ori[name_lat] < 47.1), drop=True)
mdl_v_ori=mdl_v_ori.where((mdl_v_ori[name_lon] > 10.38) & (mdl_v_ori[name_lon] < 13.1) & (mdl_v_ori[name_lat] > 44.7) & (mdl_v_ori[name_lat] < 47.1), drop=True)

lon_min,lon_max,lat_min,lat_max=10.468,10.635,44.715,44.880

mdl_pr_or_tri = mdl_pr_or.where((mdl_pr_or[name_lon] > lon_min) & (mdl_pr_or[name_lon] < lon_max) &\
		                        (mdl_pr_or[name_lat] > lat_min) & (mdl_pr_or[name_lat] < lat_max), drop=True) 

mdl_u_ori_tri = mdl_u_ori.where((mdl_u_ori[name_lon] > lon_min) & (mdl_u_ori[name_lon] < lon_max) &\
		                        (mdl_u_ori[name_lat] > lat_min) & (mdl_u_ori[name_lat] < lat_max), drop=True) 

mdl_v_ori_tri = mdl_v_ori.where((mdl_v_ori[name_lon] > lon_min) & (mdl_v_ori[name_lon] < lon_max) &\
		                        (mdl_v_ori[name_lat] > lat_min) & (mdl_v_ori[name_lat] < lat_max), drop=True) 




ax = plt.subplot(projection=ccrs.PlateCarree())

##PRECIPITAZIONE ORiGINALE
ax.scatter(create_list_coords(mdl_pr_or_tri[name_lon_rot].values,mdl_pr_or_tri[name_lat_rot].values)[:,0],
	   	   create_list_coords(mdl_pr_or_tri[name_lon_rot].values,mdl_pr_or_tri[name_lat_rot].values)[:,1],
		   transform=CRS_ORIG_PR,color='black', label="Center cell pr orig",alpha=0.55)


mdl_pr_or_tri["pr" if "pr" in list(mdl_pr_or_tri.data_vars) else "precipitation_flux"].plot.pcolormesh(color='black',alpha=0.15,
			    			add_colorbar=False,
						    cmap="Blues",
						    levels=1,
							transform=CRS_ORIG_PR)
#VENTO ORIGINALE
#COMPONENTE U
ax.scatter(create_list_coords(mdl_u_ori_tri[name_lon_rot].values,mdl_u_ori_tri[name_lat_rot].values)[:,0],
	   	   create_list_coords(mdl_u_ori_tri[name_lon_rot].values,mdl_u_ori_tri[name_lat_rot].values)[:,1],
		   transform=CRS_ORIG_U,color='red', alpha=0.15,label="Center cell u orig")

# mdl_u_ori_tri.uas.plot.pcolormesh(color='red',alpha=0.15,
# 							   add_colorbar=False,
# 							   levels=1,
# 							   transform=CRS_ORIG_U)
#COMPONENTE V
ax.scatter(create_list_coords(mdl_v_ori_tri[name_lon_rot].values,mdl_v_ori_tri[name_lat_rot].values)[:,0],
	   	   create_list_coords(mdl_v_ori_tri[name_lon_rot].values,mdl_v_ori_tri[name_lat_rot].values)[:,1],
		   transform=CRS_ORIG_U,color='blue', alpha=0.15,label="Center cell v orig")

mdl_v_ori_tri.vas.plot.pcolormesh(color='blue',alpha=0.15,
							   add_colorbar=False,
							   levels=1,
							   transform=CRS_ORIG_U)

# ax.set_extent([10.4680,10.535,44.715,44.780])
ax.set_extent([10.45,10.55,44.7,44.82])
# or_slice.pr.plot.scatter(ax=ax,transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))
# eth_re_tri.pr.plot.scatter(ax=ax,c='red')
##PRECIPITAZIONE RIMAPPATA
ax.scatter(create_list_coords(mdl_pr_re.lon,mdl_pr_re.lat)[:,0],
	   	   create_list_coords(mdl_pr_re.lon,mdl_pr_re.lat)[:,1],
	       color='green',label="center cell pr conservative")
gl=ax.gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(create_list_coords(mdl_pr_re.lon,mdl_pr_re.lat)[:,0] + (0.02749919891357422/2))
gl.ylocator = mticker.FixedLocator(create_list_coords(mdl_pr_re.lon,mdl_pr_re.lat)[:,1] + (0.02749919891357422/2))
# gl2=ax.gridlines(draw_labels=True,linewidth=.5, color='indianred', alpha=0.5, linestyle='--')
# gl2.xlocator = mticker.FixedLocator(l_rx_or_pr  + (0.02749919891357422/2))
# gl2.ylocator = mticker.FixedLocator(l_ry_or_pr + (0.02749919891357422/2))
plt.title(f"Model {model}")
plt.legend(bbox_to_anchor =(1.15, 0.01),edgecolor='black',shadow=True,ncol=2)
plt.savefig(f"figures/scatter_grid_wind_{model}.png")
plt.close()

