#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import json
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
from datetime import datetime,timedelta
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from shapely.geometry import box, mapping
from scipy.interpolate import griddata
from geocube.rasterize import (rasterize_points_griddata, 
                               rasterize_points_radial)
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                            r2_score,mean_absolute_percentage_error)


from utils import *

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience")

# empty_common=xr.load_dataset("data/empty_common_grid.nc")
mask_st=xr.open_dataset("data/mask_stations_nan_common.nc")

stations_path = glob("stations/text/*.csv")
dates_path    = glob("data/dates/*.csv")
assert len(dates_path)==len(stations_path), "Number of station files != number of dates files"

df=[]
for path_d,path_s in tqdm(zip(dates_path,stations_path),total=len(dates_path)):
    date,prec=pd.read_csv(path_d,parse_dates=[0],date_parser=pd.to_datetime),pd.read_csv(path_s)

    df.append(pd.concat([date,prec],axis=1))

range_dates=pd.date_range(start="1981-01-01 00:00:00",end="2020-12-31 22:00:00",freq="H")
meta_stations=pd.read_csv("meta_station_updated_col.csv",index_col=0)

#assert len(range_dates) == len(df[0]['date']), "Unable to check length date on single df"

df_init=pd.DataFrame(data=range_dates,columns=['date'])
print(df_init.head())
print(df_init.dtypes)

for _ in tqdm(df,total=len(df)):
    df_init=pd.merge(df_init,_,how='left',on='date') 


df_init_t=df_init.transpose().reset_index()

df_init_t.columns=["name"] + [colnames for colnames in df_init_t.iloc[0,1:]] 
df_init_t.drop(df_init_t.index[0],axis=0,inplace=True)
df_init_t.head()


merged_df=pd.merge(df_init_t,meta_stations,on='name')   

gdf=gpd.GeoDataFrame(data=merged_df,geometry=gpd.points_from_xy(merged_df.lon, merged_df.lat), crs=4326)
# gdf.columns=["name"]+[col.to_datetime64() for col in gdf.columns[1:350640]]+[col for col in gdf.columns[350640:]]

gdf.columns=["name"]+[f"t_{i}" for i,col in enumerate(gdf.columns[1:350640])]+[col for col in gdf.columns[350640:]]

timestamps=[col.to_datetime64() for col in df_init_t.columns[1:350640]]


def crop_to_extent(xr,xmin=10.38,xmax=13.1,ymin=44.7,ymax=47.1):
    """
    Functions that select inside a given extent

    Parameters
    ----------
    xr : xarrayDataset, 
        xarray dataset to crop
    
    xmin,xmax,ymin,ymax: coordinates of the extent desired.    
    
    Returns
    -------
    
    cropped_ds: xarray dataset croppped

    Examples
    --------
    """
    xr_crop=xr.where((xr.lon > xmin) & (xr.lon < xmax) &\
                     (xr.lat > ymin) & (xr.lat < ymax), 
                     drop=True)

    return xr_crop

year=2000
  
all_st=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/pr_st_*.nc").load()

q999_st=all_st.quantile(q=[0.999],dim='time')


shp_triveneto = gpd.read_file("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

fig,ax = plt.subplots(1,subplot_kw={"projection":ccrs.PlateCarree()},
                    figsize=(12,16))

xr.where(np.isnan(q999_st),0,q999_st).pr.isel(quantile=0).plot.pcolormesh(ax=ax,
                                            cbar_kwargs={"shrink":0.25},
                                            cmap=plt.cm.rainbow,extend="both",
                                            levels=9)
shp_triveneto.boundary.plot(ax=ax, edgecolor="red")

ax.set_extent([10.4,13.1,44.6,47.1])
plt.savefig("ss.png")
plt.close()

