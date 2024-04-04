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
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

from utils import *

import warnings
warnings.filterwarnings('ignore')
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"

os.chdir("/home/lcesarini/2022_resilience/")

ds_rem=xr.open_dataset(f"{PATH_COMMON_DATA}//MOHC/CPM/pr/MOHC_ECMWF-ERAINT_200104010030-200104302330.nc")

meta = pd.read_csv("stations/meta_stations_eval.csv")


from scripts.utils import *

ele=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/datiDallan/CPM_ETH_Italy_2000-2009_pr_hour.nc")
eth__rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ETH/CPM/pr/ETH_ECMWF-ERAINT_*.nc").load()

ij=pd.read_table("griglia_ele.txt",header=None)
ij.columns=["name","i","j"]
ij.i=ij.i-1
ij.j=ij.j-1


STATIONS = ["VE_0088","VE_0100","TN_0147","AA_4740","VE_0239","VE_0011"]

idx_nat=pd.read_csv("indices_nat.txt",header=None,sep=" ")
print(idx_nat[idx_nat[0]=="VE_0011"])
## COORDS STATION LON=11.9045451571426,LAT=46.4293691505874
gds=gpd.GeoDataFrame(meta,
                        geometry=gpd.points_from_xy(meta["lon"],
                                                    meta["lat"], 
                        crs="EPSG:4326"))




print(ds.lon[1]-ds.lon[0])
gds=gds[gds.name=="VE_0011"]



st_VE_0011=pd.read_csv("/home/lcesarini/VE_0011_station.csv")
metrics_eth=[]
for idx,name_st in enumerate(STATIONS):
    st_old=pd.read_csv(f"/home/lcesarini/{name_st}.csv")
    i_lon=idx_nat[idx_nat[0]==name_st][1].item()
    i_lat=idx_nat[idx_nat[0]==name_st][2].item()

    metrics_eth.append({
    "name":name_st,
    "mean":eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.mean() - st_old.pr.mean(),
    "std" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.std() - st_old.pr.std(),
    "min" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.min() - st_old.pr.min(),
    "q25":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.25]) - np.quantile(st_old.pr,q=[0.25])).item(),
    "q50":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.5]) - np.quantile(st_old.pr,q=[0.5])).item(),
    "q75":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.75]) - np.quantile(st_old.pr,q=[0.75])).item(),
    "max" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.max() - st_old.pr.max(),
    })

print(pd.DataFrame.from_dict(metrics_eth).transpose())





{
"mean":eth__rg.isel(lon=197,lat=116).pr.values.mean() - st_VE_0011.pr.mean(),
"std":eth__rg.isel(lon=197,lat=116).pr.values.min() - st_VE_0011.pr.min(),
"min":eth__rg.isel(lon=197,lat=116).pr.values.min() - st_VE_0011_2.pr.min(),
"q25,q50,q75":np.quantile(eth__rg.isel(lon=197,lat=116).pr.values,q=[0.25,0.5,0.75]) - np.quantile(st_VE_0011.pr,q=[0.25,0.5,0.75]),
"max":eth__rg.isel(lon=197,lat=116).pr.values.max() - st_VE_0011.pr.max(),
}


ij[ij.name == "VE_0011"]
B=ele.isel(rlon=196,rlat=141).pr

for idx,name_st in enumerate(STATIONS):
    gds_st=gds[gds.name==name_st]
    lon_grid=ds_rem.sel(lon=gds_st.lon.item(),lat=gds_st.lat.item(),method="nearest").lon.item()
    lat_grid=ds_rem.sel(lon=gds_st.lon.item(),lat=gds_st.lat.item(),method="nearest").lat.item()

    A=eth__rg.sel(lon=lon_grid,lat=lat_grid,method='nearest').pr

    pd.DataFrame(pd.to_datetime(A.time.values)).\
        to_csv(f"/home/lcesarini/2022_resilience/data/check_nathalia/dates_{name_st}_station.csv",
                index=0,
                header = ["date"]
                )
    pd.DataFrame(A.values).\
        to_csv(f"/home/lcesarini/2022_resilience/data/check_nathalia/pr_{name_st}_station.csv",
                index=0,
                header = ["pr"]
                )
    
# eth__rg.isel(lon=197,lat=116).pr.values.max() - st_VE_0011_2.pr.max() 
metrics_st=[]
for idx,name_st in enumerate(STATIONS):
    st_new=pd.read_csv(f"/home/lcesarini/2022_resilience/data/check_nathalia/{name_st}_station.csv")
    i_lon=idx_nat[idx_nat[0]==name_st][1].item()
    i_lat=idx_nat[idx_nat[0]==name_st][2].item()

    metrics_st.append({
    "name":name_st,
    "mean":eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.mean() - st_new.pr.mean(),
    "std" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.std() - st_new.pr.std(),
    "min" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.min() - st_new.pr.min(),
    "q25":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.25]) - np.quantile(st_new.pr,q=[0.25])).item(),
    "q50":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.5]) - np.quantile(st_new.pr,q=[0.5])).item(),
    "q75":(np.quantile(eth__rg.isel(lon=i_lon,lat=i_lat).pr.values,q=[0.75]) - np.quantile(st_new.pr,q=[0.75])).item(),
    "max" :eth__rg.isel(lon=i_lon,lat=i_lat).pr.values.max() - st_new.pr.max(),
    })

print(pd.DataFrame.from_dict(metrics_st).transpose())

print(
f"""
mean:{eth__rg.isel(lon=197,lat=116).pr.values.mean() - st_VE_0011.pr.mean()}
std:{eth__rg.isel(lon=197,lat=116).pr.values.min() - st_VE_0011.pr.min()}
min:{eth__rg.isel(lon=197,lat=116).pr.values.min() - st_VE_0011_2.pr.min()}
q25,q50,q75:{np.quantile(eth__rg.isel(lon=197,lat=116).pr.values,q=[0.25,0.5,0.75]) - np.quantile(st_VE_0011.pr,q=[0.25,0.5,0.75])}
max:{eth__rg.isel(lon=197,lat=116).pr.values.max() - st_VE_0011.pr.max() }
"""
)


if __name__=="__main__":

    gds=gpd.GeoDataFrame(meta,
                         geometry=gpd.points_from_xy(meta["lon"],
                                                     meta["lat"], 
                         crs="EPSG:4326"))

    
    print(ds.lon[1]-ds.lon[0])
    gds=gds[gds.name=="VE_0011"]
    for idx,(lon,lat) in enumerate(zip(gds.lon,gds.lat)):
        print(gds.name,lon,lat)

        lon_grid,lat_grid=ds_rem.sel(lon=lon,lat=lat,method="nearest").lon.item(),ds_rem.sel(lon=lon,lat=lat,method="nearest").lat.item()

        i=np.argwhere(ds_rem.lon.values == lon_grid).item() 
        j=np.argwhere(ds_rem.lat.values == lat_grid).item()


        with open("indices_nat.txt","a") as writer:
            writer.write(f"{gds.name[idx]} {i} {j}\n")
        # pd.DataFrame([gds.name[idx],i,j],)
        # ds_rem.isel(lon=i,lat=j)

