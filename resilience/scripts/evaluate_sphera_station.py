#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import time
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
# import xarray.ufuncs as xu 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from timeit import default_timer as timer
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 
PATH_CSV="/home/lcesarini/2022_resilience/csv/"
from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth
    

def compute_freq(ds):

    freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1).sum(dim='time') / np.max(ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape)

    return freq

def compute_int(ds):

    MeanIntensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                            ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                            np.nan).mean(dim='time', skipna=True)

    return MeanIntensity

def compute_wet_hours(ds):
    wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                                ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                np.nan)

    pXX  = wet_ds.quantile(q=[0.99,0.999], dim = 'time',skipna=True) 

    return pXX



""""""

shp_triveneto = gpd.read_file("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

sta_val=xr.open_mfdataset(f"{PATH_COMMON_DATA}/stations/pr/pr_st_*.nc").load()

# sphera = [xr.open_mfdataset(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{yr}*") for yr in np.arange(1996,2020)]

# sphera_ds=xr.concat(sphera,dim="time")
# sphera_ds=sphera_ds.rename({'longitude':'lon','latitude':'lat'})
# sphera_ds['lon'] = mask['lon'].values
# sphera_ds['lat'] = mask['lat'].values

# sph_tri=get_triveneto(sphera_ds,sta_val)
# sph_tri=sph_tri.load()

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI','CMCC','KIT']
seasons=['DJF','JJA']
ll_ds=[]
time_ds=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/CNRM/CPM/pr/*.nc")
arr_time=time_ds.sel(time=time_ds['time.year'].isin(np.arange(2000,2010))).time.values 
for nm in tqdm(name_models):
    if nm == 'ETH':
        cpm=fix_eth()
    else:
        cpm = xr.open_mfdataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{nm}/CPM/pr/*.nc").load()
    
    cpm=cpm.sel(time=cpm['time.year'].isin(np.arange(2000,2010)))
    cpm_tri=get_triveneto(cpm.pr,sta_val)

    cpm_tri['time']=arr_time
    ll_ds.append(cpm_tri)

cpm_tri=xr.concat(ll_ds,name_models).rename({'concat_dim':'name'}).to_dataset(name='pr')


cpm_tri=get_triveneto(xr.load_dataset(f"/home/lcesarini/2022_resilience/output/DJF/ENSEMBLE_q.nc").pr,sta_val)
nc_to_csv(cpm_tri, name=f"q999_ensemble_all_DJF",M="q")

# sta_sph=sta_val.sel(time=sta_val.time.isin(sph_tri.time))

"""
GET frequency of wet hours
"""
# for seas in seasons:
    # sta_sph_q=get_season(sta_sph,seas)
    # sph_tri_q=get_season(sph_tri,seas)
    # cpm_tri_q=get_season(cpm_tri.mean(dim='name'),seas)

    # nc_to_csv(compute_freq(sta_sph_q), name=f"freq_station_all_{seas}",M="f")
    # nc_to_csv(compute_freq(sph_tri_q), name=f"freq_sphera_all_{seas}",M="f")
    # nc_to_csv(compute_freq(cpm_tri_q).load(), name=f"freq_ensemble_all_{seas}",M="f")

"""
GET mean intensity of wet hours
"""
# for seas in seasons:
#     sta_sph_q=get_season(sta_sph,seas)
#     sph_tri_q=get_season(sph_tri,seas)

#     nc_to_csv(compute_int(sta_sph_q), name=f"mean_intensity_station_all_{seas}",M="i")
#     nc_to_csv(compute_int(sph_tri_q), name=f"mean_intensity_sphera_all_{seas}",M="i")
"""
GET heavy precipitation on wethours wet hours
"""
# for seas in seasons:
#     sta_sph_q=get_season(sta_sph,seas)
#     sph_tri_q=get_season(sph_tri,seas)

#     wh_sta=compute_wet_hours(sta_sph_q)
#     wh_sph=compute_wet_hours(sph_tri_q)

#     nc_to_csv(wh_sta.isel(quantile=0), name=f"heavy_99_on_WH_station_all_{seas}",M="q")
#     nc_to_csv(wh_sph.isel(quantile=0), name=f"heavy_99_on_WH_sphera_all_{seas}", M="q")

#     nc_to_csv(wh_sta.isel(quantile=1), name=f"heavy_999_on_WH_station_all_{seas}",M="q")
#     nc_to_csv(wh_sph.isel(quantile=1), name=f"heavy_999_on_WH_sphera_all_{seas}", M="q")

"""
GET QUANTILE: 99th & 99.9th
"""

for seas in seasons:
    # sta_sph_q=get_season(sta_sph,seas).quantile(dim='time',q=[0.99,0.999])
    # sph_tri_q=get_season(sph_tri,seas).quantile(dim='time',q=[0.99,0.999])
    cpm_tri_q=get_season(cpm_tri.mean(dim='name'),seas).quantile(dim='time',q=[0.99,0.999])

    # nc_to_csv(sta_sph_q.sel(quantile=0.99), name=f"q99_station_all_{seas}",M="q")
    # nc_to_csv(sta_sph_q.sel(quantile=0.999),name=f"q999_station_all_{seas}",M="q")

    # nc_to_csv(sph_tri_q.sel(quantile=0.99), name=f"q99_sphera_all_{seas}",M="q")
    # nc_to_csv(sph_tri_q.sel(quantile=0.999),name=f"q999_sphera_all_{seas}",M="q")

    nc_to_csv(cpm_tri_q.sel(quantile=0.99), name=f"q99_ensemble_all_{seas}",M="q")
    nc_to_csv(cpm_tri_q.sel(quantile=0.999),name=f"q999_ensemble_all_{seas}",M="q")

# for seas in seasons:
#     sta_sph_q=get_season(sta_sph,seas)
#     sph_tri_q=get_season(sph_tri,seas)

#     avg_pr_sta=sta_sph_q.resample(time="1D").sum().mean(dim='time')
#     avg_pr_sph=sph_tri_q.resample(time="1D").sum().mean(dim='time')

#     nc_to_csv(avg_pr_sta, name=f"{seas}_mean_prec_station_all",M="pr")
#     nc_to_csv(avg_pr_sph, name=f"{seas}_mean_prec_sphera_all",M="pr")


# for seas in seasons:

#     df_sta_99=pd.read_csv(f"{PATH_CSV}q99_station_all_{seas}.csv")
#     df_sph_99=pd.read_csv(f"{PATH_CSV}q99_sphera_all_{seas}.csv")


#     df=pd.concat([df_sta_99.q,df_sph_99.q],axis=1)
#     df.columns=["Station","SPHERA"]

#     sns.boxplot(data=df.melt(),x="variable",y="value")
#     plt.title(f"99th percentile su tutto il periodo per SPHERA e stazioni on {seas}")
#     plt.savefig(f"/home/lcesarini/2022_resilience/99th_percentile_{seas}.png")
#     plt.close()

#     df_sta_999=pd.read_csv(f"{PATH_CSV}q999_station_all_{seas}.csv")
#     df_sph_999=pd.read_csv(f"{PATH_CSV}q999_sphera_all_{seas}.csv")


#     df=pd.concat([df_sta_999.q,df_sph_999.q],axis=1)
#     df.columns=["Station","SPHERA"]

#     sns.boxplot(data=df.melt(),x="variable",y="value")
#     plt.title(f"999th percentile su tutto il periodo per SPHERA e stazioni on {seas}")
#     plt.savefig(f"/home/lcesarini/2022_resilience/999th_percentile_{seas}.png")
#     plt.close()


"""
bilinear remap COSMO
"""
# import subprocess as sb
# ll=sb.check_output("ls /mnt/data/lcesarini/COSMO/REA_2/TOT_PRECIP/*.grb",shell=True).decode("utf-8").splitlines()

# for i in tqdm(ll):
#     bn=os.path.basename(i)
#     sb.run(f"cdo remapbil,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt {i} /mnt/data/lcesarini/tmp/{bn}",shell=True)

# cosmo_rem = xr.open_mfdataset("/mnt/data/lcesarini/tmp/TOT_PRECIP.2D.201812.grb",engine='pynio')


# gdf_sta_99=gpd.GeoDataFrame(df_sta_99,geometry=gpd.points_from_xy(df_sta_99['lon'], df_sta_99['lat']))
# gdf_sph_99=gpd.GeoDataFrame(df_sph_99,geometry=gpd.points_from_xy(df_sph_99['lon'], df_sph_99['lat']))
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# gdf_sta_99.crs=gdf_sph_99.crs="EPSG:4326"
# #PLOT BOTH STATIONS AND SPHERA
# fig,ax=plt.subplots(1,2,figsize=(10,10),subplot_kw={'projection': ccrs.PlateCarree()})
# im1=gdf_sta_99.plot(ax=ax[0],markersize=15,column='q',label="Station",legend=False)
# im2=gdf_sph_99.plot(ax=ax[1],markersize=15,column='q',label="SPHERA",legend=False)
# [ax[_].set_title("99th percentile") for _ in range(2)]
# [ax[_].coastlines() for _ in range(2)]
# [ax[_].add_feature(cfeature.BORDERS) for _ in range(2)]
# [ax[_].add_feature(cfeature.STATES) for _ in range(2)]
# # Create a common colorbar
# cmap = mpl.cm.viridis
# bounds = [2,3,4,5,6]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=ax, orientation='horizontal',
#              label="Discrete intervals with extend='both' keyword")

# plt.savefig("/home/lcesarini/2022_resilience/99th_percentile.png")
# plt.close()

"""
EVALUATE DIURNAL CYCLE
"""

# for seas in seasons:

#     diurnal_sta=compute_quantiles_by_hour(sta_sph.sel(time=slice('2000-01-01','2010-01-01')),0.95,seas)
#     diurnal_sta=compute_quantiles_by_hour(sta_sph.sel(time=slice('2000-01-01','2010-01-01')),0.95,seas)
#     diurnal_sph=compute_quantiles_by_hour(sph_tri,0.95,seas)
#     diurnal_sph=compute_quantiles_by_hour(sph_tri,0.95,seas)

#     plt.plot(diurnal_sta.hour,diurnal_sta,'-o',c="red",label="Station")
#     plt.plot(diurnal_sph.hour,diurnal_sph,'-o',c="blue",label="SPHERA")
#     plt.legend()
#     plt.title(f"Diurnal cycle in {seas}")
#     plt.show()


#     np.nanquantile(sta_sph.sel(time=sta_sph["time.hour"].isin(1)).pr,q=0.999)
#     np.nanquantile(sta_sph.sel(time=sta_sph["time.hour"].isin(12)).pr,q=0.999)

    



