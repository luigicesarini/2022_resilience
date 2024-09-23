#!/home/luigi.cesarini//.conda/envs/my_xclim_env/bin/python

import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/resilience")
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
from scipy.io import loadmat
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from scipy.stats import genextreme as gev
os.chdir("/mnt/beegfs/lcesarini/2022_resilience")

from smev import *


PATH_DATA="/mnt/beegfs/lcesarini/2022_resilience/data_smev"




if __name__=="__main__":

    YEAR=1992
    TYPE='numpy' # choiches numpy or panda
    S=SMEV(
        threshold=0,
        separation=24,
        return_period=np.array([100,500]),
        # return_period=get_return_period(),
        # durations=[15,30,45,60,120,180,360,720,1440],
        durations=[15,30],
        time_resolution=5
    )

    data=pd.read_parquet(f"{PATH_DATA}/s0001_v3.parquet")
    if TYPE=="pandas":
        """"
        Get ordinary events
        """
        idx_ordinary=S.get_ordinary_events(data=data,dates=data.index,name_col='value')
        """
        Remove short events
        """
        arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary,25)

        dict_param={}
        dict_rp={}


        for d in range(len(S.durations)):

            arr_conv=np.convolve(data.value,np.ones(int(S.durations[d]/S.time_resolution),dtype=int),'same')

            # Create xarray dataset

            ds = xr.Dataset(
                {
                    f'tp{S.durations[d]}': (['time'], arr_conv.reshape(-1)),
                },
                coords={
                    'time':data.index.values.reshape(-1)
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data",unit = '[m]')
            )

            # ds.sel(time=slice(arr_dates[-7,1],arr_dates[-7,0]))[f'tp{durations[d]}'].max(skipna=True).item()
            ll_vals=[ds.sel(time=slice(arr_dates[_,1],arr_dates[_,0]))[f'tp{S.durations[d]}'].max(skipna=True).item() for _ in range(arr_dates.shape[0])]
            ll_yrs=[int(arr_dates[_,1][0:4]) for _ in range(arr_dates.shape[0])]

            # Create xarray dataset
            ds_ams = xr.Dataset(
                {
                    'vals': (['year'], ll_vals),
                },
                coords={
                    'year':ll_yrs
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data", unit = '[m]')
            ) * 60 / S.durations[d]

            shape,scale=estimate_smev_parameters(
                            ds_ams.sel(year=YEAR).vals.values,
                            'value', [0.75, 1])
            
            smev_RP=smev_return_values(S.return_period, shape, scale, n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())
            
            dict_param[f"{S.durations[d]}"]=scale,shape
            dict_rp[f"{S.durations[d]}"]=smev_RP

        print(f"\n{YEAR}")
        # print(f"\nN° of ordinary events: {arr_dates.shape[0]}")
        # print(f"Threshold: {threshold:.2f}")
        # print(f"Separation between events in hours: {separation}")
        # print(f"Avg ordinary events per year: {n_ordinary:.2f}\n")

        df_rp = pd.DataFrame(dict_rp)
        df_rp.index=S.return_period

        print(df_rp)
        # for rp,value in  zip(RP,dict_rp["60"]):
        #     print(f"{rp:.3f}:  {value:.2f}") 
    elif TYPE=="numpy":
        """"
        Get ordinary events
        """
        df_arr=np.array(data.value)
        df_dates=np.array(data.index)
        idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates,name_col='value')

        """
        Remove short events
        """
        arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary,25)
        print(arr_vals)
        dict_param={}
        dict_rp={}


        for d in range(len(S.durations)):

            arr_conv=np.convolve(df_arr,np.ones(int(S.durations[d]/S.time_resolution),dtype=int),'same')

            # Create xarray dataset

            ds = xr.Dataset(
                {
                    f'tp{S.durations[d]}': (['time'], arr_conv.reshape(-1)),
                },
                coords={
                    'time':df_dates.reshape(-1)
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data",unit = '[m]')
            )
            # ds.sel(time=slice(arr_dates[-7,1],arr_dates[-7,0]))[f'tp{durations[d]}'].max(skipna=True).item()
            ll_vals=[ds.sel(time=slice(arr_dates[_,1],arr_dates[_,0]))[f'tp{S.durations[d]}'].max(skipna=True).item() for _ in range(arr_dates.shape[0])]
            ll_yrs=[arr_dates[_,1].astype('datetime64[Y]').item().year for _ in range(arr_dates.shape[0])]
            

            # Create xarray dataset
            ds_ams = xr.Dataset(
                {
                    'vals': (['year'], ll_vals),
                },
                coords={
                    'year':ll_yrs
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data", unit = '[m]')
            ) * 60 / S.durations[d]

            shape,scale=estimate_smev_parameters(
                            ds_ams.sel(year=YEAR).vals.values,
                            'value', [0.75, 1])
            
            smev_RP=smev_return_values(S.return_period, shape, scale, n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())
            
            dict_param[f"{S.durations[d]}"]=scale,shape
            dict_rp[f"{S.durations[d]}"]=smev_RP

        print(f"\n{YEAR}")
        # print(f"\nN° of ordinary events: {arr_dates.shape[0]}")
        # print(f"Threshold: {threshold:.2f}")
        # print(f"Separation between events in hours: {separation}")
        # print(f"Avg ordinary events per year: {n_ordinary:.2f}\n")

        df_rp = pd.DataFrame(dict_rp)
        df_rp.index=S.return_period

        print(df_rp)
        # for rp,value in  zip(RP,dict_rp["60"]):
        #     print(f"{rp:.3f}:  {value:.2f}") 
