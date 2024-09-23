#!/mnt/data/lcesarini/miniconda3/miniconda3/envs/colorbar/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
# sys.path.append("/mnt/data/lcesarini/2024_guycarpenter")
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")
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

# os.chdir("/mnt/data/lcesarini/2024_guycarpenter")

# from src.utils import *
from resilience.smev.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-y","--year",
                    help="year", type = np.int64,
                    required=False,choices=np.arange(1900,2101))

args = parser.parse_args()



PATH_DATA="/mnt/beegfs/lcesarini/2022_resilience/data_smev"

dd=loadmat(f"{PATH_DATA}/s0001_v3.mat")
values=dd['S']['vals'][0,0]
dd

values=np.where(np.isnan(values),0,values)
dates=pd.read_csv(f"{PATH_DATA}/datetime_data.csv",parse_dates=['Time'], date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S'))
time_resolution=5 # in mu
#create dataframe (2945664,2) from values.reshape(-1) and dates.values.reshape(-1)
df=pd.DataFrame({'date':dates.values.reshape(-1), 'value':values.reshape(-1)})


# durations=pd.read_csv(f"{PATH_DATA}/durations.csv")
durations=[15,30,45,60,120,180,360,720,1440]

YEAR=args.year if args.year is not None else 2000
RP = get_return_period()
#use date column as index
df.set_index('date',inplace=True)
# total_prec = df.groupby(df.index.year)['value'].sum()
# mean_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)['value'].mean()
# sd_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)['value'].std()
# count_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)['value'].count()

threshold=0#np.quantile(df.value,q=0.9999)
# threshold=20
separation=24

idx_ordinary=get_ordinary_events(df,'value', threshold, separation)

ll_short=[True if ev[-1]-ev[0] >= pd.Timedelta(minutes=25) else False for ev in idx_ordinary]
ll_dates=[(ev[-1].strftime("%Y-%m-%d %H:%M:%S"),ev[0].strftime("%Y-%m-%d %H:%M:%S")) if ev[-1]-ev[0] >= pd.Timedelta(minutes=25) else (np.nan,np.nan) for ev in idx_ordinary]

arr_vals=np.array(ll_short)[ll_short]
arr_dates=np.array(ll_dates)[ll_short]
filtered_list = [x for x, keep in zip(idx_ordinary, ll_short) if keep]
list_year=pd.DataFrame([filtered_list[_][0].year for _ in range(len(filtered_list))],columns=['year'])
n_ordinary_per_year=list_year.reset_index().groupby(["year"]).count()
n_ordinary=n_ordinary_per_year.mean().values.item()

dict_param={}
dict_rp={}

for d in range(len(durations)):
    arr_conv=np.convolve(df.value,np.ones(int(durations[d]/time_resolution),dtype=int),'same')

    # Create xarray dataset

    ds = xr.Dataset(
        {
            f'tp{durations[d]}': (['time'], arr_conv.reshape(-1)),
        },
        coords={
            'time':df.index.values.reshape(-1)
        },
        attrs = dict(description = f"Array of {durations[d]} minutes precipitation data",
                                            unit = '')
    )


    # ds.sel(time=slice(arr_dates[-7,1],arr_dates[-7,0]))[f'tp{durations[d]}'].max(skipna=True).item()
    ll_vals=[ds.sel(time=slice(arr_dates[_,1],arr_dates[_,0]))[f'tp{durations[d]}'].max(skipna=True).item() for _ in range(arr_dates.shape[0])]
    ll_yrs=[int(arr_dates[_,1][0:4]) for _ in range(arr_dates.shape[0])]

    # Create xarray dataset
    ds_ams = xr.Dataset(
        {
            'vals': (['year'], ll_vals),
        },
        coords={
            'year':ll_yrs
        },
        attrs = dict(description = f"Array of {durations[d]} minutes precipitation data",
                                            unit = '')
    ) * 60 / durations[d]


    # AMS=ds_ams.groupby('year').max() 


    # yrs=AMS.year.values
    # M = AMS.year.values.shape[0]
    # emp_T = 1/(1-plotting_position(M))


    # win_sz=1; 
    # min_yr=min(yrs); 
    # max_yr=max(yrs)-win_sz+1
    # win_n=yrs.shape[0]-win_sz+1

    # MEV_phat = np.zeros(shape=(win_n,3)) * np.nan
    # MEV_qnt = np.zeros(shape=(win_n,RP.shape[0])) * np.nan 



    shape,scale=estimate_smev_parameters(
                    ds_ams.sel(year=YEAR).vals.values,
                    'value', [0.75, 1])
    
    smev_RP=smev_return_values(RP, shape, scale, n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())
    
    dict_param[f"{durations[d]}"]=scale,shape
    dict_rp[f"{durations[d]}"]=smev_RP

print(f"\n{YEAR}")
# print(f"\nN° of ordinary events: {arr_dates.shape[0]}")
# print(f"Threshold: {threshold:.2f}")
# print(f"Separation between events in hours: {separation}")
# print(f"Avg ordinary events per year: {n_ordinary:.2f}\n")

df_rp = pd.DataFrame(dict_rp)
df_rp.index=RP



print(df_rp)
# for rp,value in  zip(RP,dict_rp["60"]):
#     print(f"{rp:.3f}:  {value:.2f}") 
