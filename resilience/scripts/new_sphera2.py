#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")
import argparse
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt 
from shapely.geometry import mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")

"""
BEGIN PARSER
"""
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-y','--year',
                    required=False,default='2000',
                    help='seasons to analyse')

parser.add_argument('-m','--month', 
                    required=False,default='01',
                    # choices=['f','i','q'],
                    help='metrics to analyse')

parser.add_argument('-ev','--env_var', 
                    required=False,default='pr',
                    choices=['pr','mw'],
                    help='environmental variable to analyse')

parser.add_argument('-p','--plot',action="store_true",
                    help='Plot the output')

parser.add_argument('-c','--compute',action='store_true',
                    help='Compute the output')

args = parser.parse_args()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT"
PATH_OUT="/mnt/beegfs/lcesarini/SPHERA"

NAME_VAR=args.env_var
PLOT=args.plot
YYYY_MM=f"{args.year}{args.month}"
YEAR=args.year
DEBUG=False
if DEBUG:
    NAME_VAR='pr'
    YYYY_MM="199501"
    YEAR=1995
"""
END PARSER
"""

matching_files=glob(f'/mnt/beegfs/lcesarini/SPHERA/original/{NAME_VAR}/*{YEAR}**remapped.grb2*')
ll_files=np.array(matching_files)[[(YYYY_MM in xx) and ('idx' not in xx) and ('2020' not in xx) for xx in matching_files]]
print(ll_files)

ds_ori=xr.open_mfdataset(ll_files,engine='cfgrib')

def remove_step(ds):

    ll_ds=[
        ds.isel(step=0,time=np.arange(0,ds.tp.shape[0],24)),
        ds.isel(step=1,time=np.arange(1,ds.tp.shape[0],24)),
        ds.isel(step=2,time=np.arange(2,ds.tp.shape[0],24)),
        ds.isel(step=3,time=np.arange(3,ds.tp.shape[0],24)),
        ds.isel(step=4,time=np.arange(4,ds.tp.shape[0],24)),
        ds.isel(step=5,time=np.arange(5,ds.tp.shape[0],24)),
        ds.isel(step=6,time=np.arange(6,ds.tp.shape[0],24)),
        ds.isel(step=7,time=np.arange(7,ds.tp.shape[0],24)),
        ds.isel(step=8,time=np.arange(8,ds.tp.shape[0],24)),
        ds.isel(step=9,time=np.arange(9,ds.tp.shape[0],24)),
        ds.isel(step=10,time=np.arange(10,ds.tp.shape[0],24)),
        ds.isel(step=11,time=np.arange(11,ds.tp.shape[0],24)),
        ds.isel(step=12,time=np.arange(12,ds.tp.shape[0],24)),
        ds.isel(step=13,time=np.arange(13,ds.tp.shape[0],24)),
        ds.isel(step=14,time=np.arange(14,ds.tp.shape[0],24)),
        ds.isel(step=15,time=np.arange(15,ds.tp.shape[0],24)),
        ds.isel(step=16,time=np.arange(16,ds.tp.shape[0],24)),
        ds.isel(step=17,time=np.arange(17,ds.tp.shape[0],24)),
        ds.isel(step=18,time=np.arange(18,ds.tp.shape[0],24)),
        ds.isel(step=19,time=np.arange(19,ds.tp.shape[0],24)),
        ds.isel(step=20,time=np.arange(20,ds.tp.shape[0],24)),
        ds.isel(step=21,time=np.arange(21,ds.tp.shape[0],24)),
        ds.isel(step=22,time=np.arange(22,ds.tp.shape[0],24)),
        ds.isel(step=23,time=np.arange(23,ds.tp.shape[0],24))
    ]

    ds_decum=xr.concat(ll_ds,dim='time')
    ds_decum=ds_decum.sortby('time')

    return ds_decum

def decumulate_prec(ds_cumulato):
    ll_dsds_2=[]
    # for i,t in  zip(np.tile(np.arange(0,24),int(ds_ori_slice.tp.shape[0]/24)),np.arange(ds_ori_slice.tp.shape[0])):
    list_timestamps=ds_cumulato.time.values
    list_steps=[str(timestamp)[11:19] for timestamp in ds_cumulato.time.values]

    # print(len(list_steps),len(list_timestamps))

    LL=list(ds_cumulato.coords)
    print(LL)
    LL.remove("longitude")
    LL.remove("latitude")
    print(LL)

    for i in np.arange(len(list_timestamps)):
        STEP=list_steps[i]
        if STEP == "01:00:00":
            ll_dsds_2.append(ds_cumulato.sel(
                time=list_timestamps[i],
                # step=STEP
                ).tp.drop_vars(LL)
                )
        elif STEP == "00:00:00":
            ll_dsds_2.append((ds_cumulato.sel(
                time=list_timestamps[i],
                # step="24:00:00"
                )-ds_cumulato.sel(
                    time=list_timestamps[i-1],
                    # step="23:00:00"
                    )).tp#.drop_vars(LL)
                    )
        else:
            ll_dsds_2.append((ds_cumulato.sel(
                time=list_timestamps[i],
                # step=list_steps[i]
                )-ds_cumulato.sel(
                    time=list_timestamps[i-1],
                    # step=list_steps[i-1]
                    )).tp#.drop_vars(LL)
                    )
        
    new_ds=xr.concat(ll_dsds_2,dim='time')
    new_ds['time']=list_timestamps
    new_ds=new_ds.sortby('time').load()

    return new_ds


def get_wethours(ds):
    if hasattr(ds,"tp"):
        _x=xr.where(ds.tp > 0.1, ds.tp, np.nan)
    else:
        _x=xr.where(ds > 0.1, ds, np.nan)
    return _x

# ds_dec_one_cell=remove_step(ds_one_cell)
# 
ds_dec_ori=remove_step(ds_ori)

ds_dec_ori2=decumulate_prec(ds_dec_ori.drop_vars("surface"))





        
# print(new_ds['time'].shape)


if "remapped" in ll_files[0]:
    ds_dec_ori2.assign_attrs(ds_ori.attrs).to_dataset(name='pr',promote_attrs=True).to_netcdf(f"{PATH_OUT}/decumulated/new/SPHERA_{YYYY_MM}.nc",encoding={'pr':{'zlib':True}})
else:
    ds_dec_ori2.assign_attrs(ds_ori.attrs).to_dataset(name='pr',promote_attrs=True).to_netcdf(f"{PATH_OUT}/decumulated/new/SPHERA_{YYYY_MM}_original.nc",encoding={'pr':{'zlib':True}})


"""
FUCKING TEST

First on quantile
"""
if PLOT:
    test_slice_remapped=xr.load_dataset(f"/mnt/data/lcesarini/SPHERA_{YYYY_MM}.nc")
    test_slice=xr.load_dataset(f"/mnt/data/lcesarini/SPHERA_{YYYY_MM}_original.nc")


    def get_q_by_h(ds):
        x=ds.groupby(ds["time.hour"]).quantile(q=0.99)
        if "longitude" in ds.coords:
            return x.mean(dim=["longitude","latitude"])
        else:
            return x.mean(dim=["x","y"])

    ds_q_remapped=get_q_by_h(test_slice_remapped)

    xxx=test_slice.groupby(test_slice["time.hour"]).quantile(q=0.99)
    xxxx=xxx.mean(dim=["x","y"])
    ds_q_original=get_q_by_h(test_slice)
    ds_q_original=xxxx#get_q_by_h(test_slice)


    ds_q_original.pr.plot(marker='*',label='original')
    ds_q_remapped.pr.plot(marker='*',label='remapped')
    plt.suptitle(f"Diurnal cycle of 99th quantile for the year {YYYY_MM}")
    plt.legend()
    plt.show()


    """
    FUCKING TEST

    Second on mean
    """
    test_slice_wh=xr.where(test_slice > 0.1, test_slice,np.nan )
    test_slice_avg=test_slice_wh.groupby(test_slice_wh["time.hour"]).mean()
    ds_avg=test_slice_avg.mean(dim=["latitude","longitude"]).load()
    ds_avg.pr.plot(marker='*')
    plt.ylabel("Prec. (mm/hr)")
    plt.suptitle(f"Diurnal cycle of wet hours for the year {YYYY_MM}")
    plt.show()