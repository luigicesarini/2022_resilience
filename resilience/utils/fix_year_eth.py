#!/home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from glob import glob
from tqdm import tqdm
from datetime import datetime, timedelta
# from utils import *

os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")

def fix_eth():
    eth=xr.open_mfdataset([f'/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_{year}01010030_{year}12312330.nc' for year in range(2000,2009)])

    time_2009=np.arange(np.datetime64('2009-01-01T00:33:00.000000000'),
                        np.datetime64('2010-01-01T00:33:00.000000000'),
                        timedelta(hours=1))

    time_bnds_0=np.arange(np.datetime64('2009-01-01T00:00:00.000000000'),
                        np.datetime64('2010-01-01T00:00:00.000000000'),
                        timedelta(hours=1))

    time_bnds_1=np.arange(np.datetime64('2009-01-01T01:00:00.000000000'),
                        np.datetime64('2010-01-01T01:00:00.000000000'),
                        timedelta(hours=1))

    assert time_bnds_0.shape==time_bnds_1.shape,"Timess probleeemmm"

    time_bnds=np.concatenate([time_bnds_0.reshape(-1,1),time_bnds_1.reshape(-1,1)],axis=1)

    fn='/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200901010030_200912312330.nc'

    # xr.load_dataset(fn, engine='cfgrib')


    ds=nc.Dataset(fn)

    # eth_newt_eth=xr.DataArray(ds.variables['pr'][:],
    #                             coords={
    #                                 'time':time_2009,
    #                                 'lat':eth.lat,
    #                                 'lon':eth.lon,

    #                                 },
    #                             dims={
    #                                 "time":time_2009.shape[0],
    #                                 "lat":eth.lat.shape[0],
    #                                 "lon":eth.lon.shape[0]
    #                             }
    #                             )

    # eth_newt_eth=eth_newt_eth.expand_dims(dim={"bnds": 2})
    # eth_newt_eth.shape
    # ds_eth_new=eth_newt_eth.to_dataset(name='pr',promote_attrs=True)
    # ds_eth_new.assign(time_bnds=((ds_eth_new.time,ds_eth_new.bnds),time_bnds))


    ds = xr.Dataset(
        data_vars=dict(
            pr=(["time", "lat", "lon"], ds.variables['pr'][:]),
            time_bnds=(["time", "bnds"], time_bnds),
        ),
        coords=dict(
            lon=eth.lon,
            lat=eth.lat,
            time=time_2009,
        ),
        attrs=eth.attrs,
    )




    all_eth=xr.concat([eth,ds],dim='time')
    eth__rg=all_eth.load()

    return eth__rg