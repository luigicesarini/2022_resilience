#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python

import os
import numpy as np 
import xarray as xr
from glob import glob 
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from netCDF4 import Dataset
import warnings
warnings.filterwarnings('ignore')

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/beegfs/lcesarini/BIAS_CORRECTED/" 
PATH_SPHERA="/mnt/beegfs/lcesarini/SPHERA/decumulated/new"
def get_unlist(ll:list) -> list:
    ul=[]
    for sublist in ll:
        for file in sublist:
            ul.append(file)
    return ul

def create_list_coords(ls_lon:np.ndarray[float],ls_lat:np.ndarray[float]) -> list[float]:
    """
    Create an array of ALL the combinations from two lists of points
    """
    xx,yy=np.meshgrid(ls_lon,ls_lat)
    list_xy=np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)],axis=1)
    return list_xy

def get_season(ds:xr.DataArray,season:str='JJA') -> xr.DataArray:
    if season=='SON':
        return ds.isel(time=ds['time.season'].isin("SON"))
    elif season=='DJF':
        return ds.isel(time=ds['time.season'].isin("DJF"))
    elif season=='MAM':
        return ds.isel(time=ds['time.season'].isin("MAM"))
    elif season=='JJA':
        return ds.isel(time=ds['time.season'].isin("JJA"))  

def get_range_yrs(mdl:str)->np.ndarray:
    """
    Find the start and finish year for the model.    
    """
    ll_files=glob(f"/mnt/beegfs/lcesarini/DATA_FPS/Historical/{mdl}/CPM/pr/*.nc")

    years=[]
    if (len(ll_files)==0) or (mdl=="FZJ-IBG3-WRF381CA") or (mdl=="FZJ-IDL-WRF381DA"):
        pass    
    else:
        for file in ll_files:
            years.append(int(os.path.basename(file).split('_')[2][:4]))
        
    return np.arange(min(years),max(years)+1)

def get_reference_data(REF:str,yrs_train:np.ndarray[float]):
    """
    Get reference data depending on the `AREA`
    """

    if  REF == "SPHERA":
        list_ref_train=[glob(f"{PATH_SPHERA}/*{year}*") for year in yrs_train]
        list_xr=[xr.open_dataset(xr.backends.NetCDF4DataStore(Dataset(path, mode='r'))) for path in get_unlist(list_ref_train)]
        ref_train=xr.concat(list_xr,dim='time')

        return ref_train
    elif REF == "STATIONS":

        ref_train=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_train]).load()
        max_sta=np.nanmax(ref_train.pr.values[:,:,:],axis=2)
        return ref_train,max_sta

def get_list_coords(AREA:str,ref_train:xr.DataArray,max_sta:xr.DataArray) -> list:
    if AREA =='triveneto':
        x,y=np.where((max_sta > 0))
        x_,y_=np.where(np.isnan(max_sta))

        xy=np.concatenate([y.reshape(-1,1),x.reshape(-1,1)],axis=1)
        xy_=np.concatenate([y_.reshape(-1,1),x_.reshape(-1,1)],axis=1)
        assert xy.shape[0] == 172, f"Number of cells, different from number {xy.shape[0]}"
    else:
        ref_train_tri=ref_train

        name_lon="lon" if 'lon' in list(ref_train.coords) else 'longitude'
        name_lat="lat" if 'lat' in list(ref_train.coords) else 'latitude'
        xy=create_list_coords(np.arange(ref_train_tri[name_lon].values.shape[0]).reshape(-1,1),
                            np.arange(  ref_train_tri[name_lat].values.shape[0]).reshape(-1,1))
        
        
        xy_=np.where(np.isnan(np.nanmax(ref_train_tri.pr,axis=0)))

    return xy,xy_

def get_model_data(mdl:str,yrs_train:np.ndarray,list_train:list,list_valid:list) -> list[xr.DataArray] :
    
    if mdl == "ETH":
        
        from resilience.utils.fix_year_eth import fix_eth
        eth=fix_eth()

        mod_train=eth.sel(time=eth['time.year'].isin(yrs_train))#.load()
        mod_adjust=xr.open_mfdataset([item for list in list_valid for item in list]).load()
        
    else:   
        mod_train=xr.open_mfdataset([item for list in list_train for item in list]).load()
        mod_adjust=xr.open_mfdataset([item for list in list_valid for item in list]).load()

    return mod_train,mod_adjust

def get_single_cell(REF:str,AREA:str,REF_TRAIN:xr.DataArray,MOD_TRAIN:xr.DataArray,MOD_ADJUST:xr.DataArray,LON:float,LAT:float) -> list[xr.DataArray]:
    """
    ATTENTION:Adjust 0.1 & 0.2 for wet hours or not, and units when not working with hourly rainfall.
    """
    if REF == "SPHERA":
        if AREA=="triveneto":
            lon=REF_TRAIN.isel(lon=LON,lat=LAT).lon.item()
            lat=REF_TRAIN.isel(lon=LON,lat=LAT).lat.item()
            ref_train_sc=REF_TRAIN.sel(longitude=lon,latitude=lat,method='nearest').\
                where(REF_TRAIN.sel(longitude=lon,latitude=lat,method='nearest').pr > 0.1).assign_attrs(units="mm/hr")
        elif AREA=="northern_italy":
            lon,lat=LON,LAT
            
            ref_train_sc=REF_TRAIN.isel(longitude=lon,latitude=lat).\
                   where(REF_TRAIN.isel(longitude=lon,latitude=lat).pr > 0.1).assign_attrs(units="mm/hr")
        
        #Filter only WET hours for single cell
        mod_train_sc=MOD_TRAIN.isel(lon=lon,lat=lat).\
            where(MOD_TRAIN.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
        mod_adjust_sc=MOD_ADJUST.isel(lon=lon,lat=lat).\
            where(MOD_ADJUST.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
    elif REF == "STATIONS":
        lon=REF_TRAIN.isel(lon=LON,lat=LAT).lon.item()
        lat=REF_TRAIN.isel(lon=LON,lat=LAT).lat.item()
        ref_train_sc=REF_TRAIN.sel(lon=lon,lat=lat,method='nearest').\
               where(REF_TRAIN.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
    
        #Filter only WET hours for single cell
        mod_train_sc=MOD_TRAIN.sel(lon=lon,lat=lat,method='nearest').\
               where(MOD_TRAIN.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
        # sta_train_sc=sta_train.where(sta_train.pr > 0.2).assign_attrs(units="mm/hr").load()

        mod_adjust_sc=MOD_ADJUST.sel(lon=lon,lat=lat,method='nearest').\
                where(MOD_ADJUST.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")

    return ref_train_sc,mod_train_sc,mod_adjust_sc

def main():
    list_mod_train=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/CNRM/CPM/pr/*{year}*") for year in np.arange(2000,2010)]
    list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/Rcp85/CNRM/CPM/pr/*{year}*") for year in np.arange(2041,2050)]
    
    ref_train=get_reference_data('SPHERA',np.arange(2000,2002))
    mt,ma=get_model_data("CNRM",np.arange(2000,2010),list_mod_train,list_mod_adjust)
    xy,xy_=get_list_coords("northern_italy",mt,mt)
    print(xy[0,0],xy[0,1])
    rt_sc,mt_sc,ma_sc=get_single_cell(
        REF='SPHERA',AREA='northern_italy',REF_TRAIN=ref_train,MOD_TRAIN=mt,MOD_ADJUST=ma,LON=xy[0,0],LAT=xy[0,1]
    )
    print(mt_sc,ma_sc)


if __name__=="__main__":
    main()