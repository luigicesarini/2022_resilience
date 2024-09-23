#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python

import os
import pickle
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")
mask=xr.open_dataset("data/mask_stations_nan_common.nc")

def get_observed(name_station:str="AA_0220"):
    """
    Utility function that returns the observations and the dates of the recordings of station of interest

    @path_o: string file of the path to the file. !!Eventually the location of station data will be standardized 
    @path_dates: string file of the path to the file of dates !!Eventually the location of dates will be standardized 

    @name_station: name of the station of interest

    RETURNS

    Arrays: two array of dates and observations 
    """

    records_ith = pd.read_csv(f"stations/text/prec_{name_station}.csv")

    dates_ith   = pd.read_csv(f"data/dates/{name_station}.csv")

    # print(records_ith.head(),dates_ith.head())

    assert np.array(records_ith).shape[0]==np.array(dates_ith).shape[0],\
           f"Shapes of dates {np.array(records_ith).shape[0]} and records {np.array(dates_ith).shape[0]} are different."


    return np.array(records_ith), np.array(dates_ith['date'], dtype=np.datetime64)


def get_mod_at_obs(model:xr.Dataset, name_station:str,
                   rotated_cartopy,proj_cartopy, 
                   rotated:bool=True):
    """
    Function that returns the modelled recordings at the location, and for the dates
    of the station of interest.
    """

    #Get location of the station from metadata
    meta = pd.read_csv("meta_station_updated_col.csv")
    
    lon,lat = meta[meta.name == name_station]['lon'].values.item(),meta[meta.name == name_station]['lat'].values.item()
    if rotated:
        #rotate the latlon coords in cordex europe CRS
        rotated_coords=rotated_cartopy.transform_point(lon,lat,proj_cartopy)
    else:
        rotated_coords=(lon,lat)


    if ("rlon" in list(model.coords)) & ("rlat" in list(model.coords)):
        ds_overlap_station = model.sel(rlon=rotated_coords[0],rlat=rotated_coords[1],method="nearest")
    elif ("longitude" in list(model.coords)) & ("latitude" in list(model.coords)):
        ds_overlap_station = model.sel(longitude=rotated_coords[0],latitude=rotated_coords[1],method="nearest")
    elif ("lon" in list(model.coords)) & ("lat" in list(model.coords)):
        ds_overlap_station = model.sel(lon=rotated_coords[0],lat=rotated_coords[1],method="nearest")
    elif ("x" in list(model.coords)) & ("y" in list(model.coords)):
        ds_overlap_station = model.sel(x=rotated_coords[0],y=rotated_coords[1],method="nearest")

    return ds_overlap_station

def get_overlapping_dates(dates_o:np.datetime64,dates_m:np.datetime64):
    """
    Utility function that returns the available range between station data and observations

    date must be in format '%Y-%m-%d T%H:%M:%S' e.g.,'2020-12-31T22:00:00.000000'
    """

    min_slice_date = np.array([dates_o.min(),dates_m.min().values]).max()
    max_slice_date = np.array([dates_o.max(),dates_m.max().values]).min()

    return min_slice_date,max_slice_date  


def get_bbox(model_to_rotate,ds):
    """
    bbox_italy=(6.7499552751, 36.619987291, 18.4802470232, 47.1153931748))
    lat_bolo = 44.494887

    Bounding box of Po valley and alpine range:
    bbox = (6.70,44.35,18.50,47.20)

    Parameters
    ----------
    model_to_rotate : str, defaults to None
        Name of the model that we want to project on a regular grid.

    Returns
    -------
    

    Examples
    --------
        
        
    """

    if hasattr(model_to_rotate,'Lambert_Conformal'):
        
        rot = ccrs.LambertConformal(central_longitude = ds[list(ds.data_vars)[0]].longitude_of_central_meridian, 
                                    central_latitude  = ds[list(ds.data_vars)[0]].latitude_of_projection_origin,
                                    false_easting     = ds[list(ds.data_vars)[0]].false_easting,
                                    false_northing    = ds[list(ds.data_vars)[0]].false_northing)

        rotated_coords=rot.transform_point(ds.x[0:10],ds.y[0:10],ccrs.PlateCarree())


    if hasattr(model_to_rotate,'laea'):
        rot = ccrs.LambertAzimuthalEqualArea(central_longitude =52, 
                                    central_latitude  =10, 
                                    false_easting=4321000,
                                    false_northing=3210000, 
                                    globe=None)
    

def compute_metrics(ds,meters=True,max=False,quantile=[0.999],wethours=False):
    if meters:
        if hasattr(ds,"data_vars"):
            """
            !!! REMEMBER THAT FOR POINT WITH all NA's FREQUENCY IS EQUAL TO 0 INSTEAD of NAs !!!
            """
            #COMPUTE FREQUENCY
            freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1).sum(dim='time') / np.max(ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape)
            #COMPUTE INTENSITY
            MeanIntensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                                    ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                    np.nan).mean(dim='time', skipna=True)
            VarIntensity   = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                                    ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                    np.nan).var(dim='time', skipna=True)
            # SkewIntenisty  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
            #                         ds["tp" if "tp" in list(ds.data_vars) else "pr"],
            #                         np.nan).skew(dim='time', skipna=True)
            #COMPUTE pXX
            if wethours:
                wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                                ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                np.nan)

                pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 
        

            else:
                pXX  = ds["tp" if "tp" in list(ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) 
        else:        
            #COMPUTE FREQUENCY
            freq = (ds > 0.1).sum(dim='time') / np.nanmax(ds.shape)
            #COMPUTE INTENSITY
            MeanIntensity  = xr.where(ds > 0.1,
                                      ds,
                                      np.nan).mean(dim='time', skipna=True)
            VarIntensity   = xr.where(ds > 0.1,
                                      ds,
                                      np.nan).var(dim='time', skipna=True)
            # SkewIntenisty  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
            #                         ds["tp" if "tp" in list(ds.data_vars) else "pr"],
            #                         np.nan).skew(dim='time', skipna=True)
            #COMPUTE pXX
            if wethours:
                wet_ds  = xr.where(ds > 0.1,
                                   ds,
                                   np.nan)

                pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 
        

            else:
                pXX  = ds.quantile(q=quantile, dim = 'time',skipna=True) 

    else:
        #COMPUTE FREQUENCY
        freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1).sum(dim='time') / np.nanmax(ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape)
        #COMPUTE INTENSITY
        if max:
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600,
                                  np.nan).max(dim='time', skipna=True)
        else:        
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600,
                                  np.nan).mean(dim='time', skipna=True)
        #COMPUTE pXX
        if wethours:
            wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                               ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                               np.nan)

            pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) * 3600

        else:
            pXX  = ds["tp" if "tp" in list(ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) * 3600

        
        
    return freq,MeanIntensity,VarIntensity,pXX

    
def compute_metrics_cmcc(ds,meters=True,max=False,quantile=[0.999],wethours=False):
    if meters:
        #COMPUTE FREQUENCY
        freq = (ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"] > 0.1).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"].shape[0]
        #COMPUTE INTENSITY
        if max:
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"] > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"],
                                  np.nan).max(dim='time', skipna=True)
        else:        
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"] > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"],
                                  np.nan).mean(dim='time', skipna=True)
        #COMPUTE pXX
        if wethours:
            wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"] > 0.1,
                               ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"],
                               np.nan)

            pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 

        else:
            pXX  = ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"].quantile(q=quantile, dim = 'time') 

    else:
        #COMPUTE FREQUENCY
        freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape[0]
        #COMPUTE INTENSITY
        if max:
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600,
                                  np.nan).max(dim='time', skipna=True)
        else:        
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600 > 0.1,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] * 3600,
                                  np.nan).mean(dim='time', skipna=True)
        #COMPUTE pXX
        if wethours:
            wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"] > 0.1,
                               ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"],
                               np.nan)

            pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) * 3600

        else:
            pXX  = ds["tp" if "tp" in list(ds.data_vars) else "TOT_PREC"].quantile(q=quantile, dim = 'time',skipna=True) * 3600

        
        
    return freq,intensity,pXX


def compute_metrics_stat(ds,meters=True,max=False,quantile=[0.999],wethours=False):
    if meters:
        if hasattr(ds,"data_vars"):
            #COMPUTE FREQUENCY
            freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "pr"].time.shape[0]
            #COMPUTE INTENSITY
            if max:
                intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2,
                                    ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                    np.nan).max(dim='time', skipna=True)
                
            else:        
                intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2,
                                    ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                    np.nan).mean(dim='time', skipna=True)
            #COMPUTE pXX
            if wethours:
                wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2,
                                ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                                np.nan)

                pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True)

            else:
                pXX  = ds["tp" if "tp" in list(ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) 
        else:
            #COMPUTE FREQUENCY
            freq = (ds > 0.2).sum(dim='time') / ds.time.shape[0]
            #COMPUTE INTENSITY
            if max:
                intensity  = xr.where(ds > 0.2,
                                      ds,
                                      np.nan).max(dim='time', skipna=True)
            else:        
                intensity  = xr.where(ds > 0.2,
                                      ds,
                                      np.nan).mean(dim='time', skipna=True)
            #COMPUTE pXX
            if wethours:
                wet_ds  = xr.where(ds,
                                ds,
                                np.nan)

                pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True)

            else:
                pXX  = ds.quantile(q=quantile, dim = 'time',skipna=True) 

    else:
        #COMPUTE FREQUENCY
        freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"]  > 0.1).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape[0]
        #COMPUTE INTENSITY
        if max:
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"]  > 0.2,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] ,
                                  np.nan).max(dim='time', skipna=True)
        else:        
            intensity  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"]  > 0.2,
                                  ds["tp" if "tp" in list(ds.data_vars) else "pr"] ,
                                  np.nan).mean(dim='time', skipna=True)
        #COMPUTE pXX
        if wethours:
            wet_ds  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2,
                               ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                               np.nan)

            pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 

        else:
            pXX  = ds["tp" if "tp" in list(ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) 
        
        
    return freq,intensity,pXX


def compute_metrics_wind(ds,quantile=[0.999],threshold=5):
    """
    1) Mean wind speed
    2) 99th percentile of wind
    3) Winds over a threshold (i.e., 5m/s)

    
    """
    if hasattr(ds,"data_vars"):
        #COMPUTE MEAN WIND SPEED
        mean_speed = ds["mw"].mean(dim='time')
        #COMPUTE pXX
        pXX  = ds["mw"].quantile(q=quantile, dim = 'time',skipna=True) 
        #COMPUTE OVER THRESHOLD
        thr  = xr.where(ds["mw"] > threshold,1,0).sum(dim='time') / ds["mw"].shape[0]
                                
        
    return mean_speed,pXX,thr


def get_season(ds,season='JJA'):
    if season=='SON':
        return ds.isel(time=ds['time.season'].isin("SON"))
    elif season=='DJF':
        return ds.isel(time=ds['time.season'].isin("DJF"))
    elif season=='MAM':
        return ds.isel(time=ds['time.season'].isin("MAM"))
    elif season=='JJA':
        return ds.isel(time=ds['time.season'].isin("JJA"))  


def convert_coords(X=-3.9932448863983154,
                Y=0.2524249851703644,
                reg=ccrs.CRS("WGS84"),
                rot=ccrs.PlateCarree()):
    
    """
    Obtained the coordinates int the destination CRS (i.e., rot) from 
    the source CRS (i.e., reg).

    Useful to find the lonlat from curvilinear grid and obtain the bounds for 
    conservative remapping
    """

    return reg.transform_point(X,Y,src_crs=rot)[0],reg.transform_point(X,Y,src_crs=rot)[1]

def create_list_coords(ls_lon,ls_lat):
    """
    Create an array of ALL the combinations from two lists of points
    """
    xx,yy=np.meshgrid(ls_lon,ls_lat)
    list_xy=np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)],axis=1)
    return list_xy


def degree_to_meters(x:float):
    #1:111139=diffdegree:x
    return 111139 * x

def get_range(ds):
    return np.nanmin(ds.values), np.nanmax(ds.values)

def get_triveneto(ds,sta_val):
    if 'lat' in list(ds.coords):

        return (ds * mask.mask).isel(lon=ds.lon.isin(sta_val.lon),
                                     lat=ds.lat.isin(sta_val.lat))

    elif 'longitude' in list(ds.coords):
        return (ds * mask.mask).isel(longitude=ds.longitude.isin(sta_val.lon),
                                     latitude=ds.latitude.isin(sta_val.lat))

def clip_ds(ds,xmin,xmax,ymin,ymax):
    #xmin=6.5,xmax=13.9,ymin=43.25,ymax=47.5    
    if 'lat' in list(ds.coords):
        ds_clipped=ds.where((ds.lon >= xmin) & (ds.lon <= xmax) & 
                            (ds.lat >= ymin) & (ds.lat <= ymax), drop=True)
    elif 'x' in list(ds.coords):
        ds_clipped=ds.where((ds.lon >= xmin) & (ds.lon <= xmax) & 
                            (ds.lat >= ymin) & (ds.lat <= ymax), drop=True)
    elif 'longitude' in list(ds.coords):
        ds_clipped=ds.where((ds.lon >= xmin) & (ds.lon <= xmax) & 
                            (ds.lat >= ymin) & (ds.lat <= ymax), drop=True)
    return ds_clipped


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
    if 'lat' in list(xr.coords):
        xr_crop=xr.where((xr.lon > xmin) & (xr.lon < xmax) &\
                        (xr.lat > ymin) & (xr.lat < ymax), 
                        drop=True)
    elif 'latitude' in list(xr.coords):
        xr_crop=xr.where((xr.longitude >= xmin) & (xr.longitude <= xmax) &\
                        (xr.latitude >= ymin) & (xr.latitude <= ymax), 
                        drop=True)

    return xr_crop

def get_unlist(ll:list):
    ul=[]
    for sublist in ll:
        for file in sublist:
            ul.append(file)
    return ul

def nc_to_csv(ds:xr.Dataset,
              name:str,
              M:str,
              csv=True):
    """
    Functions that select inside a given extent

    Parameters
    ----------
    ds : xarrayDataset, 
        xarray dataset to crop
    
    name: str,
        name of the csv.

    M:     str,
        name of the metric    
    
    Returns
    -------
    
    Store the data in a csv

    Examples
    --------
    """
    if csv:
        if hasattr(ds,'data_vars'):
            df=ds[list(ds.data_vars)[0]].to_pandas()
        else:
            df=ds.to_pandas()
        df.reset_index(names=['lat']).\
        melt(id_vars="lat",value_vars=df.columns,var_name="lon",value_name=M).\
            to_csv(f"/mnt/beegfs/lcesarini//2022_resilience/csv/{name}.csv",index=False)
    else:
        if hasattr(ds,'data_vars'):
            df=ds[list(ds.data_vars)[0]].to_pandas()
        else:
            df=ds.to_pandas()
        dff=df.reset_index(names=['lat']).\
        melt(id_vars="lat",value_vars=df.columns,var_name="lon",value_name=M)

        return dff



def read_file_events(EV,MDL,THR,SEAS,INDICES,WH=False):
    """
    Functions that select inside a given extent

    Parameters
    ----------
    EV : str, 
        environmental varaible [pr,mw,combined]
    
    MDL: str,
        name of the model.

    THR:     str or int,
        threshold use to find the events

    SEAS:     str,
        Season of the event [JJA,MAM,SON,DJF]
    
    INDICES:     list,
        list of indices of land grid pointzzz
    
    Returns
    -------
    
    Store the data in a csv

    Examples
    --------
    """
        
    if EV == "combined":
        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_len_events_{THR}_{THR}_{SEAS}.pkl', 'rb') as file:
            len_per_above_threshold=pickle.load(file)

        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_mean_events_{THR}_{THR}_{SEAS}.pkl', 'rb') as file:
            mean_per_periods=pickle.load(file)

        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_max_events_{THR}_{THR}_{SEAS}.pkl', 'rb') as file:
            max_per_periods=pickle.load(file)
    elif EV == "mw":
        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_len_events_{THR}_{SEAS}.pkl', 'rb') as file:
            len_per_above_threshold=pickle.load(file)

        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_mw_mean_events_{THR}_{SEAS}.pkl', 'rb') as file:
            mean_per_periods=pickle.load(file)

        with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_mw_max_events_{THR}_{SEAS}.pkl', 'rb') as file:
            max_per_periods=pickle.load(file)    
    else:
        if WH:
            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_len_events_{THR}_{SEAS}_WH.pkl', 'rb') as file:
                len_per_above_threshold=pickle.load(file)

            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_mean_events_{THR}_{SEAS}_WH.pkl', 'rb') as file:
                mean_per_periods=pickle.load(file)

            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_max_events_{THR}_{SEAS}_WH.pkl', 'rb') as file:
                max_per_periods=pickle.load(file)    
        else:
            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_len_events_{THR}_{SEAS}.pkl', 'rb') as file:
                len_per_above_threshold=pickle.load(file)

            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_mean_events_{THR}_{SEAS}.pkl', 'rb') as file:
                mean_per_periods=pickle.load(file)

            with open(f'/mnt/data/lcesarini/EVENTS/{EV}/{MDL}_max_events_{THR}_{SEAS}.pkl', 'rb') as file:
                max_per_periods=pickle.load(file)    
    
    return (
        [len_per_above_threshold[ii.item()] for ii in INDICES],
        [mean_per_periods[ii.item()] for ii in INDICES],
        [max_per_periods[ii.item()] for ii in INDICES]
    )