#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

def convert_coord(xy:tuple, pole:tuple, to_rotated:bool=True ):
    """
    Rotate this fucking coordinates
    """
    print(pole)
    lon = xy[0] * pi / 180
    lat = xy[1] * pi / 180

    theta = (90 - pole[1])    #rotation around y-axis
    phi   = pole[0]           #rotation around z-axis

    phi   = (phi * pi) / 180
    theta = (theta * pi) / 180

    x = cos(lon) * cos(lat)
    y = sin(lon) * cos(lat)
    z = sin(lat) 

    if to_rotated:
        
        x_new = cos(theta) * cos(phi) * x + cos(theta) * sin(phi) * y + sin(theta) * z
        y_new = -sin(phi) * x + cos(phi) * y
        z_new = -sin(theta) * cos(phi) * x - sin(theta) * sin(phi) * y + cos(theta) * z
    
    else:

        phi   = -phi
        theta = -theta

        x_new = cos(theta) * cos(phi) * x + sin(phi) * y + sin(theta) * cos(phi) * z
        y_new = -cos(theta) * sin(phi) * x + cos(phi) * y - sin(theta) * sin(phi) * z 
        z_new = -sin(theta) * x + cos(theta) * z

    lon_new = atan2(y_new,x_new)
    lat_new = asin(z_new)
    """TO DEGREE AGAIN"""
    lon_new = (lon_new * 180) / pi
    lat_new = (lat_new * 180) / pi
    lon_new = lon_new + 180

    return [lon_new,lat_new]

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
