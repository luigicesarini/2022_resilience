#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python
import os
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini//2022_resilience/")

def pattern_correlation(arr1,arr2,weights=1,type="centred"):
    """
    Computes the pattern correlation according to NCL function and IPCC definition,
    with centered and uncentered methods

    https://archive.ipcc.ch/ipccreports/tar/wg1/458.htm

    Parameters
    ----------
    arr1 : xr.Dataset, defaults to None
        model xarray dataset 

    arr2 : xr.Dataset, defaults to None
        observation xarray dataset 

    weights : np.array, defaults to 1
        Array containing the weights, if scalar == 1, No wweights applied

    type : str, defaults to "centred"
        A string choosing the method to use. Either centred of uncentred depending on the removal of seasonal value. 

    Returns
    -------
    returns the spatial correlation of between model and observations across all grid points
    of a given region.

    Examples
    --------
    """
    # arr1=ens.isel(lon=range(6),lat=range(6))#ens_sta
    # arr2=rea.isel(lon=range(6),lat=range(6))#sta.isel(quantile=0)

    if isinstance(arr1,xr.Dataset):
        arr1=arr1[list(arr1.data_vars)[0]].values

    if isinstance(arr2,xr.Dataset):
        arr2=arr2[list(arr2.data_vars)[0]].values

    assert arr1.shape == arr2.shape,"Different shape not allowed"
    
    weights=np.ones(shape=arr2.shape)
    
    if type == "centred":
        sumweights   = np.nansum(np.where(np.isfinite(arr1),weights,np.nan))
        xAvgArea     = np.nansum(arr1*weights)/sumweights      
        yAvgArea     = np.nansum(arr2*weights)/sumweights

        xAnom    = arr1 - xAvgArea          
        yAnom    = arr2 - yAvgArea

        xyCov    = np.nansum(weights*xAnom*yAnom)
        xAnom2   = np.nansum(weights*np.square(xAnom))
        yAnom2   = np.nansum(weights*np.square(yAnom))
    elif type == "uncentred":
        xyCov    = np.nansum(weights*arr1*arr2)
        xAnom2   = np.nansum(weights*np.square(arr1))
        yAnom2   = np.nansum(weights*np.square(arr2))
   

    if xAnom2 > 0 and yAnom2 > 0:
        r   = xyCov / ( np.sqrt(xAnom2) * np.sqrt(yAnom2))

    return r


def spatial_variability(arr1,arr2):
    """
    Spatial variability—ratio (observations ) of spatial standard
    deviations of seasonal values across all grid points of a selected region.
    
    Computes the pattern correlation according to NCL function and IPCC definition,
    with centered and uncentered methods

    https://archive.ipcc.ch/ipccreports/tar/wg1/458.htm

    Parameters
    ----------
    arr1 : xr.Dataset, defaults to None
        model xarray dataset 

    arr2 : xr.Dataset, defaults to None
        observation xarray dataset 

    weights : np.array, defaults to 1
        Array containing the weights, if scalar == 1, No wweights applied

    type : str, defaults to "uncentred"
        A string choosing the method to use. Either centred of uncentred depending on the removal of seasonal value. 

    Returns
    -------
    returns the spatial correlation of between model and observations across all grid points
    of a given region.

    Examples
    --------
    """
    if isinstance(arr1,xr.Dataset):
        arr1=arr1[list(arr1.data_vars)[0]].values

    if isinstance(arr2,xr.Dataset):
        arr2=arr2[list(arr2.data_vars)[0]].values

    assert arr1.shape == arr2.shape,"Different shape not allowed"

    v=1
    
    return v 