#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python
"""
Script containing metrics used to evalute the performance of CPM models
"""
import os
import sys
sys.path.append("/mnt/beegfs/lcesarini//2022_resilience/")
import numpy as np 
import xarray as xr 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings('ignore')

from resilience.utils import *
from resilience.utils.retrieve_data import get_season

os.chdir("/mnt/beegfs/lcesarini//2022_resilience/")

class ComputeMetrics:
    def __init__(self,ds) -> None:

        if isinstance(ds,xr.Dataset):
            print(f">> ds IS an xarray dataset")
            self.ds = ds
        else:
            print(f">>ERROR:ds is not an xarray dataset")
            
    def compute_wind(self,thr=10.8,quantile=[0.99]):
        """
        The threshold for frequency and intenisty of wind are taken from:

        https://it.wikipedia.org/wiki/Scala_di_Beaufort


        """
        #COMPUTE MEAN WIND SPEED
        mean_wind=self.ds["mw"].mean(dim='time') 
        #COMPUTE FREQUENCY
        freq = (self.ds["mw"] > thr).sum(dim='time') / self.ds["mw"].shape[0]
        #COMPUTE INTENSITY
        # intensity  = xr.where(self.ds["mw"] > thr,
        #                       self.ds["mw"],
        #                       np.nan).mean(dim='time', skipna=True)
        #COMPUTE pXX
        pXX  = self.ds["mw"].quantile(q=quantile, dim = 'time',skipna=True) 

        return mean_wind,freq,pXX
    
    def compute_tp(self,meters=True,max=False,quantile=[0.999],wethours=False):
        if meters:
            if hasattr(self.ds,"data_vars"):
                """
                !!! REMEMBER THAT FOR POINT WITH all NA's FREQUENCY IS EQUAL TO 0 INSTEAD of NAs !!!
                """
                #COMPUTE FREQUENCY
                freq = (self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1).sum(dim='time') / np.max(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"].shape)
                #COMPUTE INTENSITY
                MeanIntensity  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1,
                                        self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"],
                                        np.nan).mean(dim='time', skipna=True)
                VarIntensity   = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1,
                                        self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"],
                                        np.nan).var(dim='time', skipna=True)
                # SkewIntenisty  = xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
                #                         ds["tp" if "tp" in list(ds.data_vars) else "pr"],
                #                         np.nan).skew(dim='time', skipna=True)
                #COMPUTE pXX
                if wethours:
                    wet_ds  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1,
                                    self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"],
                                    np.nan)

                    pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 
            

                else:
                    pXX  = self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) 
            else:        
                #COMPUTE FREQUENCY
                freq = (self.ds > 0.1).sum(dim='time') / np.nanmax(self.ds.shape)
                #COMPUTE INTENSITY
                MeanIntensity  = xr.where(self.ds > 0.1,
                                        self.ds,
                                        np.nan).mean(dim='time', skipna=True)
                VarIntensity   = xr.where(self.ds > 0.1,
                                        self.ds,
                                        np.nan).var(dim='time', skipna=True)
                # SkewIntenisty  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1,
                #                         self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"],
                #                         np.nan).skew(dim='time', skipna=True)
                #COMPUTE pXX
                if wethours:
                    wet_ds  = xr.where(self.ds > 0.1,
                                       self.ds,
                                       np.nan)

                    pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) 
            

                else:
                    pXX  = self.ds.quantile(q=quantile, dim = 'time',skipna=True) 

        else:
            #COMPUTE FREQUENCY
            freq = (self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] * 3600 > 0.1).sum(dim='time') / np.nanmax(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"].shape)
            #COMPUTE INTENSITY
            if max:
                intensity  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] * 3600 > 0.1,
                                    self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] * 3600,
                                    np.nan).max(dim='time', skipna=True)
            else:        
                intensity  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] * 3600 > 0.1,
                                    self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] * 3600,
                                    np.nan).mean(dim='time', skipna=True)
            #COMPUTE pXX
            if wethours:
                wet_ds  = xr.where(self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"] > 0.1,
                                self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"],
                                np.nan)

                pXX  = wet_ds.quantile(q=quantile, dim = 'time',skipna=True) * 3600

            else:
                pXX  = self.ds["tp" if "tp" in list(self.ds.data_vars) else "pr"].quantile(q=quantile, dim = 'time',skipna=True) * 3600

            
            
        return freq,MeanIntensity,VarIntensity,pXX

        
    def compute_metrics_cmcc(self,meters=True,max=False,quantile=[0.999],wethours=False):
        if meters:
            #COMPUTE FREQUENCY
            freq = (self.ds["tp" if "tp" in list(self.ds.data_vars) else "TOT_PREC"] > 0.1).sum(dim='time') / self.ds["tp" if "tp" in list(self.ds.data_vars) else "TOT_PREC"].shape[0]
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
            #COMPUTE FREQUENCY
            freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "pr"].shape[0]
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

def compute_quantiles_by_hour(ds,q,SEAS):

    if ('lon' in list(ds.coords)) and ('x' not in list(ds.coords)):
        lon,lat='lon','lat'
    elif 'longitude' in list(ds.coords):
        lon,lat='longitude','latitude'
    elif 'x' in list(ds.coords):
        lon,lat='x','y'

    ds=get_season(ds,SEAS)
    q99_hourly=ds.groupby(ds['time.hour']).quantile(q=q)
    
    if hasattr(q99_hourly,'data_vars'):
        return q99_hourly['pr' if 'pr' in list(q99_hourly.data_vars) else 'tp'].mean(dim=[lat,lon])
    else:
        return q99_hourly.mean(dim=[lat,lon])
    


def compute_quantiles_by_hour_wind(ds,q,SEAS):

    if 'lon' in list(ds.coords):
        lon,lat='lon','lat'
    elif 'longitude' in list(ds.coords):
        lon,lat='longitude','latitude'
    elif 'x' in list(ds.coords):
        lon,lat='x','y'

    ds=get_season(ds,SEAS)
    q99_hourly=ds.groupby(ds['time.hour']).quantile(q=q)
    return q99_hourly['mw' if 'mw' in list(q99_hourly.data_vars) else 'bohh'].mean(dim=[lat,lon])

if __name__=="__main__":
    ds=xr.load_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ICTP/CPM/pr/ICTP_ECMWF-ERAINT_20050101000000-20060101000000.nc")
    obj=ComputeMetrics(ds)
    print(obj.compute_wind())
