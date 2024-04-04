#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python

import os
os.chdir("/mnt/beegfs/lcesarini//2022_resilience/")
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from colorama import Fore, Back, Style
from cartopy import feature as cfeature
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')
path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"



meta_station = pd.read_csv("meta_station_updated_col.csv")


def findNA(array_date:np.array,
             min_date=np.datetime64('1982-01-01T00:00:00'),
             max_date=np.datetime64('2020-12-31T22:00:00'),
             ):
    """
    """
    np_dates = array_date#np.array(array_date['date'],dtype=np.datetime64)
    minmin=np.max([np.array(min_date,dtype='datetime64[s]'),np.array(np_dates.min(), dtype="datetime64[s]")])
    maxmax=np.min([np.array(max_date,dtype='datetime64[s]'),np.array(np_dates.max(), dtype="datetime64[s]")])
    
    np_range=np.arange(minmin,maxmax,timedelta(hours=1))
    
    array_isnotin = np.isin(np_range,np_dates, invert =True)

    missing_date = np_range[array_isnotin]

    return missing_date
    


def read_data(name_station="AA_0220"):
    dates  = pd.read_csv(f"data/dates/{name_station}.csv") 
    prec_0 = pd.read_csv(f"stations/text/prec_{name_station}.csv")

    np_dates = np.array(dates['date'],dtype=np.datetime64)
    minmin=np_dates.min()
    maxmax=np_dates.max()
    
    np_range=np.arange(minmin,maxmax,timedelta(hours=1))

    idx_missing=np.isin(np_range,np_dates, invert =True)
    
    np_range[idx_missing]



if __name__ == "__main__":

    print("Test missing NAs function")
    name_station="VE_0235"
    dates = np.array(pd.read_csv(f"data/dates/{name_station}.csv")['date'],
                     dtype=np.datetime64)

    dates_missing = findNA(array_date=dates)

    print(f"Yes there are missing data" if dates_missing.shape[0] > 0 else f"Not missing in {name_station}")
    print(f"Hours missing: {dates_missing.shape[0]}")


# pd.DataFrame(np_range).columns

# df_range=pd.DataFrame(np_range).rename({"0":"date"},axis=1)
# pd.DataFrame(np_dates)
