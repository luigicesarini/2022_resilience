#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import seaborn as sns
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from windrose import WindroseAxes 
from shapely.geometry import mapping

from cartopy import feature as cfeature

from utils import *


os.chdir("/home/lcesarini/2022_resilience/")

# from scripts.utils import *

seasons=['DJF','JJA']
list_ms=['Frequency','Intensity','Heavy Prec.']
abbr_ms=['f','i','q']

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()
PATH_WIND="/mnt/data/lcesarini/wind"
PATH_COMMON="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")


meta_stazioni=pd.read_csv(f"{PATH_WIND}/meta_stazioni_vento_4326.csv")
meta_stazioni

speed_10=glob("/mnt/data/lcesarini/wind/10m/*")

stat=pd.read_csv(speed_10[0],encoding="utf-8")

if __name__ == "__main__":
    print("NANANAN")
