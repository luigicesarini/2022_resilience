#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience")
from resilience.utils import *
import rasterio
import argparse
# import rioxarray
import numpy as np 
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from xclim import sdba
import subprocess as sb
from scipy import stats
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from xclim.core.calendar import convert_calendar,get_calendar

import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini/2022_resilience")
from resilience.utils import *
from resilience.bias.utils import *

"""

Time to run the bias correction for 8 models:
 ~ 90 minutes 

"""
"""
PARSER
"""
parser = argparse.ArgumentParser()
parser.add_argument("-p","--period",
                    help="Which period to adjust the data",
                    required=True,default="VALIDATION",
                    choices=["TRAIN","VALIDATION"]  
                    )
parser.add_argument("-mt","--model_train",
                    help="Which run to use for training the model",
                    required=True,default="ECMWF-ERAINT",choices=["ECMWF-ERAINT","Historical"]  
                    )
parser.add_argument("-ref","--reference",
                    help="Which dataset to use for bias correction",
                    required=True,default="SPHERA",
                    choices=["SPHERA","STATIONS"]  
                    )
parser.add_argument("-a","--area",
                    help="Area over which the bias correction is applied",
                    default='triveneto',choices=["triveneto","northern_italy"]
                    )
parser.add_argument("-seas","--season",
                    help="Season to correct",
                    default='JJA',choices=["MAM","JJA","SON","DJF"]
                    )
parser.add_argument("-m","--model",
                    help="Which model to correct",
                    default='CNRM',choices=["CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","ETH"]
                    )
parser.add_argument("-db","--debug",
                    help="Debugging some error",
                    action="store_true"
                    )
parser.add_argument("--split",
                    help="How to split the data, random or sequential",
                    choices=['RANDOM','SEQUENTIAL']
                    )
args = parser.parse_args()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/beegfs/lcesarini/BIAS_CORRECTED/" 
# BM  = str(sys.argv[1])
# MDL = str(sys.argv[2])
REF = args.reference
ADJUST = args.period
SEASONS=['SON','DJF','MAM','JJA'] #,
AREA=args.area
SPLIT=args.split
mdl=args.model
seas=args.season
DEBUG=args.debug
MODEL_TRAIN=args.model_train
# #rint all the arguments
# print(f"period_:{ADJUST}")
# print(f"model_train_:{MODEL_TRAIN}")
# print(f"reference_:{REF}")
# print(f"area_:{AREA}")
# print(f"season_:{seas}")
# print(f"model_:{mdl}")
# print(f"split_:{SPLIT}")


# print(f"debug_:{DEBUG}")
if DEBUG:
    SLICE='far'
    mdl='ETH'
    seas='JJA'
    REF = "SPHERA"
    ADJUST = "VALIDATION"
    AREA="northern_italy"
    MODEL_TRAIN='Historical'
"""
QM bias correction only on the wet hours.
"""

# if MODEL_TRAIN == "Historical":
#     yrs_train_mod=np.arange(1996,2006)
#     yrs_train=np.arange(2000,2010)
# else:
#     yrs_train_mod=np.arange(2000,2010)
#     yrs_train=np.arange(2000,2010)

range_yrs=get_range_yrs(mdl)
if (range_yrs.shape[0] % 2 != 0):
    print("Odd number of years") 

if SPLIT == "SEQUENTIAL":
    yrs_train = range_yrs[:range_yrs.shape[0]//2]
    yrs_valid = range_yrs[range_yrs.shape[0]//2:]
elif SPLIT == "RANDOM":
    yrs_train = np.sort(np.random.choice(np.arange(range_yrs[0],range_yrs[-1]),size=int(range_yrs.shape[0]//2),replace=False))
    yrs_valid = np.sort(np.setdiff1d(np.arange(range_yrs[0],range_yrs[-1]),yrs_train))

if ADJUST == "TRAIN":
    yrs_valid = yrs_train

#LOAD REFERENCE DATA
if  REF == "SPHERA":
    ref_train=get_reference_data(REF,yrs_train)

elif REF == "STATIONS":
    ref_train,max_sta=get_reference_data(REF,yrs_train)
    

"""
Get coordinates depending on the `AREA`
"""
if AREA =='triveneto':
    xy,xy_=get_list_coords(AREA,ref_train,max_sta)
elif AREA == "northern_italy":
    xy,xy_=get_list_coords(AREA,ref_train,ref_train)

"""
LOAD THE DATA FOR EACH MODEL

"""
print(f"Running:{mdl} {datetime.today().strftime('%d/%m/%Y %H:%M:%S')}")

list_mod_train =[glob(f"{PATH_COMMON_DATA}/{MODEL_TRAIN}/{mdl}/CPM/pr/*{year}*") for year in yrs_train]
list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/{MODEL_TRAIN}/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
# x=xr.open_mfdataset(list_mod_adjust[0]).load()

mod_train,mod_adjust=get_model_data(mdl,yrs_train,list_mod_train,list_mod_adjust)

if get_calendar(mod_train) == '360_day':
    mod_train=mod_train.convert_calendar('standard',align_on='year')

if get_calendar(mod_adjust) == '360_day':
    mod_adjust=mod_adjust.convert_calendar('standard',align_on='year')

# mod_adjust=xr.open_mfdataset([item for list in list_mod_adjust[:10] for item in list])
# mod_adjust=xr.open_mfdataset(get_unlist(list_mod_adjust)[:6])
# mod_adjust=xr.open_mfdataset(get_unlist(list_mod_adjust)[5])
# mod_adjust
# xr.open_mfdataset('/home/giorgia.fosser/DATA_FPS/Rcp85/CNRM/CPM/pr/pr_ALP-3_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-AROME41t1_x2yn2v1_1hr_209501010030-209512312330.nc')
"""
Starts the correction
"""

list_xr_qdm=[]
list_xr_eqm=[]


for id_coord in xy:
# for id_coord in tqdm(xy,total=42976):
    
    ref_train_sc,mod_train_sc,mod_adjust_sc=get_single_cell(
        REF='SPHERA',AREA='northern_italy',
        REF_TRAIN=ref_train,MOD_TRAIN=mod_train,MOD_ADJUST=mod_adjust,
        LON=id_coord[0],LAT=id_coord[1]
    )

    Qs=np.arange(0.1,1,0.001)

    QDM = sdba.QuantileDeltaMapping.train(
        ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr"),
        mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr, 
        nquantiles=Qs, 
        group="time", 
        kind="+"
    )

    EQM = sdba.EmpiricalQuantileMapping.train(
        ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr"),
        mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr, 
        nquantiles=Qs, 
        group="time", 
        kind="+"
    )
    if REF=="SPHERA":
        if hasattr(mod_adjust_sc,'surface') :
            mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest").drop_vars(['surface'])
            mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest").drop_vars(['surface'])
        else:
            mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest")
            mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest")
        
        list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).drop_vars(['longitude','latitude']).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
        list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).drop_vars(['longitude','latitude']).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
    elif REF=="STATIONS":
        mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                        extrapolation="constant", interp="nearest")
        mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                        extrapolation="constant", interp="nearest")
        
        list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
        list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
        
        if xy_[0].shape[0] > 0:
            for id_coord in xy_:
                if REF == "SPHERA":
                    lon=max_sta.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                    lat=max_sta.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                elif REF == "STATIONS":
                    lon=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                    lat=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
            
                na_sta_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').isel(time=mod_adjust['time.season'].isin(seas))
                list_xr_qdm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])
                list_xr_eqm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])

ds_adj_qdm=xr.combine_by_coords(list_xr_qdm)
ds_adj_eqm=xr.combine_by_coords(list_xr_eqm)

if not os.path.exists(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/QDM/{mdl}/pr/"):
    os.makedirs(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/QDM/{mdl}/pr/")

if not os.path.exists(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/EQM/{mdl}/pr/"):
    os.makedirs(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/EQM/{mdl}/pr/")


for YR in np.unique(ds_adj_eqm['time.year']):
    ds_adj_eqm.isel(time=ds_adj_eqm['time.year'].isin(YR)).to_netcdf(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/EQM/{mdl}/pr/{YR}_{mdl}_CORR_{MODEL_TRAIN}_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")    
    ds_adj_qdm.isel(time=ds_adj_eqm['time.year'].isin(YR)).to_netcdf(f"{PATH_BIAS_CORRECTED}/{MODEL_TRAIN}/QDM/{mdl}/pr/{YR}_{mdl}_CORR_{MODEL_TRAIN}_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")


# if __name__ == "__main__":

#     range_yrs=get_range_yrs(mdl)
#     if (range_yrs.shape[0] % 2 != 0):
#         print("Odd number of years") 

#     if SPLIT == "SEQUENTIAL":
#         yrs_train = range_yrs[:range_yrs.shape[0]//2]
#         yrs_valid = range_yrs[range_yrs.shape[0]//2:]
#     elif SPLIT == "RANDOM":
#         yrs_train = np.sort(np.random.choice(np.arange(range_yrs[0],range_yrs[-1]),size=int(range_yrs.shape[0]//2),replace=False))
#         yrs_valid = np.sort(np.setdiff1d(np.arange(range_yrs[0],range_yrs[-1]),yrs_train))

#     #LOAD REFERENCE DATA
#     if  REF == "SPHERA":
#         ref_train=get_reference_data(REF,yrs_train)

#     elif REF == "STATIONS":
#         ref_train,max_sta=get_reference_data(REF,yrs_train)
#     print(f"Reference data loaded")

#     """
#     Get coordinates depending on the `AREA`
#     """
#     if AREA =='triveneto':
#         xy,xy_=get_list_coords(AREA,ref_train,max_sta)
#     elif AREA == "northern_italy":
#         xy,xy_=get_list_coords(AREA,ref_train,ref_train)

#     """
#     LOAD THE DATA FOR EACH MODEL

#     """
#     print(f"Running:{mdl} {datetime.today().strftime('%d/%m/%Y %H:%M:%S')}")

#     list_mod_train =[glob(f"{PATH_COMMON_DATA}/{MODEL_TRAIN}/{mdl}/CPM/pr/*{year}*") for year in yrs_train]
#     list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/{MODEL_TRAIN}/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
#     # x=xr.open_mfdataset(list_mod_adjust[0]).load()

#     mod_train,mod_adjust=get_model_data(mdl,yrs_train,list_mod_train,list_mod_adjust)

#     print(f"Model data loaded")
