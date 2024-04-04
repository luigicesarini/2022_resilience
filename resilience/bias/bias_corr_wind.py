#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience")
from resilience.utils import *
import rasterio
import argparse
import subprocess
# import rioxarray
import numpy as np 
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from xclim import sdba
from scipy import stats
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini/2022_resilience")
from resilience.utils import *

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

parser.add_argument("-ref","--reference",
                    help="Which dataset to use for bias correction",
                    required=True,default="SPHERA",
                    choices=["SPHERA","STATIONS"]  
                    )

parser.add_argument("-s","--split",
                    help="How are the data split for fitting and evaluating the correction",
                    required=True,default="SEQUENTIAL",
                    choices=["RANDOM","SEQUENTIAL"]  
                    )
parser.add_argument("-m","--model",
                    help="Which model to correct",
                    default='triveneto',choices=["CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","ETH"]
                    )

parser.add_argument("-a","--area",
                    help="Area over which the bias correction is applied",
                    default='triveneto',choices=["triveneto","northern_italy"]
                    )
args = parser.parse_args()

lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/"
PATH_BIAS_CORRECTED = f"/mnt/beegfs/lcesarini/BIAS_CORRECTED/" 
# BM  = str(sys.argv[1])
# MDL = str(sys.argv[2])
REF = args.reference
ADJUST = args.period
SEASONS=['SON','DJF','MAM','JJA'] #,
SPLIT= args.split
AREA=args.area
mdl=args.model
DEBUG=False
if DEBUG:
    REF = "SPHERA"
    ADJUST = "VALIDATION"
    SEASONS=['JJA'] #
    SPLIT= "SEQUENTIAL"
    AREA="northern_italy"
"""
QM bias correction only on the wet hours. Otherwise the frequency of wet hours plays a crucial role
I split the fit in the first 5 years (2000-2004), and the test/adjustment on the following 5 years (2005-2009)
SPLIT BETWEEN TRAIN and ADJUST PERIOD
"""

#Create subroutine that based on SPLIT selects the years to use for bias correction
if SPLIT == "SEQUENTIAL":
    yrs_train = np.arange(2000,2005)
    yrs_valid = np.arange(2005,2010)
elif SPLIT == "RANDOM":
    yrs_train = np.sort(np.random.choice(np.arange(2000,2010),size=5,replace=False))
    yrs_valid = np.sort(np.setdiff1d(np.arange(2000,2010),yrs_train))

#LOAD DATA
if  REF == "SPHERA":
    list_ref_train=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/mw/*{year}*") for year in yrs_train]


if (ADJUST=="TRAIN") & (REF == "SPHERA"):
    list_ref_adjust=list_ref_train
elif (ADJUST=="VALIDATION") & (REF == "SPHERA"):
    list_ref_adjust=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/mw/*{year}*") for year in yrs_valid]

# mask_sa=xr.open_dataset("output/mask_study_area.nc")
ref_train=xr.open_mfdataset([item for list in list_ref_train for item in list]).load()
ref_adjust=xr.open_mfdataset([item for list in list_ref_adjust for item in list]).load()

# ref_train_tri=crop_to_extent(ref_train,
#                          sta_adjust.lon.min().item()-0.01,sta_adjust.lon.max().item()+0.01,
#                          sta_adjust.lat.min().item()-0.01,sta_adjust.lat.max().item()+0.01)
# ref_adjust_tri=crop_to_extent(ref_train,
#                          sta_adjust.lon.min().item()-0.01,sta_adjust.lon.max().item()+0.01,
#                          sta_adjust.lat.min().item()-0.01,sta_adjust.lat.max().item()+0.01)
ref_train_tri=ref_train
ref_adjust_tri=ref_adjust

xy=create_list_coords(np.arange(ref_train_tri.lon.values.shape[0]).reshape(-1,1),
                        np.arange(ref_train_tri.lat.values.shape[0]).reshape(-1,1))

xy_=np.where(np.isnan(np.nanmax(ref_train_tri.mw,axis=0)))
    

#sta_adjust.sel(lon=11.25736141204834,lat=45.25744152069092).mw
"""
DO THE BIAS CORRECTION FOR EACH CELL THAT CONTAINS A STATION, THUS:
- Comparison between Model and station
"""
# mdl="ICTP"

print(f"Running:{mdl} {datetime.today().strftime('%d/%m/%Y %H:%M:%S')}")
list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in yrs_valid]
x=xr.open_mfdataset(list_mod_adjust[0]).load()

list_mod_train=[glob(f"{PATH_COMMON_DATA}/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in yrs_train]
if ADJUST == "TRAIN":
    list_mod_adjust=list_mod_train
    mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
    mod_adjust=mod_train
elif ADJUST == "VALIDATION":
    list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in yrs_valid]
    mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
    mod_adjust=xr.open_mfdataset([item for list in list_mod_adjust for item in list]).load()

# print(f"dates calibration:{mod_train.time.values[0]}-{mod_train.time.values[-1]}")
# print(f"dates adjustment:{mod_adjust.time.values[0]}-{mod_adjust.time.values[-1]}")

ref_train['lon']=mod_adjust.lon.values
ref_train['lat']=mod_adjust.lat.values
ref_adjust['lon']=mod_adjust.lon.values
ref_adjust['lat']=mod_adjust.lat.values

for seas in tqdm(SEASONS,total=4): 
    print(seas)
    list_xr_qdm=[]
    list_xr_eqm=[]

    for id_coord in xy:
        lon,lat=id_coord[0],id_coord[1]

        ref_train_sc=ref_train.isel(lon=lon,lat=lat).assign_attrs(units="m/s")
        ref_adjust_sc=ref_adjust.isel(lon=lon,lat=lat).assign_attrs(units="m/s")

        mod_train_sc=mod_train.isel(lon=lon,lat=lat).assign_attrs(units="m/s")

        mod_adjust_sc=mod_adjust.isel(lon=lon,lat=lat).assign_attrs(units="m/s")


        #WRITE A CONDITION TO CHECK IF DATA ARE IN CHUNKS"""
        #ASSIGN UNITS
        # Qs=np.arange(0,1,0.0001) 
        Qs=np.arange(0,1,0.001)
        # Qs=np.append(Qs,np.arange(0.99,1,0.0001) )
        QDM = sdba.QuantileDeltaMapping.train(
            ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).mw.assign_attrs(units="m/s"),
            mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).mw.assign_attrs(units="m/s"), 
            nquantiles=Qs, 
            group="time", 
            kind="+"
        )

        EQM = sdba.EmpiricalQuantileMapping.train(
            ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).mw.assign_attrs(units="m/s"),
            mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).mw.assign_attrs(units="m/s"), 
            nquantiles=Qs, 
            group="time", 
            kind="+"
        )

        mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).mw.assign_attrs(units='m/s'), 
                                        extrapolation="constant", interp="nearest")
        mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).mw.assign_attrs(units='m/s'), 
                                        extrapolation="constant", interp="nearest")
        list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='mw'))
        list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).\
                        expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='mw'))
            
    # if xy_[0].shape[0] > 0:
    #     for id_coord in xy_:
    #         if REF == "SPHERA":
    #             lon=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
    #             lat=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
    #         elif REF == "STATIONS":
    #             lon=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
    #             lat=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
        
    #         na_sta_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').isel(time=mod_adjust['time.season'].isin(seas))
    #         list_xr_qdm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","mw"]])
    #         list_xr_eqm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","mw"]])

    ds_adj_qdm=xr.combine_by_coords(list_xr_qdm)
    ds_adj_eqm=xr.combine_by_coords(list_xr_eqm)
    
    if not os.path.exists(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/mw/"):
        os.makedirs(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/mw/")

    if not os.path.exists(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/mw/"):
        os.makedirs(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/mw/")

    # if ADJUST=="TRAIN":
    #     ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/mw/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
    #     ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/mw/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
    # elif ADJUST=="VALIDATION":
    ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/mw/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{args.area}.nc")
    ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/mw/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{args.area}.nc")

