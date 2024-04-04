#! /mnt/data/lcesarini/miniconda3/miniconda3/envs/colorbar/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
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

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")
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
parser.add_argument("-a","--area",
                    help="Area over which the bias correction is applied",
                    default='triveneto',choices=["triveneto","northern_italy"]
                    )
parser.add_argument("-db","--debug",
                    help="Debugging some error",
                    action="store_true"
                    )
args = parser.parse_args()

lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 
# BM  = str(sys.argv[1])
# MDL = str(sys.argv[2])
REF = args.reference
ADJUST = args.period
SEASONS=['SON','DJF','MAM','JJA'] #,
SPLIT= args.split
AREA=args.area
DEBUG=args.debug
print(f"debug_:{DEBUG}")
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
    list_ref_train=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{year}*") for year in yrs_train]


if (ADJUST=="TRAIN") & (REF == "SPHERA"):
    list_ref_adjust=list_ref_train
elif (ADJUST=="VALIDATION") & (REF == "SPHERA"):
    list_ref_adjust=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{year}*") for year in yrs_valid]

# mask_sa=xr.open_dataset("output/mask_study_area.nc")
if REF =="SPHERA":
    ref_train=xr.open_mfdataset([item for list in list_ref_train for item in list]).load()
    ref_adjust=xr.open_mfdataset([item for list in list_ref_adjust for item in list]).load()
    sta_adjust=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_train])
    max_sta=np.nanmax(sta_adjust.pr.values[:,:,:],axis=2)
elif REF == "STATIONS":
    ref_train=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_train]).load()
    if ADJUST=="TRAIN":
        ref_adjust=ref_train
    elif ADJUST=="VALIDATION":
        ref_adjust=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_valid]).load()
    all_sta=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
    max_sta=np.nanmax(all_sta.pr.values[:,:,:],axis=2)

if AREA =='triveneto':
    x,y=np.where((max_sta > 0))
    x_,y_=np.where(np.isnan(max_sta))

    xy=np.concatenate([y.reshape(-1,1),x.reshape(-1,1)],axis=1)
    xy_=np.concatenate([y_.reshape(-1,1),x_.reshape(-1,1)],axis=1)
    assert xy.shape[0] == 172, f"Number of cells, different from number {xy.shape[0]}"
else:
    """
    FOR CANCIA BASINS 
    """
    lat=[
        46.41240788,46.43990707, 46.46740627,46.49490547
    ]

    lon=[
        12.21983337, 12.24733257, 12.27483177
    ]
    
    list_xy=create_list_coords(lon,lat)
    
    ref_train=ref_train.sel(longitude=np.unique(list_xy[:,0]),latitude=np.unique(list_xy[:,1]),method="nearest")
    ref_adjust=ref_adjust.sel(longitude=np.unique(list_xy[:,0]),latitude=np.unique(list_xy[:,1]),method="nearest")
    # ref_train_tri=crop_to_extent(ref_train,
    #                          sta_adjust.lon.min().item()-0.01,sta_adjust.lon.max().item()+0.01,
    #                          sta_adjust.lat.min().item()-0.01,sta_adjust.lat.max().item()+0.01)
    # ref_adjust_tri=crop_to_extent(ref_train,
    #                          sta_adjust.lon.min().item()-0.01,sta_adjust.lon.max().item()+0.01,
    #                          sta_adjust.lat.min().item()-0.01,sta_adjust.lat.max().item()+0.01)

    # ref_train_tri=ref_train
    # ref_adjust_tri=ref_adjust

    xy=create_list_coords(np.arange(ref_train.longitude.values.shape[0]).reshape(-1,1),
                          np.arange(ref_train.latitude.values.shape[0]).reshape(-1,1))
        
    xy_=np.where(np.isnan(np.nanmax(ref_train.pr,axis=0)))
    

#sta_adjust.sel(lon=11.25736141204834,lat=45.25744152069092).pr
"""
DO THE BIAS CORRECTION FOR EACH CELL THAT CONTAINS A STATION, THUS:
- Comparison between Model and station
"""

# list_mdl=["CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","ETH"] #
list_mdl=["KNMI"] #
#

for mdl in tqdm(list_mdl): 
    list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
    x=xr.open_mfdataset(list_mod_adjust[0]).load()


    # print(x.lat[1]-x.lat[0])
    if mdl == "ETH":
        from resilience.utils.fix_year_eth import fix_eth
        eth=fix_eth()
        # mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
        mod_train=eth.sel(time=eth['time.year'].isin(yrs_train)).load()
        if ADJUST=="TRAIN":
            mod_adjust=mod_train
        elif ADJUST=="VALIDATION":
            # mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()
            mod_adjust=eth.sel(time=eth['time.year'].isin(yrs_valid)).load()
        
    else:   
        list_mod_train=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_train]
        if ADJUST == "TRAIN":
            list_mod_adjust=list_mod_train
            mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
            mod_adjust=mod_train
        elif ADJUST == "VALIDATION":
            list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
            mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
            mod_adjust=xr.open_mfdataset([item for list in list_mod_adjust for item in list]).load()

    # mod_adjust.lat.values[:2] == ref_train.latitude.values[:2]
    # print(f"{mdl}")
    # print(f"dates calibration:{mod_train.time.values[0]}-{mod_train.time.values[-1]}")
    # print(f"dates adjustment:{mod_adjust.time.values[0]}-{mod_adjust.time.values[-1]}")


    # for seas in tqdm(SEASONS,total=len(SEASONS)): 
    for seas in SEASONS: 
        list_xr_qdm=[]
        list_xr_eqm=[]

        """
        CHECK A BIG CHUNK OF CODE TO DEBUG HEAVY PREC for DJF
        """

        for id_coord in xy:
            print(id_coord)
            if REF == "SPHERA":
                if AREA=="triveneto":
                    lon=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                    lat=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                    ref_train_sc=ref_train.sel(longitude=lon,latitude=lat,method='nearest').\
                        where(ref_train.sel(longitude=lon,latitude=lat,method='nearest').pr > 0.1).assign_attrs(units="mm/hr")
                    ref_adjust_sc=ref_adjust.sel(longitude=lon,latitude=lat,method='nearest').\
                        where(ref_adjust.sel(longitude=lon,latitude=lat,method='nearest').pr > 0.1).assign_attrs(units="mm/hr")
                elif AREA=="northern_italy":
                    lon,lat=id_coord[0],id_coord[1]
                    
                    ref_train_sc=ref_train.isel(longitude=lon,latitude=lat).\
                        where(ref_train.isel(longitude=lon,latitude=lat).pr > 0.1).assign_attrs(units="mm/hr")
                    ref_adjust_sc=ref_adjust.isel(longitude=lon,latitude=lat).\
                        where(ref_adjust.isel(longitude=lon,latitude=lat).pr > 0.1).assign_attrs(units="mm/hr")
            elif REF == "STATIONS":
                lon=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                lat=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                ref_train_sc=ref_train.sel(lon=lon,lat=lat,method='nearest').\
                    where(ref_train.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
                ref_adjust_sc=ref_adjust.sel(lon=lon,lat=lat,method='nearest').\
                    where(ref_adjust.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
            #Filter only WET hours for single cell
            if REF == "SPHERA":                
                mod_train_sc=mod_train.isel(lon=lon,lat=lat).\
                    where(mod_train.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
                # sta_train_sc=sta_train.where(sta_train.pr > 0.2).assign_attrs(units="mm/hr").load()

                mod_adjust_sc=mod_adjust.isel(lon=lon,lat=lat).\
                    where(mod_adjust.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
            else:
                mod_train_sc=mod_train.sel(lon=lon,lat=lat,method='nearest').\
                    where(mod_train.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
                # sta_train_sc=sta_train.where(sta_train.pr > 0.2).assign_attrs(units="mm/hr").load()

                mod_adjust_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').\
                    where(mod_adjust.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")

            #WRITE A CONDITION TO CHECK IF DATA ARE IN CHUNKS"""
            #ASSIGN UNITS
            # Qs=np.arange(0,1,0.0001) 
            Qs=np.arange(0.1,1,0.001)

            QDM = sdba.QuantileDeltaMapping.train(
            # QM = sdba.EmpiricalQuantileMapping.train(
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
                mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                                extrapolation="constant", interp="nearest").drop_vars(['surface'])
                mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                                extrapolation="constant", interp="nearest").drop_vars(['surface'])
                list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).drop_vars(['longitude','latitude']).\
                                expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
                list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).drop_vars(['longitude','latitude']).\
                                expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
            elif REF=="STATIONS":
                mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                                extrapolation="constant", interp="nearest")
                # mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                #                                 extrapolation="constant", interp="nearest")
                mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                                extrapolation="constant", interp="nearest")
                
                # xx=mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr
                # xr.where(~np.isnan(xx),xx,0)
                # x1=EQM.adjust(mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr,
                #                                 extrapolation="constant", interp="nearest")
                
                # x2=EQM.adjust(xr.where(~np.isnan(xx),xx,0).assign_attrs(units="mm/hr"),
                #                                 extrapolation="constant", interp="nearest")
                

                # x1.max() 
                # x2.max()
                ref_train_sc  = get_season(ref_train_sc,seas)
                ref_adjust_sc = get_season(ref_adjust_sc,seas)
                mod_train_sc  = get_season(mod_train_sc,seas)
                mod_adjust_sc = get_season(mod_adjust_sc,seas)



            if DEBUG:
                list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).\
                                expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
                list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).\
                                expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
                
                if xy_[0].shape[0] > 0:
                    for id_coord in xy_:
                        if REF == "SPHERA":
                            lon=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                            lat=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                        elif REF == "STATIONS":
                            lon=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                            lat=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                    
                        na_sta_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').isel(time=mod_adjust['time.season'].isin(seas))
                        list_xr_qdm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])
                        list_xr_eqm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])

            ds_adj_qdm=xr.combine_by_coords(list_xr_qdm)
            ds_adj_eqm=xr.combine_by_coords(list_xr_eqm)
            
            if not os.path.exists(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/"):
                os.makedirs(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/")

            if not os.path.exists(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/"):
                os.makedirs(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/")

            # if ADJUST=="TRAIN":
            #     ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
            #     ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
            # elif ADJUST=="VALIDATION":
            ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")
            ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")

