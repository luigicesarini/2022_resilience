#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
"""
Compute metrics for bias corrected wind
1) Mean wind speed
2) 99th percentile of wind
3) Winds over a threshold (i.e., 5m/s)
"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
# import xarray.ufuncs as xu 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 

from resilience.utils import *

"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-ap","--adjust_period",
                    help="Which period to adjust the data",
                    required=True,default="VALIDATION",
                    choices=["TRAIN","VALIDATION"]  
                    )

parser.add_argument("-nq","--number_quantile",
                    help="Number of quantile to adjust",
                    required=True,default=1000
                    )

parser.add_argument("-rp","--reference_dataset",
                    help="Which dataset to use as reference",
                    required=True,default="STATIONS",
                    choices=["STATIONS","SPHERA"]  
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


args = parser.parse_args()

WH=False
lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

REF=args.reference_dataset #"STATIONS"
ADJUST = args.adjust_period #"VALIDATION"
NQUANT=args.number_quantile
SPLIT=args.split
AREA=args.area
# REF="STATIONS" #"STATIONS"
# ADJUST = "VALIDATION" #"VALIDATION"
# NQUANT="1000"
# SPLIT="SEQUENTIAL"
print(REF,ADJUST,NQUANT,SPLIT)

mask=xr.open_dataset("data/mask_stations_nan_common.nc")

if __name__=="__main__":

    if (ADJUST=="TRAIN") & (SPLIT == "SEQUENTIAL"):
        st,en=2000,2005
    elif (ADJUST=="TRAIN") & (SPLIT == "RANDOM"):
        years = np.unique(xr.open_dataset(f"{PATH_BIAS_CORRECTED}/EQM/CNRM/mw/CNRM_CORR_{REF}_DJF_Q1000_{SPLIT}_VALIDATION.nc").time.dt.year.values)
    elif (ADJUST=="VALIDATION") & (SPLIT == "SEQUENTIAL"):
        st,en=2005,2010
    elif (ADJUST=="VALIDATION") & (SPLIT == "RANDOM"):
        years = np.unique(xr.open_dataset(f"{PATH_BIAS_CORRECTED}/EQM/ETH/pr/ETH_CORR_STATIONS_DJF_Q1000_RANDOM_VALIDATION.nc").time.dt.year.values)

    if SPLIT == "SEQUENTIAL":
        list_ref_val=[glob(f"{PATH_COMMON_DATA}/reanalysis/{REF}/mw/*{year}*") for year in np.arange(st,en)]
    elif SPLIT == "RANDOM":
        list_ref_val=[glob(f"{PATH_COMMON_DATA}/reanalysis/{REF}/mw/*{year}*") for year in years]


    list_mdl=["CNRM","KNMI","ICTP","HCLIMcom","KIT","CMCC","ETH"]

    for mdl in tqdm(list_mdl, total=len(list_mdl)): 

    # for mdl in ["ICTP" , "CNRM"]: 
    #     for year in np.arange(2000,2010):
    #         list_mod_val=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") ]
    #         mod_val=xr.open_mfdataset([item for list in list_mod_val for item in list]).load()
    #         print(f"{mdl}_{year}: {mod_val.mw.max().item():.2f}")


        for SEAS in tqdm(['JJA'],total=4): #

            if mdl =='ICTP':
                if SPLIT == "SEQUENTIAL":
                    list_mod_val=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in np.arange(st,en)]
                    mod_val=xr.open_mfdataset([item for list in list_mod_val for item in list]).load()
                    mod_val=mod_val.isel(time=mod_val.time.dt.year.isin(np.arange(st,en)))
                elif SPLIT == "RANDOM":
                    list_mod_val=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in years]
                    mod_val=xr.open_mfdataset([item for list in list_mod_val for item in list]).load()
                    mod_val=mod_val.isel(time=mod_val.time.dt.year.isin(np.arange(st,en)))
            else:
                if SPLIT == "SEQUENTIAL":
                    list_mod_val=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in np.arange(st,en)]
                    mod_val=xr.open_mfdataset([item for list in list_mod_val for item in list]).load()
                elif SPLIT == "RANDOM":
                    list_mod_val=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/mw/*{year}*") for year in years]
                    mod_val=xr.open_mfdataset([item for list in list_mod_val for item in list]).load()

            ref_val=xr.open_mfdataset([item for list in list_ref_val for item in list]).load()

            # mod_eqm=xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/*{REF}*").load()
            mod_eqm=xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/mw/*{REF}**{SEAS}**{NQUANT}**{SPLIT}**{AREA}*").load()
            # xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/*SPHERA*")
            # mod_qdm=xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/*{REF}*").load()
            mod_qdm=xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/mw/*{REF}**{SEAS}**{NQUANT}**{SPLIT}**{AREA}*").load()
            
            if SPLIT== "SEQUENTIAL":
                mod_eqm=mod_eqm.isel(time=mod_eqm.time.dt.year.isin(np.arange(st,en)))
                mod_qdm=mod_qdm.isel(time=mod_qdm.time.dt.year.isin(np.arange(st,en)))

                ref_val=get_season(ref_val.isel(time=ref_val.time.dt.year.isin(np.arange(st,en))),season=SEAS)
                mod_val=get_season(mod_val.isel(time=mod_val.time.dt.year.isin(np.arange(st,en))),season=SEAS)


            name_models=["Reference","Model","Model_EQM","Model_QDM"]
            array_model=[ref_val,mod_val,mod_eqm,mod_qdm]
        
            dict_metrics={}
            for name,mdl_arr in zip(name_models,array_model):
                    dict_0={name:compute_metrics_wind(mdl_arr,quantile=0.999,threshold=5)}
                    
                    dict_metrics.update(dict_0)

            for idx,metrica in enumerate(["mean","q","thr"]):
                if (REF =="SPHERA") & (not os.path.exists(f"/home/lcesarini/2022_resilience/output/mw/{SEAS}/{REF}_{metrica}_{NQUANT}_{SPLIT}_{ADJUST}_{AREA}.nc")):
                    dict_metrics['Reference'][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/mw/{SEAS}/{REF}_{metrica}_{NQUANT}_{SPLIT}_{ADJUST}_{AREA}.nc")
                dict_metrics['Model'][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/mw/{SEAS}/{mdl}_{metrica}_biased_{REF}_{NQUANT}_{SPLIT}_{ADJUST}_{AREA}.nc")
                dict_metrics['Model_EQM'][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/mw/{SEAS}/{mdl}_{metrica}_EQM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}_{AREA}.nc")
                dict_metrics['Model_QDM'][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/mw/{SEAS}/{mdl}_{metrica}_QDM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}_{AREA}.nc")

        # dict_metrics.keys()
        # ((dict_metrics['Reference'][0] - get_triveneto(dict_metrics['Model'][0])) / get_triveneto(dict_metrics['Model'][0])).plot.pcolormesh(levels=10,cmap='RdBu')
        # plt.savefig(f"figures/diff_{mdl}_{SEAS}_{ADJUST}.png")
        # plt.close()  

        # if 'longitude' in list(dict_metrics['Reference'][0].coords):
        #     dict_metrics['Reference'][0].sel(longitude=lon,latitude=lat,method='nearest').values.reshape(-1).shape
        #     dict_metrics['Reference'][1].sel(longitude=lon,latitude=lat,method='nearest')
        # else:
        #     dict_metrics['Reference'][0].sel(lon=lon,lat=lat,method='nearest').values.reshape(-1).shape
        #     dict_metrics['Reference'][1].sel(lon=lon,lat=lat,method='nearest')
        
        # dict_metrics['Stations'][0].sel(lon=lon,lat=lat,method='nearest').values.reshape(-1).shape
        # dict_metrics['Stations'][1].sel(lon=lon,lat=lat,method='nearest').values.reshape(-1).shape

        # (~np.isnan(dict_metrics['Model_EQM'][1]).values.reshape(-1)).sum()
        
        #Extract pair of lon,lat where tecdf is not nan
        # lon_lat=np.array([dict_metrics['Model_EQM'][1].lon,dict_metrics['Model_EQM'][1].lat]).T
        # lon_lat


        
        # plot_panel_rotated(
        #     figsize=(26,7),
        #     nrow=1,ncol=3,
        #     list_to_plot=[dict_metrics['Stations'][3],dict_metrics['Model_QDM'][3],dict_metrics['Model_EQM'][3]],
        #     name_fig=f"bias_corrected_mohc",
        #     list_titles=["Station","QDM","EQM"],
        #     levels=[lvl_q,lvl_q,lvl_q],
        #     suptitle=f"Heavy Precipitation (mm)",
        #     name_metric=["[mm/h]","[mm/h]","[mm/h]"],
        #     SET_EXTENT=False,
        #     cmap=[cmap_q,cmap_q,cmap_q]
        # )
        

        # mdl_sta = (dict_metrics['Model'][3] * mask.mask).isel(lon=dict_metrics['Model'][3].lon.isin(dict_metrics['Stations'][3].lon),
        #                                                       lat=dict_metrics['Model'][3].lat.isin(dict_metrics['Stations'][3].lat))

        # if 'longitude' in list(dict_metrics['Reference'][0].coords):
        #     dict_metrics['Reference'][3].coords['longitude'] = dict_metrics['Model'][3].lon.values
        #     dict_metrics['Reference'][3].coords['latitude'] = dict_metrics['Model'][3].lat.values
        # else:    
        #     dict_metrics['Reference'][3].coords['lon'] = dict_metrics['Model_EQM'][3].lon.values
        #     dict_metrics['Reference'][3].coords['lat'] = dict_metrics['Model_EQM'][3].lat.values

        # if 'longitude' in list(dict_metrics['Reference'][0].coords):
        #     ref_q=dict_metrics['Reference'][3].rename({'longitude':'lon','latitude':'lat'})
        # else:
        #     ref_q=dict_metrics['Reference'][3]








        # fig=plt.figure(figsize=(16,12))
        # ax=plt.axes(projection=ccrs.PlateCarree())
        # pcm=dict_metrics['Stations'][3].rio.write_crs("epsg:4326").plot.pcolormesh(levels=lvl_q,
        #                             cmap=cmap_q,ax=ax,
        #                             add_colorbar=False)
        # shp_triveneto.boundary.plot(edgecolor='red',ax=ax,linewidths=1.5)
        # ax.set_title(f"Bias Corrected Value of MOHC",fontsize=20)
        # ax.set_extent([10.39,13.09980774,44.70745754,47.09988785])
        # gl = ax.gridlines(
        #         draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
        #     )
        # gl.xlocator = mpl.ticker.FixedLocator([10.5, 11.5, 12.5])
        # gl.ylocator = mpl.ticker.FixedLocator([45, 46, 47])
        # gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
        # gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
        # cbar=fig.colorbar(pcm, ax=ax, 
        #                     extend='both', 
        #                     orientation='vertical',
        #                     shrink=1,
        #                     pad = 0.075)
        # cbar.ax.tick_params(labelsize=30)
        # cbar.ax.set_ylabel('[%]',fontsize=25,rotation=0)
        # fig.suptitle("Heavy precipitation (p99.9)",fontsize=30)
        # ax.add_feature(cfeature.BORDERS)
        # ax.add_feature(cfeature.LAKES)
        # ax.add_feature(cfeature.RIVERS)
        # ax.coastlines()
        # plt.savefig(f"figures/bias_corrected_mohc.png")
        # plt.close()


        # mask_bc=xr.where(np.isnan(dict_metrics['Stations'][3]),np.nan,1).drop_vars(['quantile'])
        # mask_bc.where((mask_bc.lon >= 10.459884643554688) & (mask_bc.lon <= 12.934812545776367) &\
        #               (mask_bc.lat >= 44.92745113372803) & (mask_bc.lat <= 46.96239185333252), drop=True)
        # dict_metrics['Model_EQM'][3].lon.values.max(),dict_metrics['Model_EQM'][3].lon.values.min()
        # dict_metrics['Model_EQM'][3].lat.values.max(),dict_metrics['Model_EQM'][3].lat.values.min()
        # dict_metrics['Stations'][3].lon.values


    # SEAS="DJF"
    # REF="STATIONS"
    # NQUANT=1000
    # SPLIT="SEQUENTIAL"
    # ADJUST="VALIDATION"

    # for metrica in ["f","i","v","q"]:
    #     list_bia=[xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_{metrica}_biased_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc") for mdl in list_mdl]
    #     list_eqm=[xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_{metrica}_EQM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc") for mdl in list_mdl]
    #     list_qdm=[xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_{metrica}_QDM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc") for mdl in list_mdl]

    #     ens_bia=xr.concat(list_bia,dim='model').mean(dim='model')
    #     ens_eqm=xr.concat(list_eqm,dim='model').mean(dim='model')
    #     ens_qdm=xr.concat(list_qdm,dim='model').mean(dim='model')
        
    #     ens_bia.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_{metrica}_biased_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc")
    #     ens_eqm.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_{metrica}_EQM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc")
    #     ens_qdm.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_{metrica}_QDM_{REF}_{NQUANT}_{SPLIT}_{ADJUST}.nc")
        # sta=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/JJA/STATIONS_{metrica}.nc")



