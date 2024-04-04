#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
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
import subprocess as sp
import geopandas as gpd
import matplotlib as mpl
from random import sample
# import xarray.ufuncs as xu 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from timeit import default_timer as timer
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 

from resilience.utils import *
"""
ll_files=sp.check_output("find /home/lcesarini/2022_resilience/output/DJF  -type f -newermt '2023-09-01' -exec ls {} \;",shell=True,text=True)
ll=ll_files.splitlines()
"""
shp_triveneto = gpd.read_file("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]


"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-ap","--adjust_period",
                    help="Which period to adjust the data",
                    required=True,default="VALIDATION",
                    choices=["TRAIN","VALIDATION"]  
                    )

parser.add_argument("-m","--metric",
                    help="Metric to choose",
                    required=True,default="f",
                    choices=["f","i","v","q"]  
                    )
parser.add_argument("-s","--split",
                    help="How is the temporal data?",
                    choices=["SEQUENTIAL","RANDOM"],
                    )

parser.add_argument("-nq","--number_quantile",
                    help="Number of quantile to adjust",
                    required=True,default=10000
                    )

parser.add_argument("-b","--boxplots",
                    help="Plot boxplots of bias",
                    action="store_true"                    )

parser.add_argument("-M","--maps",
                    help="Plot maps of bias",
                    action="store_true",
                    )
args = parser.parse_args()


WH=False
lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

REF="STATIONS"
DEBUG=False
if DEBUG:
    ADJUST = 'VALIDATION'
    NQUANT=1000
    BOXPLOTS=True
    MAPS=True
    metrics='q'
    SPLIT="SEQUENTIAL"
else:
    ADJUST = args.adjust_period
    NQUANT=args.number_quantile
    BOXPLOTS=args.boxplots
    MAPS=args.maps
    SPLIT=args.split
    metrics=args.metric


if SPLIT == "RANDOM":
    ll_files=sp.check_output("find /home/lcesarini/2022_resilience/output/DJF  -type f -newermt '2023-09-09' -exec ls {} \;",shell=True,text=True)
    ll=ll_files.splitlines()
    ll

mask=xr.open_dataset("data/mask_stations_nan_common.nc")

if __name__=="__main__":
    sta_val=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()

    SEASONS=['JJA', 'SON','DJF','MAM']
    BC="QDM"
    M='q'
    for seas in SEASONS:
        list_output_mdl=glob(f"/home/lcesarini/2022_resilience/output/{seas}/*{M}_**{NQUANT}**{SPLIT}**{ADJUST}*")

        list_mdl=["MOHC","ETH","CNRM","KNMI","ICTP","HCLIMcom","KIT","CMCC"]
        # ds_mod = []

        # for mdl in tqdm(list_mdl):
        #     """
        #     Approximate time for the loop:
        #     ~ 30seconds
        #     """
        #     ds_mod.append(get_triveneto(xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}{BC}/{mdl}/pr/{mdl}_CORR_STATIONS_2005_2009_JJA_Q1000.nc").pr,sta_val).load())
        


        # ll_diurnal=[compute_quantiles_by_hour(ds_mod[idx],q=0.999,SEAS='JJA') for idx,mdl in enumerate(list_mdl)]
        # diurnal_eqm_1000=xr.concat(ll_diurnal,list_mdl).rename({'concat_dim':'model'})
        # diurnal_qdm_10000=xr.concat(ll_diurnal,list_mdl).rename({'concat_dim':'model'})
        # diurnal_qdm_1000=xr.concat(ll_diurnal,list_mdl).rename({'concat_dim':'model'})
        
        # dsds=xr.concat(ds_mod,list_mdl).rename({'concat_dim':'model'})
        
        # nc_to_csv(dsds.mean(dim='model'),f"Ensemble_mean_prec_{BC}_{seas}_1000",M='mean')
        # nc_to_csv(dsds.mean(dim='model'),f"Ensemble_q_{BC}_{seas}_1000",M='q')
        
        

        ds=[xr.open_dataset(file) for file in list_output_mdl]

        ds=xr.concat(ds,["_".join(os.path.basename(file).split("_")[:3]) if "stations" not in file else\
                        "_".join(os.path.basename(file).split("_")[:2]) for file in list_output_mdl]).\
                    rename({"concat_dim":"correction"})
        
        ds_tri=get_triveneto(ds,sta_val)

        ds_eqm=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|biased|QDM")).mean(dim='correction')
        # ds_eqm2=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|biased|QDM"))
        ds_qdm=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|EQM|biased")).mean(dim='correction')

        # nc_to_csv(ds_tri.sel(correction=f"stations_{M}"),f"Stations_{M}_{seas}_{SPLIT}",M=M)
        nc_to_csv(ds_tri.sel(correction=f"stations_{M}"),f"Stations_{M}_{seas}_{SPLIT}",M=M)
        nc_to_csv(ds_eqm,f"Ensemble_{M}_{seas}_EQM_{SPLIT}",M=M)
        nc_to_csv(ds_tri.sel(correction=ds_tri.correction.str.contains("biased")).mean(dim='correction'),f"Ensemble_{M}_{seas}_RAW_{SPLIT}",M=M)
        nc_to_csv(ds_qdm,f"Ensemble_{M}_{seas}_QDM_{SPLIT}",M=M)

        for bc in ["biased","EQM","QDM"]:
            for mdl in list_mdl:
                ds_to_print=ds_tri.sel(correction=ds_tri.correction.str.contains(bc)).sel(correction=f"{mdl}_q_{bc}")
                nc_to_csv(ds_to_print,f"{mdl}_{M}_{seas}_{bc}_{SPLIT}",M=M) 



        # sta_sli=sta_val.sel(time=slice("2005-01-01","2010-01-01"))
        # sta_sli=sta_sli.isel(time=sta_sli["time.season"].isin("JJA")).pr.values.reshape(-1)

        # gripho=xr.load_dataset(f"/mnt/data/lcesarini/gripho_3km.nc")
        # gripho_jja=gripho.isel(time=gripho['time.season'].isin(["JJA"]))
        # gripho_jja_valid=gripho_jja.isel(time=gripho_jja['time.year'].isin(np.arange(2005,2010)))

        
        # log_sta=np.log(sta_sli)
        # log_sta=log_sta[np.isfinite(log_sta)]    

        # log_gri=np.log(gripho_jja_valid.pr.values.reshape(-1))
        # log_gri=log_gri[np.isfinite(log_gri)]    
        
        # counts, bins = np.histogram(log_sta,bins=50)
        # ax=plt.axes()
        # ax.hist(log_sta,bins=np.log([0.1,0.25,0.5,1,2.5,5,10,15,25,40,60,80,100]),density=True,color='red',edgecolor='black',alpha=0.25,label='Stations')
        # ax.hist(log_gri,bins=np.log([0.1,0.25,0.5,1,2.5,5,10,15,25,40,60,80,100]),density=True,color='green',edgecolor='black',alpha=0.25,label='GRIPHO')
        
        # ax.set_xticks(np.log([0.1,0.25,0.5,1,2.5,5,10,15,25,40,60]))
        # ax.set_xticklabels([0.1,0.25,0.5,1,2.5,5,10,15,25,40,60])
        # ax.set_xlim([np.log(20),np.log(100)])
        # # ax.set_ylim([0])
        # plt.legend()
        # plt.show()

        # sns.histplot(pd.DataFrame(log_sta))
        # plt.show()