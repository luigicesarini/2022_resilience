#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
"""
1000 sample bootstrap, resmapling the 5 years of validation without replacement

Then,
if the 2.5th percentile is > 0 | 97.5th percentile is <0:
    the bias is statistically significant
"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import numba
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
from timeit import default_timer as timer

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 

from resilience.utils import *

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
parser.add_argument("-bc","--bias_correction",
                    help="Metric to choose",
                    required=True,default="EQM",
                    choices=["EQM","QDM"]  
                    )
parser.add_argument("-ref","--reference",
                    help="Correction made on which reference dataset",
                    required=True,default="STATIONS",
                    choices=["STATIONS","SPHERA"]  
                    )
parser.add_argument("-ro","--bootstrap_original",
                    help="What to resample",
                    action="store_true",
                    )

parser.add_argument("-rb","--bootstrap_adjusted",
                    help="What to resample",
                    action="store_true",
                    )

parser.add_argument("-b","--boxplots",
                    help="Plot boxplots of bias",
                    action="store_true")

parser.add_argument("-M","--maps",
                    help="Plot maps of bias",
                    action="store_true",
                    )
parser.add_argument("-nq","--number_quantile",
                    help="Number of quantile to adjust",
                    required=True,default=10000
                    )
args = parser.parse_args()


WH=False
lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

REF=args.reference
DEBUG=False
if DEBUG:
    ADJUST = 'VALIDATION'
    VAR='mw'
    REF='SPHERA'
    NQUANT=1000
    BOXPLOTS=True
    MAPS=True
    metrics='q'
    BOOT_ADJ=True
    BOOT_ORI=True
    BC_TECHN="EQM"

else:
    ADJUST = args.adjust_period
    NQUANT=args.number_quantile
    BOXPLOTS=args.boxplots
    MAPS=args.maps
    metrics=args.metric
    BOOT_ORI=args.bootstrap_original
    BOOT_ADJ=args.bootstrap_adjusted
    BC_TECHN=args.bias_correction

mask=xr.open_dataset("data/mask_stations_nan_common.nc")

if __name__=="__main__":
    
    SEASONS=['DJF','JJA'] #MAM 
    # q_999_st=xr.load_dataset("/home/lcesarini/2022_resilience/output/JJA/stations_q_VALIDATION_10000.nc")
    
    for seas in SEASONS: 
        mod=xr.load_dataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_200001010030-200012312330.nc")
        sphera  = [xr.open_mfdataset(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/{VAR}/*{yr}*") for yr in np.arange(2000,2010)]
        sta_val = xr.concat(sphera,dim='time') 
        sta_val['longitude']=mod.lon.values
        sta_val['latitude']=mod.lat.values
        sta_val=sta_val.isel(time=sta_val['time.season'].isin(seas))

        BC=BC_TECHN
        MODEL=""
        if VAR=='pr':
            list_mdl=["MOHC","ETH","CNRM","KNMI","HCLIMcom","KIT","CMCC","ICTP"]
        elif VAR=='mw':
            list_mdl=["ETH","CNRM","KNMI","HCLIMcom","KIT","CMCC","ICTP"]

        ds_mod_ori = []
        ds_mod_adj = []
        
        if BOOT_ORI:
                
            for mdl in list_mdl:
                """
                Approximate time for the loop:
                ~ 6 minutes
                """
                list_mod=[]

                if mdl == "ETH":
                    # from resilience.utils.fix_year_eth import fix_eth
                    eth=fix_eth()
                    
                    if ADJUST=="TRAIN":
                        mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
                        mod_adjust=mod_train
                    elif ADJUST=="VALIDATION":
                        mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()

                    ds_mod_ori.append(mod_adjust.pr.isel(time=mod_adjust["time.season"].isin(seas)))
                elif mdl == "ICTP":
                    if ADJUST=="TRAIN":
                        for year in np.arange(2000,2004):
                            list_mod=list_mod+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                    elif ADJUST=='VALIDATION':
                        for year in np.arange(2006,2010):
                            list_mod=list_mod+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                    
                    ds_mod_ori.append(xr.open_mfdataset(list_mod).pr.isel(time=xr.open_mfdataset(list_mod)["time.season"].isin(seas)).load())
                else:
                    if ADJUST=="TRAIN":
                        for year in np.arange(2000,2005):
                            list_mod=list_mod+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                    elif ADJUST=='VALIDATION':
                        for year in np.arange(2005,2010):
                            list_mod=list_mod+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")

                    ds_mod_ori.append(xr.open_mfdataset(list_mod).pr.isel(time=xr.open_mfdataset(list_mod)["time.season"].isin(seas)).load())

            ds_mod_dict_ori={mdl:ds_mod_ori[i] for i,mdl in enumerate(list_mdl)}
            print("Done loading the originals")

        if BOOT_ADJ:

            for mdl in list_mdl:
                """
                Approximate time for the loop:
                ~ 20 seconds
                """
                list_mod_adjust=[]
                
                if ADJUST=="TRAIN":
                    list_mod_adjust=f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{REF}**2000**{seas}**{NQUANT}*"
                elif ADJUST=='VALIDATION':
                    list_mod_adjust=f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{REF}**2005**{seas}**{NQUANT}*"

                ds_mod_adj.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr.\
                                                isel(time=xr.open_mfdataset(list_mod_adjust)["time.season"].isin(seas)),
                                                sta_val).load())

            ds_mod_dict_adj={mdl:ds_mod_adj[i] for i,mdl in enumerate(list_mdl)}
            print(f"Done loading the adjusted {BC}")

        # get_triveneto(xr.open_mfdataset(list_mod_adjust).pr.isel(time=xr.open_mfdataset(list_mod_adjust)["time.season"].isin(seas)),sta_val).load().max()
        # sta_datarray=sta_val.isel(time=sta_val["time.year"].isin(np.arange(2005,2010))).pr
        # sta_val['time','lat','lon']
        # ds_mod_dict_adj['ETH'] - sta_datarray.transpose('time','lat','lon')
        
        # ds_mod_dict_adj['ETH'].shape == sta_datarray.transpose('time','lat','lon').values.shape
        
        """
        START RESAMPLE BY YEAR
        """
        for iteration in tqdm(np.arange(1000)):
            sample_yr=np.random.choice(np.arange(2005,2010),5,replace=True)

            q_999_sta=[]
            ll_sta=[sta_val.pr.isel(time=sta_val['time.year'].isin(YR)).values for YR in sample_yr]


            if REF == "SPHERA":
                ll_sta=[np.moveaxis(x,0,2) for x in ll_sta]
            arr_sta=np.concatenate(ll_sta,axis=2)
            q_999_sta.append(np.nanquantile(arr_sta,q=0.999,axis=2))
            sta_999=np.stack(q_999_sta)[0,:,:]

            sta_999_ds = xr.Dataset(
                data_vars=dict(
                    pr=(["lat", "lon"], sta_999),
                ),
                coords=dict(
                    lon=q_999_st.lon,
                    lat=q_999_st.lat,
                ),
                attrs=q_999_st.attrs
            )
            q_999_mdl_ori=[]
            q_999_mdl_adj=[]
            for mdl in list_mdl:
                "~ 1 minutes"
                if BOOT_ORI:
                    ll_mdl_ori=[ds_mod_dict_ori[mdl].\
                                isel(time=ds_mod_dict_ori[mdl]['time.year'].isin(YR)).values for YR in sample_yr]
                    arr_mdl_ori=np.concatenate(ll_mdl_ori,axis=0)
                    q_999_mdl_ori.append(np.quantile(arr_mdl_ori,q=0.999,axis=0))
                
                if BOOT_ADJ:
                    ll_mdl_adj=[ds_mod_dict_adj[mdl].\
                                isel(time=ds_mod_dict_adj[mdl]['time.year'].isin(YR)).values for YR in sample_yr]
                
            
                    assert ll_mdl_adj[0].shape[2] > 100,"Check shape order"
                    arr_mdl_adj=np.concatenate(ll_mdl_adj,axis=2)

                    q_999_mdl_adj.append(np.quantile(arr_mdl_adj,q=0.999,axis=2))
            
            if BOOT_ORI:
                ens_999_ori=np.stack(q_999_mdl_ori).mean(axis=0)

                ens_999_ds_ori = xr.Dataset(
                    data_vars=dict(
                        pr=(["lat", "lon"], ens_999_ori),
                    ),
                    coords=dict(
                        lon=q_999_st.lon,
                        lat=q_999_st.lat,
                    ),
                    attrs=q_999_st.attrs
                )

                "ORIGINAL"
                (((ens_999_ds_ori.pr - sta_999_ds.pr) / sta_999_ds.pr) * 100).\
                to_netcdf(f"output/bootstrap/ORI/{seas}/bias_heavy_prec_{REF}_{seas}_{iteration}.nc")

            if BOOT_ADJ:
                ens_999_adj=np.stack(q_999_mdl_adj).mean(axis=0)

                ens_999_ds_adj = xr.Dataset(
                    data_vars=dict(
                        pr=(["lat", "lon"], ens_999_adj),
                    ),
                    coords=dict(
                        lon=q_999_st.lon,
                        lat=q_999_st.lat,
                    ),
                    attrs=q_999_st.attrs
                )

                "ADJUSTED"
                (((ens_999_ds_adj.pr - sta_999_ds.pr) / sta_999_ds.pr) * 100).\
                to_netcdf(f"output/bootstrap/{BC}/{seas}/bias_heavy_prec_{BC}_{REF}_{seas}_{iteration}.nc")
            
            # bc_ori=EvaluatorBiasCorrection(sta_999_ds.pr,ens_999_ds_ori.pr)
            # bc_eqm=EvaluatorBiasCorrection(sta_999_ds.pr,ens_999_ds_adj.pr)

            # sns.kdeplot(bc_ori.PBias()[~np.isnan(bc_ori.PBias())],label='ori')
            # sns.kdeplot(bc_eqm.PBias()[~np.isnan(bc_eqm.PBias())],label='eqm')
            # plt.legend()
            # plt.savefig("/home/lcesarini/prr.png")
            # plt.close()

            # sns.kdeplot((((ens_999_ds_ori.pr - sta_999_ds.pr) / sta_999_ds.pr) * 100).values.reshape(-1))
            # sns.kdeplot((((ens_999_ds_adj.pr - sta_999_ds.pr) / sta_999_ds.pr) * 100).values.reshape(-1))
            # plt.savefig("/home/lcesarini/prr.png")
            # plt.close()
