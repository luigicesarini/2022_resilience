#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import time
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
from timeit import default_timer as timer
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

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
REF="SPHERA"
DEBUG=False
if DEBUG:
    ADJUST = 'VALIDATION'
    NQUANT=10000
    BOXPLOTS=True
    MAPS=True
    metrics='q'
else:
    ADJUST = args.adjust_period
    NQUANT=10000
    BOXPLOTS=args.boxplots
    MAPS=args.maps
    metrics=args.metric

mask=xr.open_dataset("data/mask_stations_nan_common.nc")
sta_val=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
BC="EQM"
MODEL=""
seas='DJF'
ADJUST="TRAIN"
list_mdl=["MOHC","ETH","CNRM","KNMI","HCLIMcom","KIT","CMCC","ICTP"]
ds_mod = []
for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    list_mod_adjust=list_mod_adjust+glob(f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{mdl}**{REF}**2000**{seas}**Q{NQUANT}*")
    ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

"""FOR TRAIN"""
ds_mod[1]['time']=ds_mod[0].time.values
ds_mod_dict_train={mdl:ds_mod[i] for i,mdl in enumerate(list_mdl)}
concat_train=xr.concat(ds_mod_dict_train.values(),list_mdl)

ds_mod = []
for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    list_mod_adjust=list_mod_adjust+glob(f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{mdl}**{REF}**2005**{seas}**Q{NQUANT}*")
    ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

"""FOR VALIDATION"""
ds_mod[1]['time']=ds_mod[0].time.values
ds_mod[7]=ds_mod[7].isel(time=ds_mod[7]['time.year'].isin(np.arange(2005,2010)))
ds_mod_dict_valid={mdl:ds_mod[i] for i,mdl in enumerate(list_mdl)}
concat_valid=xr.concat(ds_mod_dict_valid.values(),list_mdl)

[np.all(ds_mod[i].time.values == ds_mod[0].time.values) for i,mdl in enumerate(list_mdl)]

"""
RAW TRAINING
"""

ds_mod = []

for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    if mdl == "ETH":
        from resilience.utils.fix_year_eth import fix_eth
        eth=fix_eth()
        
        if ADJUST=="TRAIN":
            mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
            mod_adjust=mod_train
        elif ADJUST=="VALIDATION":
            mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()

        ds_mod.append(get_triveneto(mod_adjust.pr,sta_val))
    elif mdl == "ICTP":
        if ADJUST=="TRAIN":
            list_mod_adjust=[f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/ICTP_ECMWF-ERAINT_{year}0101000000-{year+1}0101000000.nc" for year in np.arange(2000,2005)]
        elif ADJUST=='VALIDATION':
            for year in np.arange(2006,2010):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
        
        ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())
    else:
        if ADJUST=="TRAIN":
            for year in np.arange(2000,2005):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
        elif ADJUST=='VALIDATION':
            for year in np.arange(2005,2010):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")

        ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

ds_mod_dict_raw_t={mdl:ds_mod[i] for i,mdl in enumerate(list_mdl)}
    
co_raw_valid_t=ds_mod_dict_raw_t[mdl].isel(time=ds_mod_dict_raw_t[mdl]['time.year'].isin(np.arange(2000,2005)))
print(co_raw_valid_t.shape)
co_raw_valid_t=co_raw_valid_t.isel(time=co_raw_valid_t['time.season'].isin(seas))
print(co_raw_valid_t.shape)


"""
RAW VALIDATION
"""
ds_mod = []
ADJUST = "VALIDATION"
for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    if mdl == "ETH":
        from resilience.utils.fix_year_eth import fix_eth
        eth=fix_eth()
        
        if ADJUST=="TRAIN":
            mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
            mod_adjust=mod_train
        elif ADJUST=="VALIDATION":
            mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()

        ds_mod.append(get_triveneto(mod_adjust.pr,sta_val))
    elif mdl == "ICTP":
        if ADJUST=="TRAIN":
            list_mod_adjust=[f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/ICTP_ECMWF-ERAINT_{year}0101000000-{year+1}0101000000.nc" for year in np.arange(2000,2005)]
        elif ADJUST=='VALIDATION':
            for year in np.arange(2006,2010):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
        
        ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())
    else:
        if ADJUST=="TRAIN":
            for year in np.arange(2000,2005):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
        elif ADJUST=='VALIDATION':
            for year in np.arange(2005,2010):
                list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")

        ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

ds_mod_dict_raw_v={mdl:ds_mod[i] for i,mdl in enumerate(list_mdl)}
    
co_raw_valid_v=ds_mod_dict_raw_v[mdl].isel(time=ds_mod_dict_raw_v[mdl]['time.year'].isin(np.arange(2005,2010)))
print(co_raw_valid_v.shape)
co_raw_valid_v=co_raw_valid_v.isel(time=co_raw_valid_v['time.season'].isin(seas))
print(co_raw_valid_v.shape)

co_raw_valid=xr.concat(co_raw_valid_ll,list_mdl)
q999_raw_v=co_raw_valid.quantile(q=0.999,dim='time')

co_raw_train_ll=[]
for mdl in list_mdl:
    co_raw_train=ds_mod_dict[mdl].isel(time=ds_mod_dict[mdl]['time.year'].isin(np.arange(2000,2005)))
    co_raw_train=co_raw_train.isel(time=co_raw_train['time.season'].isin(seas))
    print(co_raw_train.shape)
    co_raw_train_ll.append(co_raw_train)

ictp_2004=get_triveneto(xr.open_mfdataset(f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{mdl}**{REF}**2005**{seas}**Q{NQUANT}*").pr,sta_val).load()
ictp_2004=ictp_2004.isel(time=ictp_2004['time.year'].isin(2004))
#remove 9th element from co_raw_train_ll
co_raw_train_ll.pop(8)

co_raw_train_ll[7]=xr.concat([co_raw_train_ll[7],ictp_2004],dim='time')

co_raw_train=xr.concat(co_raw_train_ll,list_mdl)
q999_raw_t=co_raw_train.quantile(q=0.999,dim='time')

q999_train=concat_train.quantile(dim='time',q=0.999)
q999_valid=concat_valid.quantile(dim='time',q=0.999)

q999_sta_t=sta_val.isel(time=sta_val['time.year'].isin(np.arange(2000,2005)))
q999_sta_t=q999_sta_t.isel(time=q999_sta_t['time.season'].isin(seas))
q999_sta_t=q999_sta_t.quantile(dim='time',q=0.999)

q999_sta_v=sta_val.isel(time=sta_val['time.year'].isin(np.arange(2005,2010)))
q999_sta_v=q999_sta_v.isel(time=q999_stati['time.season'].isin(seas))
q999_sta_v=q999_sta_v.quantile(dim='time',q=0.999)

x_=pd.DataFrame([q999_sta_t.pr.values.reshape(-1),
                 q999_raw_t.mean(dim='concat_dim').values.reshape(-1),
                 q999_train.mean(dim='concat_dim').values.reshape(-1),
                 q999_sta_v.pr.values.reshape(-1),
                 q999_raw_v.mean(dim='concat_dim').values.reshape(-1),
                 q999_valid.mean(dim='concat_dim').values.reshape(-1)]).transpose()
x_.columns=["Stations train",
            "Mod RAW train",
            "Mod Adj Train",
            "Stations Val",
            "Mod RAW Validation",
            "Mod Adj Validation"]

sns.boxplot(data=x_.melt(),y="value",x="variable")
plt.xlabel("")
plt.show()


x1,x2=np.where(~np.isnan(np.max(sta_val.pr.values,axis=2)))

fig,ax=plt.subplots(3,3,figsize=(24,24))
ax=ax.flatten()

for idx,(i,j) in enumerate(zip(x1[:10],x2[:10])):

    sta_sc=sta_val.isel(time=sta_val['time.year'].isin(np.arange(2000,2005))).\
            isel(time=sta_val.isel(time=sta_val['time.year'].isin(np.arange(2000,2005)))['time.season'].isin(seas)).isel(lat=i,lon=j)

    mod_sc_raw_t=co_raw_valid_t.isel(lat=i,lon=j)

    mod_sc_raw_v=co_raw_valid_v.isel(lat=i,lon=j)

    mod_sc_adj_v=ds_mod_dict_valid[mdl].isel(time=ds_mod_dict_valid[mdl]['time.year'].isin(np.arange(2005,2010))).\
            isel(time=ds_mod_dict_valid[mdl]['time.year'].isel(time=ds_mod_dict_valid[mdl]['time.year'].isin(np.arange(2005,2010)))['time.season'].isin(seas)).isel(lat=i,lon=j)

    ax[idx].plot(np.sort(sta_sc.pr.values[sta_sc.pr.values > 0.2]),
            np.arange(sta_sc.pr.values[sta_sc.pr.values > 0.2].shape[0]) / (sta_sc.pr.values[sta_sc.pr.values > 0.2].shape[0]+1),
            '-*',
            label='Observations')
    ax[idx].plot(np.sort(mod_sc_raw_t.values[mod_sc_raw_t.values > 0.1]),
            np.arange(mod_sc_raw_t.values[mod_sc_raw_t.values > 0.1].shape[0]) / (mod_sc_raw_t.values[mod_sc_raw_t.values > 0.1].shape[0]+1),
            '-^',
            label='Mod RAW Train')
    ax[idx].plot(np.sort(mod_sc_raw_v.values[mod_sc_raw_v.values > 0.1]),
            np.arange(mod_sc_raw_v.values[mod_sc_raw_v.values > 0.1].shape[0]) / (mod_sc_raw_v.values[mod_sc_raw_v.values > 0.1].shape[0]+1),
            '-v',
            label='Mod RAW VALIDATION')
    ax[idx].plot(np.sort(mod_sc_adj_v.values[mod_sc_adj_v.values > 0.1]),
            np.arange(mod_sc_adj_v.values[mod_sc_adj_v.values > 0.1].shape[0]) / (mod_sc_adj_v.values[mod_sc_adj_v.values > 0.1].shape[0]+1),
            '-o',
            label='Mod ADJ VALIDATION')
    ax[idx].set_ylim(0.975,1)
    ax[idx].set_title(f"{mdl} on cell {i:.2f},{j:.2f}")
    ax[idx].hlines(0.98,0,14,linestyle='dashed',color='red')
    ax[idx].hlines(0.99,0,14,linestyle='dashed',color='green')
    ax[idx].legend()
plt.savefig("/home/lcesarini/sdsd.png")
plt.close()

if __name__=="__main__":
    
    SEASONS=['SON','DJF','MAM', 'JJA'] 

    for seas in SEASONS: 
        list_output_mdl=glob(f"/home/lcesarini/2022_resilience/output/{seas}/*{metrics}_**{ADJUST}**{NQUANT}*")
        sta_val=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
        BC="EQM"
        MODEL=""
        list_mdl=["MOHC","ETH","CNRM","KNMI","HCLIMcom","KIT","CMCC","ICTP"]
        ds_mod = []

        for mdl in tqdm(list_mdl):
            """
            Approximate time for the loop:
            ~ 6 minutes
            """
            list_mod_adjust=[]

            if mdl == "ETH":
                from resilience.utils.fix_year_eth import fix_eth
                eth=fix_eth()
                
                if ADJUST=="TRAIN":
                    mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
                    mod_adjust=mod_train
                elif ADJUST=="VALIDATION":
                    mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()

                ds_mod.append(get_triveneto(mod_adjust.pr,sta_val))
            elif mdl == "ICTP":
                if ADJUST=="TRAIN":
                    for year in np.arange(2000,2004):
                        list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                elif ADJUST=='VALIDATION':
                    for year in np.arange(2006,2010):
                        list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                
                ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())
            else:
                if ADJUST=="TRAIN":
                    for year in np.arange(2000,2005):
                        list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")
                elif ADJUST=='VALIDATION':
                    for year in np.arange(2005,2010):
                        list_mod_adjust=list_mod_adjust+glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*")

                ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

        ds_mod_dict={mdl:ds_mod[i] for i,mdl in enumerate(list_mdl)}

        ds_mod_dict['ETH']['time']=ds_mod_dict['MOHC'].time.values

        [print(np.all(ds_mod_dict['MOHC'].time.values == ds_mod_dict[mdl].time.values)) for mdl in list_mdl]
        
        xx=xr.open_mfdataset(list_mod_adjust).pr.load()
        ds_mod_dict['ICTP'].values.shape

        # dict_q999=[xr.where(get_season(ds_mod_dict[mdl],"JJA") > 0.2, get_season(ds_mod_dict[mdl],"JJA"),np.nan).mean(dim='time') for mdl in list_mdl]
        dict_q999=[get_season(ds_mod_dict[mdl],"JJA").quantile(dim='time',q=0.999) for mdl in list_mdl]

        # dict_q999_ens=xr.concat(dict_q999,list_mdl).rename({'concat_dim':'model'}).mean(dim='model')
        dict_q999_ens=xr.concat(dict_q999,list_mdl).rename({'concat_dim':'model'}).mean(dim='model')

        nc_to_csv(dict_q999_ens,name=f"Ensemble_q_JJA_valid",M="q",csv=True)

        
        ll_diurnal=[compute_quantiles_by_hour(ds_mod_dict[mdl],q=0.999,SEAS='JJA') for mdl in list_mdl]
        diurnal_concat=xr.concat(ll_diurnal,list_mdl).rename({'concat_dim':'model'})

        diurnal_station=compute_quantiles_by_hour(sta_val.sel(time=slice("2005-01-01","2009-12-31")),0.999,'JJA')
        diurn_sta_train=compute_quantiles_by_hour(sta_val.sel(time=slice("2000-01-01","2004-12-31")),0.999,'JJA')
        """
        COMPUTE DIFFERENCES WET HOURS STATIONS vs CPMs
        """

        sta_train=sta_val.sel(time=slice("2000-01-01","2004-12-31"))

        def get_perc_wh(xarray_ds):
            if hasattr(xarray_ds,"data_vars"):
                get_not_na=xarray_ds.pr.values.reshape(-1)[~np.isnan(xarray_ds.pr.values.reshape(-1))] 
            else:
                get_not_na=xarray_ds.values.reshape(-1)[~np.isnan(xarray_ds.values.reshape(-1))] 

            return np.round((get_not_na > 0.2).sum() /  get_not_na.shape[0] * 100,2)
        
        get_perc_wh(sta_val.sel(time=slice("2000-01-01","2004-12-31")))
        get_perc_wh(sta_val.sel(time=slice("2005-01-01","2009-12-31")))

        dict_wh_adjust={mdl:get_perc_wh(ds_mod[i]) for i,mdl in enumerate(list_mdl)}
        dict_wh_adjust.update({"Stations":get_perc_wh(sta_val.sel(time=slice("2005-01-01","2009-12-31")))})
        
        dict_wh_train={mdl:get_perc_wh(ds_mod_dict[mdl]) for i,mdl in enumerate(list_mdl)}
        dict_wh_train.update({"Stations":get_perc_wh(sta_val.sel(time=slice("2000-01-01","2004-12-31")))})

        ax=plt.axes()
        diurnal_concat.mean(dim='model').plot(label="Ensemble",ax=ax, linewidth=4,color='red')
        #From get_csv_from_nc.py
        diurnal_qdm_1000.mean(dim='model').plot(label="Ensemble QDM",ax=ax, linewidth=1,color='green',linestyle='dashed')
        diurnal_qdm_10000.mean(dim='model').plot(label="Ensemble QDM 10000",ax=ax, linewidth=1,color='green',linestyle='solid')
        diurnal_eqm_1000.mean(dim='model').plot(label="Ensemble EQM",ax=ax, linewidth=4,color='magenta',linestyle='dotted')
        
        diurnal_station.plot(label="Stations",ax=ax, linewidth=4,color='blue')
        diurn_sta_train.plot(label="Stations Train",ax=ax, linewidth=4,color='grey',linestyle="-.",alpha=0.5)
        # dict_q99_hourly[f"SPHERA_{S}"].plot(label="SPHERA",ax=ax,linestyle='-', linewidth=3,color='blue')
        # dict_q99_hourly[f"VHR_CMCC_{S}"].plot(label="VHR CMCC",ax=ax,linestyle='-', linewidth=3,color='magenta')
        # dict_q99_hourly[f"STATIONS_{S}"].plot(label="STATIONS",ax=ax,marker='*', linewidth=3,color='green')
        # dict_q99_hourly[f"GRIPHO_{S}"].plot(label="GRIPHO",ax=ax,marker='+', linewidth=3,color='orange')
        ax.set_title("Diurnal cycle of heavy precipitation (99.9th percentile) by hour JJA")
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("Precipitation (mm)")
        plt.legend()
        plt.show()
        
        train=sta_val.sel(time=slice("2000-01-01","2004-12-31"))
        adjust=sta_val.sel(time=slice("2005-01-01","2009-12-31"))
        
        df_mean=pd.DataFrame(
            [train.pr.mean(dim='time').values.reshape(-1),
            adjust.pr.mean(dim='time').values.reshape(-1)]
        ).transpose()

        df_quant=pd.DataFrame(
            [train.pr.quantile(q=0.999,dim='time').values.reshape(-1),
            adjust.pr.quantile(q=0.999,dim='time').values.reshape(-1)]
        ).transpose()


        df_mean.columns=["calibration","validation"]
        df_quant.columns=["calibration","validation"]
        sns.boxplot(df_quant[['calibration','validation']],
                    # x='name',
                    # y='Mean Prec (mm/hr)',
                    # hue='Model',
                    width=0.25,
                    notch=True, showcaps=True,
                    flierprops={"marker": "x"},
                    # boxprops={"facecolor": (.8, .6, .8, 0.14)},
                    medianprops={"color": "coral"})
        plt.show()

        x1=train.pr.values.reshape(-1)[train.pr.values.reshape(-1) > 0.2]
        x2=adjust.pr.values.reshape(-1)[adjust.pr.values.reshape(-1) > 0.2]
        dfdf=pd.DataFrame([x1,x2]).transpose().rename({0:"calibration",1:"validation"},axis=1)
        sns.boxplot(data=dfdf.melt(),x="value",hue="variable")
        plt.show()

        ll_mean=[ds_mod_dict[mdl].mean(dim='time') for mdl in list_mdl]

        [nc_to_csv(ll_mean[idx],f"{mdl}_mean_prec_biased",M='mean') for idx,mdl in enumerate(list_mdl)]
        mean_concat=xr.concat(ll_mean,list_mdl).rename({'concat_dim':'model'})

        nc_to_csv(mean_concat.mean(dim='model'),f"ensemble_mean_prec_biased",M='mean')
        
        nc_to_csv(sta_val.sel(time=slice("2005-01-01","2009-12-31")).mean(dim='time'),
                  'Stations_mean_prec','mean')
        
        plot_panel_rotated((18,15),3,3,
                       [mean_concat.sel(model=mdl) for mdl in list_mdl]+[mean_concat.mean(dim='model')],
                       "map_mean_prec",
                       list_titles=[mdl for mdl in list_mdl]+["Ensemble"],
                       levels=[np.arange(0,0.4,0.1) for _ in range(9)],
                       suptitle="Mean Precipitation for JJA",
                       name_metric=["Mean Prec." for _ in range(9) ],
                       SET_EXTENT=True,
                       cmap=['rainbow' for _ in range(9)],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree()
                    )
        nc_to_csv()
        ds_mod_dict['ETH'].coords['time']=ds_mod_dict['MOHC'].time.values

        x=(ds_mod_dict['ETH']+ds_mod_dict['MOHC']+ds_mod_dict['KNMI']+ds_mod_dict['KIT']) / 4
        x2=(ds_mod_dict['CNRM']+ds_mod_dict['ICTP']+ds_mod_dict['CMCC']+ds_mod_dict['HCLIMcom']) / 4
        ens=(x+x2)/2
        del x,x2


        ens.values > 0.1



        ds_eqm=ds_mod
        ds_qdm=ds_mod

        xx_1=[(ds_mod_dict[mdl] > 20).sum(dim='time') for mdl in list_mdl]
        xx_2=[(ds_eqm[idx] > 20).sum(dim='time') for idx,mdl in enumerate(list_mdl)]
        xx_3=[(ds_qdm[idx] > 20).sum(dim='time') for idx,mdl in enumerate(list_mdl)]
        
        mean_events_1=(xx_1[0]+xx_1[1]+xx_1[2]+xx_1[3]+xx_1[4]+xx_1[5]+xx_1[6]+xx_1[7])/8
        mean_events_2=(xx_2[0]+xx_2[1]+xx_2[2]+xx_2[3]+xx_2[4]+xx_2[5]+xx_2[6]+xx_2[7])/8
        mean_events_3=(xx_3[0]+xx_3[1]+xx_3[2]+xx_3[3]+xx_3[4]+xx_3[5]+xx_3[6]+xx_3[7])/8
        stations_even=(adjust.pr > 20).sum(dim='time')

        df_events=pd.DataFrame([stations_even.values.reshape(-1),
                                mean_events_1.values.reshape(-1),
                                mean_events_2.values.reshape(-1),
                                mean_events_3.values.reshape(-1)]).\
            transpose().rename({0:"Stations",1:"Ensemble",2:"Ensemble EQM",3:"Ensemble QDM"},axis=1)

        sns.boxplot(data=df_events.melt()[df_events.melt().value > 0],y="value",x="variable")
        plt.show()




        fig,ax=plt.subplots(1,4,subplot_kw={"projection":ccrs.PlateCarree()})

        ax=ax.flatten()
        (adjust.pr > 20).sum(dim='time').plot.pcolormesh(ax=ax[0],
                                add_colorbar=False,
                                #   cbar_kwargs={"shrink":0.85},
                                levels=[0,2,4,6,8,10],
                                cmap=cmap_q,#"rainbow",                              
                                )
        mean_events_1.plot.pcolormesh(ax=ax[1],
                                add_colorbar=False,
                                #   cbar_kwargs={"shrink":0.85},
                                levels=np.arange(0,20,5),
                                cmap=cmap_q,#"rainbow",                              
                                )
        mean_events_2.plot.pcolormesh(ax=ax[2],
                                add_colorbar=False,
                                #   cbar_kwargs={"shrink":0.85},
                                levels=np.arange(0,20,5),
                                cmap=cmap_q,#"rainbow",                              
                                )
        mean_events_3.plot.pcolormesh(ax=ax[3],
                                add_colorbar=False,
                                #   cbar_kwargs={"shrink":0.85},
                                levels=np.arange(0,20,5),
                                cmap=cmap_q,#"rainbow",                              
                                )

        [shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=0.25) for i in range(4)]
        [ax[i].add_feature(cfeature.BORDERS) for i in range(4)]
        [ax[i].coastlines() for i in range(4)]
        # if SET_EXTENT:
        #     ax[i].set_extent([10.2,13.15,44.6,47.15])
        # ax[i].set_title(f"{list_titles[i]}")
        # ax[i].set_title(f"")
        gl = ax[i].gridlines(
            # draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--',
        )
        gl.xlabels=False
        nrow,ncol=1,1
        cbar=fig.colorbar(pcm, ax=ax,
                            extend='both', 
                            orientation='horizontal',
                            shrink=1.25)
        cbar.ax.tick_params(labelsize=10)
        # cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=55)
        ax.add_feature(cfeature.BORDERS)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()
        
        

        means_list=[ds_mod_dict[mdl].mean(dim='time').values.reshape(-1) for mdl in list_mdl]
        q999_list=[ds_mod_dict[mdl].quantile(q=0.999,dim='time').values.reshape(-1) for mdl in list_mdl]
        ensemble = np.mean(np.array(means_list),axis=0)
        ens_q999 = np.mean(np.array(q999_list),axis=0)
        ens_q992 = ens.quantile(q=0.999,dim='time')
        means_list
        df=pd.DataFrame(means_list+[ensemble]).transpose()
        df_99=pd.DataFrame(q999_list +[ens_q999]).transpose()
        
        df.columns=list_mdl+['Ensemble']
        df=df.melt()
        df.rename(columns={'value':'Mean Prec (mm/hr)','variable':'Model'},inplace=True)
        df['name']='CPM'
        df.head()

        ds_mod_dict['ETH'].isel(time=ds_mod_dict['ETH']['time.season'].isin('JJA'))

        stst=sta_val.isel(time=sta_val['time.year'].isin(np.arange(2005,2010)))

        def prepare_df(dict_model,
                       stations,
                       ensemble,
                       metrics="quantile",
                       seas='JJA'):
            if metrics == "quantile":
                mod_ll        =[dict_model[mdl].isel(time=dict_model[mdl]['time.season'].isin(seas)).\
                        quantile(q=0.999,dim='time').values.reshape(-1) for mdl in list_mdl]
                station_metric=stations.pr.isel(time=stations.pr['time.season'].isin(seas)).\
                        quantile(q=0.999,dim='time').values.reshape(-1)
                ens_metrics   =ensemble.isel(time=ensemble['time.season'].isin(seas)).\
                        quantile(q=0.999,dim='time').values.reshape(-1)
            elif metrics=='mean':
                mod_ll=[dict_model[mdl].isel(time=dict_model[mdl]['time.season'].isin(seas)).\
                        mean(dim='time').values.reshape(-1) for mdl in list_mdl]
                station_metric=stations.pr.isel(time=stations.pr['time.season'].isin(seas)).\
                        mean(dim='time').values.reshape(-1)
                ens_metrics   =ensemble.isel(time=ensemble['time.season'].isin(seas)).\
                        mean(dim='time').values.reshape(-1)

            ens=np.mean(np.array(mod_ll),axis=0)
            df=pd.DataFrame(mod_ll+[ens]).transpose()
            df.columns=list_mdl+['Ensemble']
            df['Observations']=station_metric
            df['Ensemble2']=ens_metrics

            return df
        
        df_mean=prepare_df(ds_mod_dict,stst,ens,metrics='mean')
        df_quantile=prepare_df(ds_mod_dict,stst,ens,metrics='quantile')
        
        sns.boxplot(df_quantile[['Ensemble','Ensemble2','Observations']],
                    # x='name',
                    # y='Mean Prec (mm/hr)',
                    # hue='Model',
                    width=0.25,
                    notch=True, showcaps=True,
                    flierprops={"marker": "x"},
                    # boxprops={"facecolor": (.8, .6, .8, 0.14)},
                    medianprops={"color": "coral"})
        plt.show()

        sns.boxplot(df_mean[['Ensemble','Ensemble2','Observations']],
            # x='name',
            # y='Mean Prec (mm/hr)',
            # hue='Model',
            width=0.25,
            notch=True, showcaps=True,
            flierprops={"marker": "x"},
            # boxprops={"facecolor": (.8, .6, .8, 0.14)},
            medianprops={"color": "coral"})
        plt.show()

        plt.scatter(df_mean.Ensemble,df_mean.Ensemble2)
        plt.axline([np.nanmin(df_mean.Ensemble),np.nanmin(df_mean.Ensemble)],
                   [np.nanmax(df_mean.Ensemble),np.nanmax(df_mean.Ensemble)],
                   c='red')
        plt.scatter(df_quantile.Ensemble,df_quantile.Ensemble2)
        plt.axline((1,0.7853981633974483),c='red')

        plt.show()
        ds_mod_dict[mdl]['time.season'].isin(seas)
        ll_cell=[ds_mod_dict[mdl].isel(lon=71,lat=8).resample(time='1Y').max() for mdl in list_mdl]
        # plt.savefig("ex.png")
        # plt.close()

        [plt.plot(x.time,x,'--',c='grey') for x,nm in zip (ll_cell,list_mdl)]
        plt.plot(xr.concat(ll_cell,'new_dim').isel(time=np.arange(1,6)).mean(dim='new_dim').time,
                 xr.concat(ll_cell,'new_dim').isel(time=np.arange(1,6)).mean(dim='new_dim'),
                 c='green',label='Ensemble of BM'
        )
        plt.plot(ens.isel(lon=71,lat=8).resample(time='1Y').max().time,
                 ens.isel(lon=71,lat=8).resample(time='1Y').max(),label='BM of ensemble')
        plt.legend()
        plt.show()
        np.sort(df_quantile.Ensemble2[~np.isnan(df_quantile.Ensemble2)])
        np.sort(df_quantile.Ensemble[~np.isnan(df_quantile.Ensemble)])


        ds_mod_all=xr.concat(ds_mod,list_mdl).rename({"concat_dim":"model"})

        list_output_mdl=[glob(f"{PATH_BIAS_CORRECTED }{BC}/{mdl}/pr/*{mdl}**CORR_STATIONS_2005_2009**{seas}**Q{NQUANT}**")[0] for mdl in list_mdl]

        mod_adjust=xr.open_mfdataset([item for list in list_mod_adjust for item in list]).load()
        
        ds=[xr.open_dataset(file) for file in list_output_mdl]
        
        list_mdl[4]
        ds[4].pr.shape

        ds=xr.concat(ds,["_".join(os.path.basename(file).split("_")[:3]) if "stations" not in file else\
                         "_".join(os.path.basename(file).split("_")[:2]) for file in list_output_mdl]).\
                    rename({"concat_dim":"correction"})


        list_output_mdl=glob(f"/home/lcesarini/2022_resilience/output/{seas}/*{metrics}_**{ADJUST}**{NQUANT}*")
        
        list_mdl=[os.path.basename(file).split("_")[0] for file in list_output_mdl]
        


        ds=[xr.open_dataset(file) for file in list_output_mdl]

        ds=xr.concat(ds,["_".join(os.path.basename(file).split("_")[:3]) if "stations" not in file else\
                          "_".join(os.path.basename(file).split("_")[:2]) for file in list_output_mdl]).\
                    rename({"concat_dim":"correction"})
        
        ds_tri=get_triveneto(ds,sta_val)

        
        ds_eqm=ds_tri.sel(correction=ds.correction.str.contains("EQM"))
        ds_ori=ds_tri.sel(correction=ds.correction.str.contains("biased"))
        ds_ori=ds.sel(correction=~ds.correction.str.contains("stations|EQM|QDM"))

        for i in range(8): 
            nc_to_csv(ds_ori.isel(correction=i),
                      ds_ori.isel(correction=i).correction.item(),
                      M=metrics)



def plot_panel_rotated(figsize,nrow,ncol,
                       list_to_plot,
                       name_fig,
                       list_titles='Any title',
                       levels=[9],
                       suptitle="Frequency for JJA",
                       name_metric=["Frequency"],
                       SET_EXTENT=True,
                       cmap=['rainbow'],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree()
                    ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levels: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=figsize,constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})
    
    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs


    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    levels=levels[i],
                                    cmap=cmap[i],#"rainbow",
                                    transform=transform                                    
                                    )

            shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=0.25)
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            # ax[i].set_title(f"")
            gl = ax[i].gridlines(
                # draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
                draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            )
            gl.xlabels=False
            if i in range(ax.shape[0]):
                # cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                #                     extend='both', 
                #                     orientation='horizontal',
                #                     shrink=1.25)
                # cbar.ax.tick_params(labelsize=10)
                # cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=55)
                ax[i].add_feature(cfeature.BORDERS)
                ax[i].set_xticklabels([])
                ax[i].set_yticklabels([])
                # ax[i].add_feature(cfeature.STATES)


        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                        transform=transform                                    
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            # ax.add_geometries(shp_triveneto['geometry'], crs=proj)
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )
            if SET_EXTENT:
                ax.set_extent([-0.5,2.5,-2.5,0.5])

            # gl.xlines=None
            # gl.xlabels_top=None
            # gl.xlabels_bottom=None

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')
            if i in [0,1,2]:
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', 
                                    orientation='vertical',
                                    shrink=0.85)
                cbar.ax.tick_params(labelsize=30)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom')
    fig.suptitle(suptitle, fontsize=12)
    # fig.subplots_adjust(wspace=0, hspace=20)

    plt.savefig(f"/home/lcesarini/2022_resilience/figures/{name_fig}.png")
    plt.close()


plot_panel_rotated(figsize=(12,6),nrow=2,ncol=4,
                list_to_plot=[ds_ori.isel(correction=xi).pr for xi in range(ds_ori.correction.shape[0])],
                name_fig="panels_board",
                list_titles='Any title',
                levels=[lvl_q for xi in range(ds_ori.correction.shape[0])],
                suptitle="Biased Heavy prec for JJA",
                name_metric=["Heavy Prec." for xi in range(ds_ori.correction.shape[0])],
                SET_EXTENT=False,
                cmap=[cmap_q for xi in range(ds_ori.correction.shape[0])],
                proj=ccrs.PlateCarree(),
                transform=ccrs.PlateCarree()
            ) 

plot_panel_rotated(figsize=(40,20),nrow=2,ncol=4,
                list_to_plot=[ds_eqm.isel(correction=xi).pr for xi in range(ds_eqm.correction.shape[0])],
                name_fig="panels_board",
                list_titles='Any title',
                levels=[lvl_q for xi in range(ds_eqm.correction.shape[0])],
                suptitle="EQM Heavy prec for JJA",
                name_metric=["Heavy Prec." for xi in range(ds_eqm.correction.shape[0])],
                SET_EXTENT=True,
                cmap=[cmap_q for xi in range(ds_eqm.correction.shape[0])],
                proj=ccrs.PlateCarree(),
                transform=ccrs.PlateCarree()
            ) 




        # ds.corre    
        # from numba import jit

        # @jit
        # def get_wh(arr):
        #     arr_re=arr.reshape(-1)
        #     arr_wh=arr_re > 0.1

        #     return arr_re[arr_wh]

        # def get_wh2(arr):
        #     arr_re=arr.reshape(-1)
        #     arr_wh=arr_re > 0.1

        #     return arr_re[arr_wh]
seas='DJF'


ds_mod_st = []
for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    list_mod_adjust=list_mod_adjust+glob(f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{mdl}**STATIONS**2005**{seas}**Q{NQUANT}*")
    ds_mod_st.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

ds_mod = []
for mdl in tqdm(list_mdl):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]

    list_mod_adjust=list_mod_adjust+glob(f"{PATH_BIAS_CORRECTED}/{BC}/{mdl}/pr/*{mdl}**SPHERA**2005**{seas}**Q{NQUANT}*")
    ds_mod.append(get_triveneto(xr.open_mfdataset(list_mod_adjust).pr,sta_val).load())

ds_mod_raw = []
for mdl in tqdm(list_mdl[1:]):
    """
    Approximate time for the loop:
    ~ 6 minutes
    """
    list_mod_adjust=[]
    list_mod_adjust=list_mod_adjust+[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{yr}**.nc*") for yr in np.arange(2005,2010)]

    ds_mod_raw.append(get_triveneto(xr.open_mfdataset( get_unlist(list_mod_adjust)).pr,sta_val).load())

(ds * mask.mask).isel(longitude=ds.longitude.isin(sta_val.lon),
                                     latitude=ds.latitude.isin(sta_val.lat))



ll_sph=get_unlist([glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{yr}**.nc*") for yr in np.arange(2005,2010)])
ds_sph=xr.open_mfdataset(ll_sph).pr

ds_sph["longitude"] = mask.mask['lon'].values
ds_sph["latitude"] = mask.mask['lat'].values
ds_sph["latitude"] == mask.mask['lat'].values

ds_sph=ds_sph.rename({'longitude':'lon','latitude':'lat'})
ds_sph * mask.mask

ds_sph=(get_triveneto(ds_sph,sta_val).load())
ds_sph=get_season(ds_sph,season=seas)

ll_sta=get_unlist([glob(f"{PATH_COMMON_DATA}/stations/pr/*{yr}**.nc*") for yr in np.arange(2005,2010)])
ds_sta=get_triveneto(xr.open_mfdataset(ll_sta),sta_val).load()
ds_sta=get_season(ds_sta,season=seas)


[print(np.all(ds_mod[0].time.values == ds_mod[i].time.values)) for i in range(7)]
[print(np.all(ds_mod_st[0].time.values == ds_mod_st[i].time.values)) for i in range(7)]
[print(np.all(ds_mod_raw[0].time.values == ds_mod_raw[i].time.values)) for i in range(7)]

ds_mod[1]['time'] = ds_mod[0].time.values
ds_mod_st[1]['time'] = ds_mod_st[0].time.values
ds_mod_raw[1]['time'] = ds_mod_raw[0].time.values

[print(np.all(ds_mod[0].time.values == ds_mod[i].time.values)) for i in range(7)]
[print(np.all(ds_mod_st[0].time.values == ds_mod_st[i].time.values)) for i in range(7)]
[print(np.all(ds_mod_raw[0].time.values == ds_mod_raw[i].time.values)) for i in range(7)]
 
sphera_adj=xr.concat([mod.sel(time=mod['time.year'].isin(np.arange(2005,2010))) for mod in ds_mod],dim='newdim')
statio_adj=xr.concat([mod.sel(time=mod['time.year'].isin(np.arange(2005,2010))) for mod in ds_mod_st],dim='newdim')
ensemble_r=xr.concat([mod.sel(time=mod['time.year'].isin(np.arange(2005,2010))) for mod in ds_mod_raw],dim='newdim')
sphera_raw=ds_sph.sel(time=ds_sph['time.year'].isin(np.arange(2005,2010)))
statio_raw=ds_sta.sel(time=ds_sta['time.year'].isin(np.arange(2005,2010)))

sph_999=sphera_adj.quantile(q=0.999,dim='time').mean(dim='newdim')
sta_999=statio_adj.quantile(q=0.999,dim='time').mean(dim='newdim')
raw_999=get_season(ensemble_r,season=seas).quantile(q=0.999,dim='time').mean(dim='newdim')

sta__qr=statio_raw.quantile(q=0.999,dim='time')
raw__qr=sphera_raw.quantile(q=0.999,dim='time')

df=pd.DataFrame([
                 sph_999.values.reshape(-1),
                 raw__qr.values.reshape(-1),
                 raw_999.values.reshape(-1),
                 sta__qr.pr.values.reshape(-1),
                 sta_999.values.reshape(-1),

              ]).transpose().\
    rename({0:"SPHERA",1:"SPHERA corre.",
            2:"RAW Model",
            3:"STATIONS corr.",4:"STATIONS"},axis=1).\
    melt()
sns.boxplot(data=df,x='variable',y='value')
plt.title(f"Correction with SPHERA and stations on {seas}")
plt.show()


bb100=20
h_day=400
stake=0.05

eur_day=bb100 * h_day/100 * stake

month=20

print(f"""
------------------------------
| {int(month*eur_day)}€ earnings per month     | 
| at {int(stake*100)}NL                     | 
| with a winrate of {bb100}bb/100 |
| N° of hours {h_day/200}            |
| Total number of hands {h_day * month} |
------------------------------
""")

print(f"{int(month*eur_day) * 12} per year")