#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

from utils import *

import warnings
warnings.filterwarnings('ignore')
path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"


parser = argparse.ArgumentParser()
parser.add_argument('-p','--product', type=str,
                    help = ("Pass the product to plot.(ALL CAPS)"),
                    default = "HCLIM" )
args = parser.parse_args()


os.chdir("/home/lcesarini/2022_resilience/")

if __name__=="__main__":

    meta = pd.read_csv("stations/meta_stations_eval.csv")
    #WINTER
    meta['freq_w']=np.nan
    meta['int_w'] =np.nan
    meta['prec_w']=np.nan
    #SUMMER
    meta['freq_s']=np.nan
    meta['int_s'] =np.nan
    meta['prec_s']=np.nan
    #AUTUMN
    meta['freq_a']=np.nan
    meta['int_a'] =np.nan
    meta['prec_a']=np.nan
    #SPRING
    meta['freq_p']=np.nan
    meta['int_p'] =np.nan
    meta['prec_p']=np.nan

    for name_station in tqdm(meta.name):

        records_ith = np.array(pd.read_csv(f"stations/text/prec_{name_station}.csv"))

        dates_ith   = np.array(pd.read_csv(f"data/dates/{name_station}.csv"))

        bms_obs     = pd.Series(records_ith.reshape(-1)[:np.min([records_ith.shape[0],dates_ith.shape[0]])],
                                index=dates_ith.reshape(-1)[:np.min([records_ith.shape[0],dates_ith.shape[0]])])

        filter_period=(bms_obs.index >= np.array("2000-01-01 00:00:00")) & (bms_obs.index < np.array("2010-01-01 00:00:00"))

        bms_filtered=bms_obs[filter_period]
         # df_bias= pd.DataFrame(pd.Series(bias_bms.reshape(-1),index=bms_obs.index).resample("1M").max())    
        filter_winter=[pd.to_datetime(idx).strftime("%m") in ['12','01','02'] for idx in bms_filtered.index]
        filter_summer=[pd.to_datetime(idx).strftime("%m") in ['06','07','08'] for idx in bms_filtered.index]
        filter_autumn=[pd.to_datetime(idx).strftime("%m") in ['09','10','11'] for idx in bms_filtered.index]
        filter_spring=[pd.to_datetime(idx).strftime("%m") in ['06','07','08'] for idx in bms_filtered.index]

        ds_a=bms_filtered[filter_autumn]
        ds_p=bms_filtered[filter_spring]
        ds_w=bms_filtered[filter_winter]
        ds_s=bms_filtered[filter_summer]
        # by_month = df_bias.groupby('Date').mean()
        freq_a = (pd.DataFrame(ds_a) > 0.2).sum() / pd.DataFrame(ds_a).shape[0] 
        int_a  = np.nanmean(pd.DataFrame(ds_a)[pd.DataFrame(ds_a) > 0.2])
        prec_a = np.quantile(pd.DataFrame(ds_a), q=0.999)

        freq_w = (pd.DataFrame(ds_w) > 0.2).sum() / pd.DataFrame(ds_w).shape[0] 
        int_w  = np.nanmean(pd.DataFrame(ds_w)[pd.DataFrame(ds_w) > 0.2])
        prec_w = np.quantile(pd.DataFrame(ds_w), q=0.999)
        
        freq_p = (pd.DataFrame(ds_p) > 0.2).sum() / pd.DataFrame(ds_p).shape[0] 
        int_p  = np.nanmean(pd.DataFrame(ds_p)[pd.DataFrame(ds_p) > 0.2])
        prec_p = np.quantile(pd.DataFrame(ds_p), q=0.999)

        freq_s = (pd.DataFrame(ds_s) > 0.2).sum() / pd.DataFrame(ds_s).shape[0] 
        int_s  = np.nanmean(pd.DataFrame(ds_s)[pd.DataFrame(ds_s) > 0.2])
        prec_s = np.quantile(pd.DataFrame(ds_s), q=0.999)

        meta.loc[meta.name==name_station[:],'freq_a']=freq_a.item()
        meta.loc[meta.name==name_station[:],'int_a'] =int_a
        meta.loc[meta.name==name_station[:],'prec_a']=prec_a

        meta.loc[meta.name==name_station[:],'freq_w']=freq_w.item()
        meta.loc[meta.name==name_station[:],'int_w'] =int_w
        meta.loc[meta.name==name_station[:],'prec_w']=prec_w

        meta.loc[meta.name==name_station[:],'freq_p']=freq_p.item()
        meta.loc[meta.name==name_station[:],'int_p'] =int_p
        meta.loc[meta.name==name_station[:],'prec_p']=prec_p
        
        meta.loc[meta.name==name_station[:],'freq_s']=freq_s.item()
        meta.loc[meta.name==name_station[:],'int_s'] =int_s
        meta.loc[meta.name==name_station[:],'prec_s']=prec_s

        
    meta.to_csv("stations/meta_stations_eval.csv",
                index=False,
                columns=meta.columns)

    

    meta = pd.read_csv("stations/meta_stations_eval.csv")

    gds=gpd.GeoDataFrame(meta,
                         geometry=gpd.points_from_xy(meta[["lon"]],
                                                     meta["lat"], 
                         crs="EPSG:4326"))

    
    titles = ['Frequency','Intensity', "Heavy Prec. (p99.9)",'Frequency','Intensity', "Heavy Prec. (p99.9)"]
    shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
    shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

    fig,ax = plt.subplots(nrows=4,
                          ncols=3,
                          figsize=(12,6),constrained_layout=True, squeeze=True,
                          subplot_kw={"projection":ccrs.PlateCarree()})

    ax=ax.flatten()


    # fig.subplots_adjust(bottom=0.01, top=0.99,right=0.99, left=0.01)

    cmap = plt.cm.rainbow

    bounds_freq=np.linspace(0.02,0.18,10)
    bounds_inte=np.linspace(0.2,2,10)
    bounds_perc=np.linspace(1,15,10)


    for i,metric in enumerate(["freq_a","int_a","prec_a",
                               "freq_w","int_w","prec_w",
                               "freq_p","int_p","prec_p",
                               "freq_s","int_s","prec_s"]):

        # print(metric,gds[metric].min(),gds[metric].max())
        if "freq" in metric:
            plt_stat=gds.plot(ax=ax[i],column=metric,cmap=cmap,legend=False,
                            scheme="User_Defined",classification_kwds=dict(bins=bounds_freq))
        elif "int" in metric:
            plt_stat=gds.plot(ax=ax[i],column=metric,cmap=cmap,legend=False,
                            scheme="User_Defined",classification_kwds=dict(bins=bounds_inte))
        elif "prec" in metric:
            plt_stat=gds.plot(ax=ax[i],column=metric,cmap=cmap,legend=False,
                            scheme="User_Defined",classification_kwds=dict(bins=bounds_perc))

        ax[i].coastlines()
        # ax[i].add_feature(cfeature.BORDERS, linestyle='--')
        # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
        # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
        # ax.add_feature(cfeature.STATES)
        ax[i].set_title(f"{titles[i] if i in [0,1,2] else ''}")
        ax[i].set_ylabel(f"{'Winter' if i in [0,3] else 'Summer'}")
        shp_triveneto.boundary.plot(ax=ax[i], edgecolor="green")

        # if i in np.arange(len(ax)-3,len(ax)):
        #     fig.colorbar(plt_stat,ax=ax[i], shrink=0.8, orientation='horizontal',
        #                  label=f"{'fraction' if i in [9] else 'mm/h' if i in [10,11] else ''}")


    plt.suptitle(f"OBS. Stations")
    plt.savefig(f"figures/stations_2_int.png")
    plt.close()


