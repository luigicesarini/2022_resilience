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
parser.add_argument('-c','--curvilinear', type=bool,
                    help = ("Is the CRS of the product 2D?"),
                    default = False )
args = parser.parse_args()

os.chdir("/home/lcesarini/2022_resilience/")

shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

gripho = f"/mnt/data/lcesarini/gripho-v1_1h_TSmin30pct_2001-2016_remap.nc"
# e_obs  = f"/mnt/data/commonData/rr_ens_mean_0.1deg_reg_1995-2010_v21.0e.nc" 
# sim_hclim = f"/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"
sim_hclim = f"/mnt/data/lcesarini/ECMWF-ERAINT/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"

if __name__=="__main__":

    PRODUCT     = args.product
    CURVILINEAR = args.curvilinear

    ds_obs=xr.open_mfdataset(gripho).load()
    ds_obs=ds_obs.isel(time=ds_obs['time.year'].isin(np.arange(2000,2009)))

    ds_proj=xr.open_mfdataset(sim_hclim).load()
    ds_proj=ds_proj.isel(time=ds_proj['time.year'].isin(np.arange(2000,2009)))

    if not all(ds_obs["time.year"].isin(ds_proj["time.year"])):
        ds_obs=ds_obs.isel(time=ds_obs["time.year"].isin(ds_proj["time.year"]))
    elif not all(ds_proj["time.year"].isin(ds_obs["time.year"])):
        ds_proj=ds_proj.isel(time=ds_proj["time.year"].isin(ds_obs["time.year"]))

    print(ds_obs.pr.shape[0]==ds_proj.pr.shape[0])

    def get_autumn(ds):
        return ds.isel(time=ds['time.season'].isin("SON"))

    def get_winter(ds):
        return ds.isel(time=ds['time.season'].isin("DJF"))

    def get_spring(ds):
        return ds.isel(time=ds['time.season'].isin("MAM"))

    def get_summer(ds):
        return ds.isel(time=ds['time.season'].isin("JJA"))


    ds_obs_autumn=get_winter(ds_obs)
    ds_obs_winter=get_winter(ds_obs)
    ds_obs_spring=get_summer(ds_obs)
    ds_obs_summer=get_summer(ds_obs)
    ds_mod_autumn=get_winter(ds_proj)
    ds_mod_winter=get_winter(ds_proj)
    ds_mod_spring=get_summer(ds_proj)
    ds_mod_summer=get_summer(ds_proj)

    # FOR HCLIM meters=False
    # if PRODUCT == "HCLIM":
    #     freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=False)
    #     freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=False)
    # elif PRODUCT == "GRIPHO":
    #     freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=True)
    #     freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=True)
    # elif PRODUCT == "ERA5":
    #     freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=False)
    #     freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=False)

    freq_obs_a,int_obs_a,perc_obs_a=compute_metrics(ds_obs_winter,meters=True,quantile=0.999)
    freq_obs_w,int_obs_w,perc_obs_w=compute_metrics(ds_obs_winter,meters=True,quantile=0.999)
    freq_obs_p,int_obs_p,perc_obs_p=compute_metrics(ds_obs_summer,meters=True,quantile=0.999)
    freq_obs_s,int_obs_s,perc_obs_s=compute_metrics(ds_obs_summer,meters=True,quantile=0.999)

    freq_mod_a,int_mod_a,perc_mod_a=compute_metrics(ds_mod_winter,meters=False,quantile=0.999)
    freq_mod_w,int_mod_w,perc_mod_w=compute_metrics(ds_mod_winter,meters=False,quantile=0.999)
    freq_mod_p,int_mod_p,perc_mod_p=compute_metrics(ds_mod_summer,meters=False,quantile=0.999)
    freq_mod_s,int_mod_s,perc_mod_s=compute_metrics(ds_mod_summer,meters=False,quantile=0.999)

    

    assert freq_obs_s.shape==freq_mod_s.shape,"OOPS"

    def bias_metrics(metric,mod,obs):
        #ADD ARGUMEN T IF TRIVENETO
        mod=mod.where((mod.lon > 10.38) & (mod.lon < 13.1) & (mod.lat > 44.7) & (mod.lat < 47.1), drop=True)
        obs=obs.where((obs.lon > 10.38) & (obs.lon < 13.1) & (obs.lat > 44.7) & (obs.lat < 47.1), drop=True)
        bias_metric=((mod - obs) / obs) * 100

        bias_metric=xr.where(np.isfinite(bias_metric),bias_metric,np.nan)

        return bias_metric



    bias_freq_a=np.nanmean(bias_metrics("freq",freq_mod_a,freq_obs_a))
    bias_freq_w=np.nanmean(bias_metrics("freq",freq_mod_w,freq_obs_w))
    bias_freq_p=np.nanmean(bias_metrics("freq",freq_mod_p,freq_obs_p))
    bias_freq_s=np.nanmean(bias_metrics("freq",freq_mod_s,freq_obs_s))
    
    bias_inte_a=np.nanmean(bias_metrics("inte",int_mod_a,int_obs_a))
    bias_inte_w=np.nanmean(bias_metrics("inte",int_mod_w,int_obs_w))
    bias_inte_p=np.nanmean(bias_metrics("inte",int_mod_p,int_obs_p))
    bias_inte_s=np.nanmean(bias_metrics("inte",int_mod_s,int_obs_s))
    
    bias_perc_a=np.nanmean(bias_metrics("perc",perc_mod_a,perc_obs_a))
    bias_perc_w=np.nanmean(bias_metrics("perc",perc_mod_w,perc_obs_w))
    bias_perc_p=np.nanmean(bias_metrics("perc",perc_mod_p,perc_obs_p))
    bias_perc_s=np.nanmean(bias_metrics("perc",perc_mod_s,perc_obs_s))

    import seaborn as sns
    heatmap=np.array([bias_freq_a,bias_freq_w,bias_freq_p,bias_freq_s,
                      bias_inte_a,bias_inte_w,bias_inte_p,bias_inte_s,
                      bias_perc_a,bias_perc_w,bias_perc_p,bias_perc_s]).reshape(4,3,order="F")
    
    cmap=plt.cm.RdBu
    norm = mpl.colors.BoundaryNorm([-50,0,50,88], 4+1, extend='both')

 
    # fig,ax = plt.subplots(nrows=2,
    #                       ncols=3,
    #                       figsize=(18,6),constrained_layout=True, squeeze=True,
    #                       subplot_kw={"projection":ccrs.PlateCarree()})

    # ax=ax.flatten()
    # cmap = plt.cm.rainbow
    # titles = ['Frequency','Intensity', "Heavy Prec. (p99.9)",'Frequency','Intensity', "Heavy Prec. (p99.9)"]
    # shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
    # shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

    # for i,metric in enumerate([bias_freq_w,bias_freq_s,bias_inte_w,
    #                            bias_inte_s,bias_perc_w,bias_perc_s]):

    #     limits = (np.nanmin(metric),np.nanmax(metric))
    #     bounds =  np.arange(limits[0],limits[1],limits[1]/6)
    #     bounds[0]  =  bounds[1] * 0.675 if bounds[1] > 0 else bounds[1] * 1.325
    #     norm = mpl.colors.BoundaryNorm(bounds.round(2), bounds.shape[0]+1, extend='both')


    #     pcm=metric.plot.pcolormesh(ax=ax[i],alpha=1,
    #                                 cmap=cmap, norm=norm, 
    #                                 add_colorbar=True, 
    #                                 cbar_kwargs={"shrink":0.6,
    #                                             "orientation":"horizontal",
    #                                             "label":f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}"}
    #                     )
    #     ax[i].coastlines()
    #     gl = ax[i].gridlines(
    #         draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
    #     )
    #     # ax[i].add_feature(cfeature.BORDERS, linestyle='--')
    #     # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
    #     # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
    #     # ax.add_feature(cfeature.STATES)
    #     ax[i].set_title(f"{titles[i] if i in [0,1,2] else ''}")
    #     ax[i].set_ylabel(f"{'Winter' if i in [0,3] else 'Summer'}")
    #     shp_triveneto.boundary.plot(ax=ax[i], edgecolor="green")


    # plt.suptitle(f"OBS. Stations")
    # plt.savefig(f"figures/bias_maps.png")
    # plt.close()
