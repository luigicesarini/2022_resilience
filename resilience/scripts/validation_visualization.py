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
from random import sample
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
parser.add_argument('-c','--curvilinear',action='store_true',
                    help = ("Is the CRS of the product 2D?"))
args = parser.parse_args()


os.chdir("/home/lcesarini/2022_resilience/")

shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

# gripho = f"/mnt/data/lcesarini/gripho-v1_1h_TSmin30pct_2001-2016_remap.nc"
gripho = f"/mnt/data/commonData/OBSERVATIONS/ITALY/gripho-v1_1h_TSmin30pct_2001-2016.nc"
e_obs  = f"/mnt/data/commonData/rr_ens_mean_0.1deg_reg_1995-2010_v21.0e.nc" 
# sim_hclim = f"/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"
#REMAPPED
sim_hclim = f"/mnt/data/lcesarini/ECMWF-ERAINT/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"
#TRIVENETO ORIGINAL
# sim_hclim = f"/mnt/data/lcesarini/ECMWF-ERAINT/HCLIMcom/CPM/pr/triveneto/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"
# sim_hclim = f"/home/lcesarini/2022_resilience/data/regrid/ECMWF-ERAINT/HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_*.nc"
era5      = [f'/mnt/data/lcesarini/ERA5/resilience/tp/tp_{year}.nc' for year in np.arange(2000,2009)]
if __name__=="__main__":

    PRODUCT     = args.product
    CURVILINEAR = args.curvilinear
    print(PRODUCT,CURVILINEAR)

    if PRODUCT == "GRIPHO":
        ds=xr.open_mfdataset(gripho).load()
        ds=ds.where((ds.lon > 10.38) & (ds.lon < 13.1) & (ds.lat > 44.7) & (ds.lat < 47.1), drop=True)
        ds=ds.isel(time=ds['time.year'].isin(np.arange(2000,2009)))
    elif (PRODUCT == "HCLIM") & CURVILINEAR:
        ds=xr.open_mfdataset(sim_hclim)
        # ds=ds.where((ds.lon > 10.38) & (ds.lon < 13.1) & (ds.lat > 44.7) & (ds.lat < 47.1), drop=True)
        ds=ds.load()
    elif (PRODUCT == "HCLIM") & (not CURVILINEAR):
        ds=xr.open_mfdataset(sim_hclim)
        ds=ds.where((ds.lon > 10.38) & (ds.lon < 13.1) & (ds.lat > 44.7) & (ds.lat < 47.1), drop=True)
        ds=ds.load()
    elif PRODUCT == "ERA5":
        ds=xr.open_mfdataset(era5)
        ds=ds.load()

    # ds_bms=ds.resample(time="1M").max()
    # ds_bms_mean=ds_bms.groupby('time.month').mean()

    if hasattr(ds,"Lambert_Conformal"):
        rot = ccrs.LambertConformal(central_longitude=16, central_latitude=45.5, 
                                    false_easting=1349205.5349238443, false_northing=732542.657192843)
    elif PRODUCT=="ERA5":
        rot=ccrs.PlateCarree()
    else:
        rot = ccrs.LambertAzimuthalEqualArea(central_longitude =10, 
                                            central_latitude   =52, 
                                            false_easting=4321000,
                                            false_northing=3210000, 
                                            globe=None)

    months=np.array(
        ["January","February","March",
        "April","May","June",
        "July","August","September",
        "October","November","December"])




    ds_autumn=ds.isel(time=ds['time.season'].isin("SON"))
    ds_winter=ds.isel(time=ds['time.season'].isin("DJF"))
    ds_spring=ds.isel(time=ds['time.season'].isin("MAM"))
    ds_summer=ds.isel(time=ds['time.season'].isin("JJA"))

    # FOR HCLIM meters=False
    if PRODUCT == "HCLIM":
        freq_a,int_a,perc_a=compute_metrics(ds_autumn,meters=False,quantile=0.999)
        freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=False,quantile=0.999)
        freq_p,int_p,perc_p=compute_metrics(ds_spring,meters=False,quantile=0.999)
        freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=False,quantile=0.999)
    elif PRODUCT == "GRIPHO":
        freq_a,int_a,perc_a=compute_metrics(ds_autumn,meters=True,quantile=0.999)
        freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=True,quantile=0.999)
        freq_p,int_p,perc_p=compute_metrics(ds_spring,meters=True,quantile=0.999)
        freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=True,quantile=0.999)
    elif PRODUCT == "ERA5":
        freq_a,int_a,perc_a=compute_metrics(ds_autumn,meters=False,quantile=0.999)
        freq_w,int_w,perc_w=compute_metrics(ds_winter,meters=False,quantile=0.999)
        freq_p,int_p,perc_p=compute_metrics(ds_spring,meters=False,quantile=0.999)
        freq_s,int_s,perc_s=compute_metrics(ds_summer,meters=False,quantile=0.999)

    titles = ['Frequency','Intensity', "Heavy Prec. (p99.9)",
              'Frequency','Intensity', "Heavy Prec. (p99.9)",
              'Frequency','Intensity', "Heavy Prec. (p99.9)"]

    list_to_plot=[freq_a,int_a,perc_a,
                  freq_w,int_w,perc_w,
                  freq_p,int_p,perc_p,
                  freq_s,int_s,perc_s]

    names_metrics=["freq_a","int_a","perc_a",
                   "freq_w","int_w","perc_w",
                   "freq_p","int_p","perc_p",
                   "freq_s","int_s","perc_s"]



    cmap = plt.cm.rainbow
    bounds_freq=np.linspace(0.02,0.18,10)
    bounds_inte=np.linspace(0.2,2,10)
    bounds_perc=np.linspace(1,15,10)
    
    PANEL=False
    
    if PANEL:
        fig,ax = plt.subplots(nrows=int(len(list_to_plot) / 3),
                            ncols=3,#int(len(list_to_plot) / 2),
                            figsize=(18,12),constrained_layout=True, squeeze=True,
                            subplot_kw={"projection":ccrs.PlateCarree()})

        ax=ax.flatten()


        # fig.subplots_adjust(bottom=0.01, top=0.99,right=0.99, left=0.01)

        for i,metric in enumerate(list_to_plot):#test.pr.isel(time=19).plot(ax=ax,alpha=0.95)
            # bounds = np.zeros((percentili.shape[0]+1,))
            # bounds[1:] =  np.nanquantile(metric,q=percentili).round(2)
            # bounds[0]  =  bounds[1]/2
            # limits = (np.nanmin(metric),np.nanmax(metric))
            # bounds =  np.arange(limits[0],limits[1],limits[1]/6)
            # bounds[0]  =  bounds[1] * 0.675

            # to_clip=metric.rio.write_crs("epsg:4326",inplace=False)
            # to_clip=metric.rio.write_crs(ds.crs.crs,inplace=False)
            # clipped = to_clip.rio.set_spatial_dims("x", "y").rio.clip(shp_triveneto.geometry.apply(mapping), shp_triveneto.crs)
            clipped=metric
            # limits = (np.nanmin(clipped),np.nanmax(clipped))
            # bounds =  np.arange(limits[0],limits[1],limits[1]/6)
            # bounds[0]  =  bounds[1] * 0.675
            if i in np.arange(0,len(ax),3):
                bounds = bounds_freq
            elif i in np.arange(1,len(ax),3):
                bounds = bounds_inte
            elif i in np.arange(2,len(ax),3):
                bounds = bounds_perc


            # if i in [0,3]:
            #     bounds = np.array([0.02,0.06,0.12,0.16,0.22])
            # elif i in [1,4]:
            #     bounds = np.array([0.2,0.4,0.6,0.75,0.9])
            # elif i in [2,5]:
            #     bounds = np.array([1,5,10,15,20,24])
            
            norm = mpl.colors.BoundaryNorm(bounds.round(2), bounds.shape[0]+1, extend='both')
            if CURVILINEAR:
                if i in np.arange(len(ax)-3,len(ax)):
                    pcm=clipped.plot.pcolormesh(ax=ax[i],alpha=1,
                                    transform=rot,add_colorbar=True if i in np.arange(len(ax)-3,len(ax)) else False,
                                    cmap=cmap, norm=norm,
                                    cbar_kwargs={"shrink":0.7,
                                                "orientation":"horizontal",
                                                "label":f"{'fraction' if i in [len(ax)-3] else 'mm/h' if i in [len(ax)-2,len(ax)-1] else ''}"}
                                    )
                else:
                    pcm=clipped.plot.pcolormesh(ax=ax[i],alpha=1,
                                transform=rot,add_colorbar=False,
                                cmap=cmap, norm=norm,
                                # cbar_kwargs={"shrink":0.8,
                                #             "orientation":"horizontal",
                                #             "label":f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}"}
                                )
            else:
                if i in np.arange(len(ax)-3,len(ax)):

                    pcm=clipped.plot.contourf(ax=ax[i],alpha=1,
                        cmap=cmap, norm=norm, add_colorbar=True if i in np.arange(len(ax)-3,len(ax)) else False, 
                        cbar_kwargs={"shrink":0.7,
                                    "orientation":"horizontal",
                                    "label":f"{'fraction' if i in [len(ax)-3] else 'mm/h' if i in [len(ax)-2,len(ax)-1] else ''}"}
                        )
                else:
                    pcm=clipped.plot.contourf(ax=ax[i],alpha=1,
                        cmap=cmap, norm=norm, add_colorbar=False, 
                        )

            ax[i].coastlines()
            gl = ax[i].gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',xlocs=[11],ylocs=[45,47]
            )
            # ax[i].add_feature(cfeature.BORDERS, linestyle='--')
            # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
            # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
            # ax.add_feature(cfeature.STATES)
            ax[i].set_title(f"{titles[i] if i in [0,1,2] else ''}")
            ax[i].set_ylabel(f"{'Winter' if i in [0,3] else 'Summer'}")
            shp_triveneto.boundary.plot(ax=ax[i], edgecolor="green")

            # if i in [3,4,5]:
            #     fig.colorbar(pcm,ax=ax[i], shrink=0.8, orientation='horizontal',
            #                  label=f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}")
        if CURVILINEAR:
            plt.savefig(f"figures/{PRODUCT}_2_95.png")
        else:
            plt.savefig(f"figures/{PRODUCT}_2_95_reg.png")
        plt.close()


    else:
        """
        SINGLE PLOT
        """

        for i,metric in enumerate(sample(list_to_plot,12)):#test.pr.isel(time=19).plot(ax=ax,alpha=0.95)
            #print(np.arange(start=np.nanmin(metric) * 1.1 ,stop=np.nanmax(metric) * 0.9, step=(np.nanmax(metric)-np.nanmin(metric))/8))
            fig,ax = plt.subplots(nrows=1,
                                  ncols=1,#int(len(list_to_plot) / 2),
                                  figsize=(6,4),constrained_layout=True, squeeze=True,
                                  subplot_kw={"projection":ccrs.PlateCarree()})
            
            if isinstance(ax,np.ndarray):
                ax=ax.flatten()       

            season= "Winter" if i in np.arange(3,6) else "Autumn" if i in np.arange(0,3) else "Spring" if i in np.arange(6,9) else "Summer"
            
            if i in np.arange(0,len(list_to_plot),3):
                bounds = bounds_freq
            elif i in np.arange(1,len(list_to_plot),3):
                bounds = bounds_inte
            elif i in np.arange(2,len(list_to_plot),3):
                bounds = bounds_perc

            
            norm = mpl.colors.BoundaryNorm(bounds.round(2), bounds.shape[0]+1, extend='both')
            if CURVILINEAR:
                if i in np.arange(len(list_to_plot)-3,len(list_to_plot)):
                    pcm=metric.plot(ax=ax,alpha=1,
                                    transform=rot,add_colorbar=True if i in np.arange(len(list_to_plot)-3,len(list_to_plot)) else False,
                                    cmap=cmap, 
                                    # norm=norm,
                                    cbar_kwargs={"shrink":0.7,
                                                "orientation":"horizontal",
                                                "label":f"{'fraction' if i in [len(list_to_plot)-3] else 'mm/h' if i in [len(list_to_plot)-2,len(list_to_plot)-1] else ''}"}
                                    )
                else:
                    pcm=metric.plot(ax=ax,alpha=1,
                                transform=rot,add_colorbar=False,
                                cmap=cmap, 
                                # norm=norm,
                                # cbar_kwargs={"shrink":0.8,
                                #              "extend":"both",
                                #              "orientation":"horizontal",
                                #              "label":f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}"}
                                )
            else:
                if i in np.arange(len(list_to_plot)-3,len(list_to_plot)):

                    pcm=metric.plot(ax=ax,alpha=1,
                        cmap=cmap, 
                        levels=np.arange(np.nanmin(metric) * 1.1 ,np.nanmax(metric) * 0.9, (np.nanmax(metric)-np.nanmin(metric))/8),
                        #norm=norm, 
                        add_colorbar=True if i in np.arange(len(list_to_plot)-3,len(list_to_plot)) else False, 
                        cbar_kwargs={"shrink":0.7,
                                     "extend":"both",
                                     "orientation":"horizontal",
                                     "label":f"{'fraction' if i in [len(list_to_plot)-3] else 'mm/h' if i in [len(list_to_plot)-2,len(list_to_plot)-1] else ''}"}
                        )
                else:
                    pcm=metric.plot(ax=ax,alpha=1,
                                             cmap=cmap, 
                                             #norm=norm,
                                             add_colorbar=False, 
                        )

            ax.coastlines()
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',xlocs=[11.5],ylocs=[45,47]
            )
            # ax[i].add_feature(cfeature.BORDERS, linestyle='--')
            # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
            # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
            # ax.add_feature(cfeature.STATES)
            ax.set_title(f"{titles[i] if i in [0,1,2] else ''}")
            ax.set_ylabel(season)
            shp_triveneto.boundary.plot(ax=ax, edgecolor="green")

            # if i in [3,4,5]:
            #     fig.colorbar(pcm,ax=ax[i], shrink=0.8, orientation='horizontal',
            #                  label=f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}")
            plt.savefig(f"figures/{PRODUCT}/{season}_{names_metrics[i]}.png")
            plt.close()
