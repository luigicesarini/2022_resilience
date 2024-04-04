#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")

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

from resilience.utils import *

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")
# from scripts.utils import *
sea_mask=xr.open_dataset("/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")

cmap_wind = (mpl.colors.ListedColormap([
                                        "#FF0000",
                                        "#FFFF00" ,
                                        "#00FF00" ,
                                        "#00FFFF" ,
                                        "#0000FF" ,
                                        "#FF00FF"
                                    ])
        .with_extremes(over='#AB0202', under='#D8EEFA'))

seasons=[
    # 'SON',
    # 'DJF',
    # 'MAM',
    'JJA'
    ]

if __name__ == "__main__":

    if False:
        for SEAS in seasons:

            name_models=[
                'MOHC','ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom',
                # "CMCC_VHR",
                "ENSEMBLE",
                # "STATIONS",
                "SPHERA"
                ]

            """
            MAKE MAPS OF QUANTILE for precipitation5x2
            """
            ll_ds=[xr.open_dataset(f"output/{SEAS}/{nm}_q.nc").pr for nm in name_models]
            print(f"Plotting panel extreme precipitation 5x2 for {SEAS}")
            #remapping alla fine piuttosto che all'inizio
            plot_panel_rotated(
            figsize=(6,10),
            nrow=5,ncol=2,
            list_to_plot=ll_ds,
            name_fig=f"PANEL_HP_{SEAS}",
            list_titles=name_models,
            levels=[np.linspace(2,26,9).astype(np.int16) for _ in name_models],
            suptitle=f"Heavy Precipitation 99.9th quantile for {SEAS}",
            name_metric=["[mm/hr]" for _ in name_models],
            SET_EXTENT=False,
            cmap=[cmap_q for _ in name_models],
            SAVE=True
        )

    # """
    # MAKE MAPS OF the 3 metrics for precipitation 5x2
    # """
    # for SEAS in seasons:

    #     name_models=[
    #         # 'MOHC','ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom',
    #         # "CMCC_VHR",
    #         "ENSEMBLE",
    #         # "STATIONS",
    #         "SPHERA"
    #         ]

    #     ll_ds=[xr.open_dataset(f"output/{SEAS}/{nm}_{m}.nc").pr for nm in name_models for m in ['f','i','q']]
    #     print(f"Plotting metrics {SEAS}")
    #     #remapping alla fine piuttosto che all'inizio
    #     plot_panel_rotated(
    #         figsize=(18,8),
    #         nrow=2,ncol=3,
    #         list_to_plot=ll_ds,
    #         name_fig=f"PANEL_METRICS_{SEAS}",
    #         list_titles=["Frequency","Intensity","Heavy Precip.","","",""],
    #         levels=[lvl_f, lvl_i, np.linspace(2,26,9).astype(np.int16),lvl_f, lvl_i, np.linspace(2,26,9).astype(np.int16) ],
    #         suptitle=f"Metrics for {SEAS}",
    #         name_metric=["[fraction]","[mm/hr]","[mm/hr]","[fraction]","[mm/hr]","[mm/hr]"],
    #         SET_EXTENT=False,
    #         cmap=[cmap_f, cmap_i, cmap_q, cmap_f, cmap_i, cmap_q ],
    #         SAVE=False
    #     )
    """
    MAKE MAPS OF the 3 metrics for precipitation 3x2
    """
    if False:
        for SEAS in seasons:

            name_models=[
                # 'MOHC','ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom',
                # "CMCC_VHR",
                "ENSEMBLE",
                # "STATIONS",
                "SPHERA"
                ]

            ll_ds=[xr.open_dataset(f"output/{SEAS}/{nm}_{m}.nc").pr for nm in name_models for m in ['f','i','q']]
            print(f"Plotting metrics rain for {SEAS}")
            #remapping alla fine piuttosto che all'inizio
            plot_panel_rotated(
                figsize=(18,8),
                nrow=2,ncol=3,
                list_to_plot=ll_ds,
                name_fig=f"PANEL_METRICS_{SEAS}",
                list_titles=["Frequency","Intensity","Heavy Precip.","","",""],
                levels=[lvl_f, lvl_i, np.linspace(2,26,9).astype(np.int16),lvl_f, lvl_i, np.linspace(2,26,9).astype(np.int16) ],
                suptitle=f"Metrics for {SEAS}",
                name_metric=["[fraction]","[mm/hr]","[mm/hr]","[fraction]","[mm/hr]","[mm/hr]"],
                SET_EXTENT=False,
                cmap=[cmap_f, cmap_i, cmap_q, cmap_f, cmap_i, cmap_q ],
                SAVE=True
            )


    """
    MAKE MAPS OF QUANTILE for wind 5x2
    """
    if True:
        for SEAS in seasons:
            name_models=[
                'ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom',
                # "CMCC_VHR",
                "ENSEMBLE",
                # "STATIONS",
                "SPHERA"
                ]
            THR=6
            ll_ds=[]

            for nm in name_models:
                if nm == "SPHERA":
                    ds_sph=xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_q.nc").mw.isel(quantile=0)
                    ds_sph['lat']=sea_mask.lat.values
                    ll_ds.append(ds_sph * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
                else:
                    ll_ds.append(xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_q.nc").mw.isel(quantile=0) * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
            
            min_wind=1
            max_wind=11111
            for ds in ll_ds:
                # """massimo dei massimi e minimo dei minimi"""
                if np.nanquantile(ds.values,0.1) > min_wind:
                    min_wind=np.nanquantile(ds.values,0.2)
                
                if np.nanmax(ds.values) < max_wind:
                    max_wind=np.nanmax(ds.values)

            min_wind=5
            print(f"Plotting extreme winds 5x2 for {SEAS}")
            #remapping alla fine piuttosto che all'inizio
            plot_panel_fixbar(
                figsize=(24,12),
                nrow=3,ncol=3,
                list_to_plot=ll_ds,
                name_fig=f"PANEL_HP_mw_{SEAS}",
                list_titles=name_models,
                # levels=[np.linspace(2,26,9).astype(np.int16) for _ in name_models],
                suptitle=f"Heavy Winds 99.9th quantile for {SEAS}",
                name_metric=["[m/s]" for _ in name_models],
                SET_EXTENT=False,
                # cmap=["gist_rainbow" for _ in name_models],
                cmap=[mpl.cm.gist_rainbow.resampled(10) for _ in name_models],
                SAVE=True,
                vmin=[min_wind for _ in name_models],vmax=[max_wind for _ in name_models]
            )


    """
    MAKE MAPS OF the 3 metrics for wind 3x2
    """
    if True:
        for SEAS in seasons:
            name_models=[
                # 'MOHC','ICTP','ETH','KIT','CMCC','CNRM','KNMI','HCLIMcom',
                # "CMCC_VHR",
                "ENSEMBLE",
                # "STATIONS",
                "SPHERA"
                ]
            THR=6
            ll_ds=[]
            for nm in name_models:
                for m in ['m','f','q']:
                    if m=="q":
                        if nm == "SPHERA":
                            ds_sph=xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").isel(quantile=0).mw
                            ds_sph['lat']=sea_mask.lat.values
                            ll_ds.append(ds_sph * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
                        else:
                            ll_ds.append(xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").isel(quantile=0).mw * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
                    elif m=="m":
                        if nm == "SPHERA":
                            ds_sph=xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").mw 
                            ds_sph['lat']=sea_mask.lat.values
                            ll_ds.append(ds_sph * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
                        else:
                            ll_ds.append(xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").mw * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf))
                    elif m=="f":
                        if nm == "SPHERA":
                            ds_sph=xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").mw 
                            ds_sph['lat']=sea_mask.lat.values
                            ll_ds.append(ds_sph * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf) * 100)
                        else:
                            ll_ds.append(xr.open_dataset(f"output/{SEAS}/mw/{nm}_{THR}_{m}.nc").mw * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf) * 100)

            min_mean_speed=1
            max_mean_speed=11111

            min_above_thr=0
            max_above_thr=1111

            min_heavy_win=0
            max_heavy_win=11111

            for idx,ds in enumerate(ll_ds):
                # """massimo dei massimi e minimo dei minimi"""
                if idx in [0,3]:
                    if np.nanquantile(ds.values,0.025) > min_mean_speed:
                        min_mean_speed=np.nanquantile(ds.values,0.025)
                    
                    if np.nanquantile(ds.values,0.95) < max_mean_speed:
                        max_mean_speed=np.nanquantile(ds.values,0.95)
                if idx in [1,4]:
                    if np.nanquantile(ds.values,0.025) > min_above_thr:
                        min_above_thr=np.nanquantile(ds.values,0.025)
                    
                    if np.nanquantile(ds.values,0.95) < max_above_thr:
                        max_above_thr=np.nanquantile(ds.values,0.95)
                if idx in [2,5]:
                    if np.nanquantile(ds.values,0.025) > min_heavy_win:
                        min_heavy_win=np.nanquantile(ds.values,0.025)
                    
                    if np.nanquantile(ds.values,0.95) < max_heavy_win:
                        max_heavy_win=np.nanquantile(ds.values,0.95)

            print(f"Plotting metrics winds for {SEAS}")

            #remapping alla fine piuttosto che all'inizio
            plot_panel_fixbar(
                figsize=(18,8),
                nrow=2,ncol=3,
                list_to_plot=ll_ds,
                name_fig=f"PANEL_METRICS_WIND_{SEAS}",
                list_titles=["Mean","Above Threshold","Heavy Wind","","",""],
                # levels=[np.arange(0.1,5.1,0.5), 
                #         [0,1,3,5,7,9,11,13,15],
                #         # [1,3,5,7,8,9,11,13],
                #         [1,4.8,5.7,7,8,10,12],
                #         np.arange(0.1,5.1,0.5), 
                #         [0,1,3,5,7,9,11,13,15],
                #         [1,4.8,5.7,7,8,10,12]
                #         # ll_ds[5].quantile(q=np.arange(0,1,0.1)).values
                #         ],
                suptitle=f"Metrics for {SEAS}",
                name_metric=["[m/s]","[%]","[m/s]","[m/s]","[%]","[m/s]"],
                SET_EXTENT=False,
                cmap=[mpl.cm.gist_rainbow.resampled(10) for _ in range(len(ll_ds))],
                SAVE=True,
                vmin=[min_mean_speed,min_above_thr,min_heavy_win,min_mean_speed,min_above_thr,min_heavy_win],
                vmax=[max_mean_speed,max_above_thr,max_heavy_win,max_mean_speed,max_above_thr,max_heavy_win]

            )
            # WH=False
            # for nm in name_models:
            #     print(nm)
            #     for idx,SEAS in enumerate(seasons):
            #         if WH:
            #             ds_f=xr.open_dataset(f"output/{SEAS}/{nm}_f_{np.int8(WH)}.nc")
            #             ds_i=xr.open_dataset(f"output/{SEAS}/{nm}_i_{np.int8(WH)}.nc")
            #             ds_q=xr.open_dataset(f"output/{SEAS}/{nm}_q_{np.int8(WH)}.nc")
            #             if nm == "STATIONS":
            #                 ds_q=ds_q.isel(quantile=0)
                        
            #             plot_panel_rotated(
            #                 figsize=(8,8),
            #                 nrow=1,ncol=3,
            #                 list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
            #                 name_fig=f"PANEL_{nm}_{SEAS}_WH",
            #                 list_titles=["Frequency","Intensity","Heavy Prec."],
            #                 levels=[lvl_f,lvl_i,lvl_q],
            #                 suptitle=f"{nm}'s metrics for {SEAS}",
            #                 name_metric=["[fraction]","[mm/h]","[mm/h]"],
            #                 SET_EXTENT=False,
            #                 cmap=[cmap_f,cmap_i,cmap_q],
            #                 SAVE=False
            #             )

            #         else:
            #             ds_f=xr.open_dataset(f"output/{SEAS}/{nm}_f.nc")
            #             #Put all zeros to np.nan
            #             ds_f=xr.where(ds_f.pr == 0, np.nan,ds_f.pr).to_dataset(name='pr')
            #             ds_i=xr.open_dataset(f"output/{SEAS}/{nm}_i.nc")
            #             ds_q=xr.open_dataset(f"output/{SEAS}/{nm}_q.nc")
            #             if nm == "STATIONS":
            #                 ds_q=ds_q.isel(quantile=0)

            #             plot_panel_rotated(
            #                 figsize=(13,3.5),
            #                 nrow=1,ncol=3,
            #                 list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
            #                 name_fig=f"PANEL_{nm}_{SEAS}",
            #                 list_titles=["Frequency","Intensity","Heavy Prec."],
            #                 levels=[lvl_f,lvl_i,lvl_q],
            #                 suptitle=f"{nm}'s metrics for {SEAS}",
            #                 name_metric=["[fraction]","[mm/h]","[mm/h]"],
            #                 SET_EXTENT=False,
            #                 cmap=[cmap_f,cmap_i,cmap_q],
            #                 SAVE=False

            #             )

            #         # st_al=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/*.nc")
            #         # st_al=st_al.isel(time=st_al['time.year'].isin(np.arange(2000,2009))).load()
                    
            #         # st_fre,st_int,st_q99=compute_metrics_stat(get_season(st_al,"JJA"))
                    
            #         # ens_f_tr=ens_f.sel(lon=st_fre.lon,lat=st_fre.lat)
            #         # ens_i_tr=ens_i.sel(lon=st_int.lon,lat=st_int.lat)
            #         # ens_q_tr=ens_q.sel(lon=st_q99.lon,lat=st_q99.lat)

            #         # bias_f = (ens_f_tr - st_fre) / st_fre * 100
            #         # bias_i = (ens_i_tr - st_int) / st_int * 100
            #         # bias_q = (ens_q_tr - st_q99) / st_q99 * 100
                    
            #         # np.nanmean(bias_f.freq)
            #         # np.nanmean(bias_i.int)
            #         # np.nanmean(bias_q.q)

            #         # lvl_q=np.arange(2,19,2)
            #         # lvl_q=9
            #         # plot_panel_rotated(
            #         #     nrow=1,ncol=3,
            #         #     list_to_plot=[ds_f.pr,ds_i.pr,ds_q.pr],
            #         #     name_fig=f"PANEL_{nm}_{SEAS}",
            #         #     list_titles=["Frequency","Intensity","Heavy Prec."],
            #         #     levels=[lvl_f,lvl_i,lvl_q],
            #         #     suptitle=f"Ensemble's metrics for {SEAS}",
            #         #     name_metric=["[fraction]","[mm/h]","[mm/h]"],
            #         #     SET_EXTENT=False,
            #         #     cmap=[cmap_f,cmap_i,cmap_q]
            #         # )

            #         # plot_panel_rotated(
            #         #     nrow=1,ncol=1,
            #         #     list_to_plot=[ds_q.isel(quantile=0).pr],
            #         #     name_fig=f"PANEL_{SEAS}",
            #         #     list_titles=["Heavy Prec."],
            #         #     levels=[lvl_f,lvl_i,lvl_q],
            #         #     suptitle=f"Station {SEAS}",
            #         #     name_metric=["[mm/h]"],
            #         #     SET_EXTENT=False,
            #         #     #cmap=[cmap_q]
            #         # )
            #         # print(ds_f)



    print("FINISHED MAKE EVENTS")
