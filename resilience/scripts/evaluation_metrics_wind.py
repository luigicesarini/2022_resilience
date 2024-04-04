#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import seaborn as sns
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


os.chdir("/home/lcesarini/2022_resilience/")
seasons=['SON','DJF','MAM','JJA']
# seasons=['DJF','JJA']
list_ms=['Mean Speed','Frequency','Heavy Winds']
abbr_ms=['m','f','q']

list_mdl=['ETH','ICTP','HCLIMcom','CNRM','KNMI','KIT','CMCC','ENSEMBLE','SPHERA']
sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()
ENV_VAR='mw'


PLOT_BOXPLOTS=False
if __name__ == "__main__":
    """RELATIVE BIAS"""
    # THR=np.int8(sys.argv[1])
    # nm=str(sys.argv[2])
    array_heatmap=np.zeros(shape=(len(seasons),3))
    array_boxplot=np.zeros(shape=(42976,len(seasons)*3))
    counter=0
    THR=6
    for nm in list_mdl:
        for i,SEAS in enumerate(seasons):
            ds_m=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/{nm}_{THR}_m.nc") * sea_mask.sftlf.values
            ds_f=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/{nm}_{THR}_f.nc") * sea_mask.sftlf.values
            ds_q=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/{nm}_{THR}_q.nc").isel(quantile=0) * sea_mask.sftlf.values

            plot_panel_rotated(
                figsize=(12,4),nrow=1,ncol=3,
                list_to_plot=[ds_m.mw,ds_f.mw * 100,ds_q.mw],
                name_fig=f"PANEL_{nm}_{SEAS}_{ENV_VAR}_{THR}",
                list_titles=[f"Mean speed m/s",
                                f"Frequency > {THR} m/s",
                                f"Heavy Winds 99.9pct."],
                levels=[np.arange(0.5,6.1,0.5),
                        np.arange(0.025,0.31,0.025) * 100,
                        np.arange(2,14,2)],
                suptitle=f"{nm}'s metrics for {SEAS}",
                name_metric=["[m/s]","[%]","[m/s]"],
                SET_EXTENT=False,
                cmap=["RdYlGn_r","RdYlGn_r","RdYlGn_r"],
                SAVE=False
            )

    for i,SEAS in enumerate(seasons):
        for j,(idx,metrics) in enumerate(zip(abbr_ms,list_ms)):

            ds_cpm=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/ENSEMBLE_{THR}_{idx}.nc") * sea_mask.sftlf.values #xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc")
            ds_rea=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/SPHERA_{THR}_{idx}.nc") * sea_mask.sftlf.values #xr.open_dataset(f"output/{SEAS}/CMCC_VHR_{idx}.nc")
            ds_rea=ds_rea.assign_coords({"lon":ds_cpm.lon.values,"lat":ds_cpm.lat.values})
            rel_bias= ( ds_cpm - ds_rea ) / ds_rea
            
            array_heatmap[i,j]=np.nanmean(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan)) * 100
            array_boxplot[:,counter]   = (rel_bias.mw * 100).values.reshape(-1)
            # array_boxplot[:,counter+1] = f"{metrics} {SEAS}"
            #print(f"{metrics} bias in {SEAS}: {np.nanmean(rel_bias.pr) * 100:.2f}%")
            counter+=1
            print(f"Corr in {SEAS} for metric {idx}: {pattern_correlation(ds_cpm,ds_rea,type='centred')}")


    np.savetxt("output/array_heatmap_wind.txt",array_heatmap) 
    cmap = (mpl.colors.ListedColormap(['#7E1104',
                                        '#E33434', 
                                        '#F58080', 
                                        '#F8BCBC', 
                                        '#FBE2E2', 
                                        'white',
                                        '#D4F7FA',
                                        '#90DEF8',
                                        '#7BB2ED',
                                        '#262BBD',
                                        '#040880'
                                            ]))
    array_heatmap=np.loadtxt(f"output/array_heatmap_wind.txt")
    df=pd.DataFrame(array_heatmap,columns=['Freq','Int','Quantile'],index=seasons)
    pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
    pcm.set(xlabel="", ylabel="")
    pcm.xaxis.tick_top()
    plt.title("Biases for DJF and JJA CPM vs SPHERA, sea ir removed")
    plt.savefig(f"figures/heatmap_bias_wind.png")
    plt.close()      
    if PLOT_BOXPLOTS:
        df_box=pd.DataFrame(array_boxplot,columns=[f'{m} {s}' for s in seasons for m in list_ms])
        # df_box['col']='Heavy Prec, JJA'
        fig=plt.figure(figsize=(14,8))
        ax=plt.axes()
        # print(df_box.melt())
        sns.set(style="darkgrid")
        sns.boxplot(y="value", 
                    x="variable",
                    data=df_box.melt(), 
                    palette=["green","red","blue"], 
                    width=0.25,
                    ax=ax)
        ax.hlines(y=-5, xmin=-1,xmax=6,linestyles='dashed',color='red')
        ax.hlines(y=25, xmin=-1,xmax=6,linestyles='dashed',color='red')
        ax.set_xlabel(f"")
        ax.set_ylabel(f"Relative Bias [%]")
        ax.set_ylim(-100,100)
        ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
        plt.title("Relative bias of CPM vs SPHERA",pad=20,fontsize=20)
        plt.show()
        # plt.savefig("figures/boxplot_rel_bias_winds.png")
        # plt.close()
    
    # # plt.figure(figsize=(6,4))
    # # plt.cm.Blues
    
    # # cmap = (mpl.colors.ListedColormap(['#7E1104',
    # #                                    '#E33434', 
    # #                                    '#F58080', 
    # #                                    '#F8BCBC', 
    # #                                    '#FBE2E2', 
    # #                                    'white',
    # #                                    '#D4F7FA',
    # #                                    '#90DEF8',
    # #                                    '#7BB2ED',
    # #                                    '#262BBD',
    # #                                    '#040880'
    # #                                     ]))
    

    # # df=pd.DataFrame(array_heatmap,columns=['Freq','Int','Quantile'],index=['SON','DJF','MAM','JJA'])
    # # pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5)
    # # pcm.set(xlabel="", ylabel="")
    # # pcm.xaxis.tick_top()

    # # xr.DataArray(array_heatmap).plot.pcolormesh(
    # #     levels=[-80,-60,-40,-25,-5,25,40,60,80,100],
        
    # #     )
    # # plt.savefig("figures/heatmap.png")
    # # plt.close()

    # #STATIONS

    # # for m in ["f","i","q"]:
    # #     if m != 'q':
    # #         ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc")
    # #     else:
    # #         ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc").isel(quantile=0)
        
    # #     ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{m}.nc")



    # #     cpm_tri=ds_cpm.isel(lon=ds_cpm.lon.isin(ds_sta.lon),lat=ds_cpm.lat.isin(ds_sta.lat))
        
    # #     bias=((cpm_tri - ds_sta) / ds_sta) * 100
    # #     bias.pr.plot.pcolormesh(levels=11,cmap='RdBu')
    # #     plt.savefig(f"figures/bias_triveneto_{m}.png")
    # #     plt.close()

    # #     print(f"{m} bias in {SEAS}: {np.nanmean(bias.pr):.2f}%")

    # #     bias_reshaped=bias.pr.values.reshape(-1,1)
    # #     np.nanmean(bias_reshaped[np.isfinite(bias_reshaped)])

    # #     xr.where(np.isfinite(bias.values),bias,np.nan)


    """CORRELATION"""
