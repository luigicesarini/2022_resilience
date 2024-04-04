#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import metview as mv
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
"""
CREATE PARSER 
"""
parser = argparse.ArgumentParser(description='Process the selection of the reanalysis datasets')
parser.add_argument('-vhr', metavar='Very high resolution', type=str,choices=['SPHERA','CMCC_VHR'],
                    help='Which reanalysis to compare')

args = parser.parse_args()

VHR_PRODUCT=args.vhr
VHR_PRODUCT='SPHERA'
PATH_OUTPUT="/home/lcesarini/2022_resilience/output"

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI','CMCC','KIT','ENSEMBLE']
seasons=['SON','DJF','MAM','JJA']
list_ms=['Frequency','Intensity','Heavy Prec.']
abbr_ms=['f','i','q']

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

# cmap_f.set_bad("white")
# cmap_i.set_bad("white")
# cmap_q.set_bad("white")


shp_triveneto = gpd.read_file("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]


sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")


PLOT_BOXPLOTS=True
if __name__ == "__main__":
    """RELATIVE BIAS"""
    array_heatmap=np.zeros(shape=(4,3))
    array_boxplot=np.zeros(shape=(42976,12)) * np.nan
    counter=0
    
    for i,SEAS in enumerate(seasons):
        for j,(idx,metrics) in enumerate(zip(abbr_ms,list_ms)):
            
            ds_cpm=xr.concat([xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/{nm}_{idx}.nc") for nm in name_models],name_models).rename({'concat_dim':'name'})
            ds_cpm=ds_cpm * sea_mask.sftlf
            # ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc") * sea_mask.sftlf
            ds_rea=xr.open_dataset(f"output/{SEAS}/{VHR_PRODUCT}_{idx}.nc") * sea_mask.sftlf.values
            
            rel_bias= ( ds_cpm - ds_rea ) / ds_rea
            rel_bias=rel_bias.sel(name=name_models[:-1])

            array_heatmap[i,j]=np.nanmedian(rel_bias.pr.values[np.isfinite(rel_bias.pr)]) * 100
            finite_res=rel_bias.pr.values[np.isfinite(rel_bias.pr)].shape[0]
            array_boxplot[:finite_res,counter]  = (rel_bias.pr.values[np.isfinite(rel_bias.pr)] * 100).reshape(-1)
            # array_boxplot[:,counter+1] = f"{metrics} {SEAS}"
            #print(f"{metrics} bias in {SEAS}: {np.nanmean(rel_bias.pr) * 100:.2f}%")
            counter+=1

            plot_panel_rotated(
                figsize=(16,16),
                nrow=3,ncol=3,
                list_to_plot=[rel_bias.pr.sel(name=nm) for nm in name_models],
                name_fig=f"EVERYTHING",
                list_titles= [f"{nm}" for nm in name_models],
                levels=[np.arange(-0.5,0.51,0.05) for _ in np.arange(len(name_models))],
                suptitle=f"",
                # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
                name_metric=["[%]" for _ in np.arange(len(name_models))],
                SET_EXTENT=False,
                cmap=["PuOr" for _ in np.arange(len(name_models))],
                SAVE=False
            )

            plot_panel_rotated(
                figsize=(12,6),
                nrow=1,ncol=2,
                list_to_plot=[rel_bias.pr.sel(name=name_models[:-1]).mean(dim='name'),
                              rel_bias.pr.sel(name=name_models[:-1]).std(dim='name')],
                name_fig=f"spread_bias",
                list_titles= ["MEAN of the ensemble","SPREAD of the ensemble"],
                levels=[np.arange(-0.5,0.51,0.05) ,np.arange(0,0.251,0.025)],
                suptitle=f"",
                # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
                name_metric=["[%]","[%]"],
                SET_EXTENT=False,
                cmap=["PuOr","YlOrBr"],
                SAVE=False
            )
    np.savetxt(f"output/array_heatmap_{VHR_PRODUCT}.txt",array_heatmap)       
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
        ax.hlines(y=-5, xmin=-1,xmax=12,linestyles='dashed',color='red')
        ax.hlines(y=25, xmin=-1,xmax=12,linestyles='dashed',color='red')
        ax.set_xlabel(f"")
        ax.set_ylabel(f"Relative Bias [%]")
        ax.set_ylim(-100,100)
        ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
        plt.title(f"Relative bias of CPM vs {VHR_PRODUCT}",pad=20,fontsize=20)
        plt.savefig(f"figures/boxplot_rel_bias_{VHR_PRODUCT}.png")
        plt.close()
    
    # plt.figure(figsize=(6,4))
    # plt.cm.Blues
    
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
    # name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI']

    # ensemble=[xr.open_dataset(f"output/JJA/{mdl}_q.nc") for mdl in name_models]

    # cnc_ens=xr.concat([x.drop_vars("quantile") for x in ensemble],name_models)

    # ens_mean   = cnc_ens.mean(dim='concat_dim')
    # ens_spread = cnc_ens.std(dim='concat_dim')
    # spread_mio = np.std(cnc_ens.pr,axis=0)
    # ens_spread.quantile(q=0.99)

    # plot_panel_rotated(
    #     nrow=1,
    #     ncol=3,
    #     list_to_plot=[ens_mean.pr,ens_spread.pr,spread_mio],
    #     name_fig="mena_spres_metview",
    #     list_titles=["Mean","Spread","Spread"],
    #     name_metric=["[mm/h]","[mm/h]","[mm/h]"],
    #     levels=[lvl_q,np.linspace(0,5,8),np.linspace(0,5,8)],
    #     cmap=[cmap_q,'RdBu',"RdBu"],
    #     SET_EXTENT=False,
    # )

    array_heatmap=np.loadtxt(f"output/array_heatmap_{VHR_PRODUCT}.txt")
    df=pd.DataFrame(array_heatmap,columns=['Freq','Int','Quantile'],index=['SON','DJF','MAM','JJA'])
    pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
    pcm.set(xlabel="", ylabel="")
    pcm.xaxis.tick_top()

    # xr.DataArray(array_heatmap).plot.pcolormesh(
    #     levels=[-80,-60,-40,-25,-5,25,40,60,80,100],
        
    #     )
    plt.savefig(f"figures/heatmap_{VHR_PRODUCT}.png")
    plt.close()

    #STATIONS
    array_heatmap=np.zeros(shape=(4,3))

    for i_s,SEAS in enumerate(seasons):
        for i_m,m in enumerate(["f","i","q"]):
            if m != 'q':
                ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc")
                ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{m}.nc") 

            else:
                ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc").isel(quantile=0)
                ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{m}.nc")

            if VHR_PRODUCT=='SPHERA':
                ds_rea=xr.open_dataset(f"output/{SEAS}/{VHR_PRODUCT}_{m}.nc")
            else:
                ds_rea=xr.open_dataset(f"output/{SEAS}/{VHR_PRODUCT}_{m}.nc")


            mask=xr.open_dataset("data/mask_stations_nan_common.nc")

            rea_tri=(ds_rea*mask.mask).isel(lon=ds_rea.lon.isin(ds_sta.lon),lat=ds_rea.lat.isin(ds_sta.lat))

            cpm_tri=(ds_cpm*mask.mask).isel(lon=ds_cpm.lon.isin(ds_sta.lon),lat=ds_cpm.lat.isin(ds_sta.lat))
            

            #BIAS CPM-STATION
            bias=((cpm_tri - ds_sta) / ds_sta) * 100
            # BIAS REANALYSIS PRODCUT-STATION
            bias_rea_st=((rea_tri - ds_sta) / ds_sta) * 100
            # BIAS CPM-REANALYSIS PRODUCT
            bias_rea_cpm=((cpm_tri - rea_tri) / rea_tri) * 100
                     
            
            #COMPUTE HEATMAP BIAS CPM-STATION
            array_heatmap[i_s,i_m]=np.nanmedian(bias.pr.values[np.isfinite(bias.pr)]) 


            col=mpl.cm.get_cmap("PuOr",12)


            cmap_q = (mpl.colors.ListedColormap([mpl.colors.rgb2hex(col(i)) for i in np.arange(1,12)])
                    .with_extremes(over=mpl.colors.rgb2hex(col(12)), under=mpl.colors.rgb2hex(col(0))))
            cmap_q.set_bad("white")
            fig=plt.figure(figsize=(16,12))
            ax=plt.axes(projection=ccrs.PlateCarree())
            pcm=bias.pr.plot.pcolormesh(levels=np.arange(-50,51,10),
                                        cmap=cmap_q,ax=ax,
                                        add_colorbar=False)
            shp_triveneto.boundary.plot(edgecolor='red',ax=ax,linewidths=1.5)
            ax.set_title("Station vs model's ensemble",fontsize=20)
            ax.set_extent([10.39,13.09980774,44.70745754,47.09988785])
            gl = ax.gridlines(
                    draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
                )
            gl.xlocator = mpl.ticker.FixedLocator([10.5, 11.5, 12.5])
            gl.ylocator = mpl.ticker.FixedLocator([45, 46, 47])
            gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            cbar=fig.colorbar(pcm, ax=ax, 
                                extend='both', 
                                orientation='vertical',
                                shrink=1,
                                pad = 0.075)
            cbar.ax.tick_params(labelsize=30)
            cbar.ax.set_ylabel('[%]',fontsize=25,rotation=0)
            fig.suptitle(f"Bias Heavy precipitation (p99.9) on {SEAS}",fontsize=30)
            plt.savefig(f"figures/bias_triveneto_{m}_{SEAS}.png")
            plt.close()

            fig=plt.figure(figsize=(16,12))
            ax=plt.axes(projection=ccrs.PlateCarree())
            pcm=xr.where(np.isnan(bias_rea_st.pr),np.nan,bias_rea_st.pr).plot.pcolormesh(levels=np.arange(-50,51,10),
                                        cmap=cmap_q,ax=ax,
                                        add_colorbar=False)
            shp_triveneto.boundary.plot(edgecolor='red',ax=ax,linewidths=1.5)
            ax.set_title(f"Station vs {VHR_PRODUCT}",fontsize=20)
            ax.set_extent([10.39,13.09980774,44.70745754,47.09988785])
            gl = ax.gridlines(
                    draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
                )
            gl.xlocator = mpl.ticker.FixedLocator([10.5, 11.5, 12.5])
            gl.ylocator = mpl.ticker.FixedLocator([45, 46, 47])
            gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            cbar=fig.colorbar(pcm, ax=ax, 
                                extend='both', 
                                orientation='vertical',
                                shrink=1,
                                pad = 0.075)
            cbar.ax.tick_params(labelsize=30)
            cbar.ax.set_ylabel('[%]',fontsize=25,rotation=0)
            fig.suptitle(f"Bias Heavy precipitation (p99.9) on {SEAS}",fontsize=30)
            plt.savefig(f"figures/bias_triveneto_{VHR_PRODUCT}_st_{m}_{SEAS}.png")
            plt.close()

            fig=plt.figure(figsize=(16,12))
            ax=plt.axes(projection=ccrs.PlateCarree())
            pcm=xr.where(np.isfinite(bias_rea_cpm.pr),bias_rea_cpm.pr,np.nan).plot.pcolormesh(levels=np.arange(-50,51,10),
                                        cmap=cmap_q,ax=ax,
                                        add_colorbar=False)
            shp_triveneto.boundary.plot(edgecolor='red',ax=ax,linewidths=1.5)
            ax.set_title(f"CPM vs {VHR_PRODUCT}",fontsize=20)
            ax.set_extent([10.39,13.09980774,44.70745754,47.09988785])
            gl = ax.gridlines(
                    draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
                )
            gl.xlocator = mpl.ticker.FixedLocator([10.5, 11.5, 12.5])
            gl.ylocator = mpl.ticker.FixedLocator([45, 46, 47])
            gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
            cbar=fig.colorbar(pcm, ax=ax, 
                                extend='both', 
                                orientation='vertical',
                                shrink=1,
                                pad = 0.075)
            cbar.ax.tick_params(labelsize=30)
            cbar.ax.set_ylabel('[%]',fontsize=25,rotation=0)
            fig.suptitle(f"Bias Heavy precipitation (p99.9) on {SEAS}",fontsize=30)
            plt.savefig(f"figures/bias_triveneto_{VHR_PRODUCT}_cpm_{m}_{SEAS}.png")
            plt.close()

            
            # print(f"{m} bias in {SEAS}: {np.nanmean(bias.pr):.2f}%")
            # print(f"{m} bias in {SEAS}: {np.nanmean(bias_rea_st.pr):.2f}%")
            # print(f"{m} bias in {SEAS}: {np.nanmean(bias_rea_cpm.pr):.2f}%")
            
            
            df_box=pd.DataFrame(np.concatenate([bias.pr.values.reshape(-1,1),
                                                bias_rea_st.pr.values.reshape(-1,1),
                                                bias_rea_cpm.pr.values.reshape(-1,1)],
                                                axis=1),
                                columns=[f'ST_CPM',f'ST_{VHR_PRODUCT}',f'{VHR_PRODUCT}_CPM'])
            # df_box['col']='Heavy Prec, JJA'
            fig=plt.figure(figsize=(14,8))
            ax=plt.axes()
            # print(df_box.melt())
            sns.set(style="darkgrid")
            sns.boxplot(y="value", 
                        x="variable",
                        data=df_box.melt()[~pd.isnull(df_box.melt().value)], 
                        palette=["green","red",'blue'], 
                        notch=True,
                        width=0.25,
                        flierprops={"marker": "*"},
                        medianprops={"color": "coral"},
                        ax=ax)
            ax.hlines(y=-5, xmin=-1,xmax=3,linestyles='dashed',color='red')
            ax.hlines(y=25, xmin=-1,xmax=3,linestyles='dashed',color='red')
            ax.set_xlabel(f"")
            ax.set_ylabel(f"Relative Bias [%]")
            ax.set_ylim(-100,100)
            # ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
            plt.title(f"Relative bias of Station vs CPM vs {VHR_PRODUCT} {SEAS}",pad=20,fontsize=20)
            plt.savefig(f"figures/boxplot_rel_bias_comparison_{VHR_PRODUCT}_{SEAS}_{m}.png")
            plt.close()



            # bias_reshaped=bias.pr.values.reshape(-1,1)
            # np.nanmean(bias_reshaped[np.isfinite(bias_reshaped)])

            # xr.where(np.isfinite(bias.values),bias,np.nan)
    np.savetxt(f"output/array_heatmap_station.txt",array_heatmap)
    plot_heatmap(f"output/array_heatmap_station.txt")       
