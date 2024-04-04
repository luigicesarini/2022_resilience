#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience/")
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


os.chdir("/mnt/beegfs/lcesarini/2022_resilience/")
seasons=['SON','DJF','MAM','JJA']
# seasons=['DJF','JJA']
list_ms=['Mean Speed','Frequency','Heavy Winds']
abbr_ms=['m','f','q']   

list_mdl=['ETH','ICTP','HCLIMcom','CNRM','KNMI','KIT','CMCC','ENSEMBLE','SPHERA']
sea_mask=xr.open_dataset("/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")
oro=xr.load_dataset("/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/KNMI/orog_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc")


# cpm=[xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/mw*.nc").sel(lon=9,lat=45.25,method='nearest').mw.values for mdl in list_mdl[:7]]
# sph=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*.nc").load()

# sph_jja=sph.sel(time=sph["time.season"].isin("JJA"))
# sph_djf=sph.sel(time=sph["time.season"].isin("DJF"))

# ax=plt.axes(projection=ccrs.PlateCarree())
# sph.isel(time=12).mw.plot.pcolormesh(ax=ax)
# ax.scatter(x=9,y=45.25,color='red')
# ax.scatter(x=7.75,y=45.25,color='blue')
# ax.scatter(x=10.25,y=45.25,color='green')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS)
# plt.show()
# ds_cpm=xr.open_dataset(f"/mnt/beegfs/lcesarini/2022_resilience/output/JJA/ENSEMBLE_q.nc") * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf) #xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc")
# def _count_threshold_periods(arr,thr):
#         above_threshold_periods = []
#         current_period = []
#         for element in arr:
#             if element > thr:
#                 current_period.append(element)
#             else:
#                 if current_period:
#                     above_threshold_periods.append(current_period)
#                     current_period = []
        
#         # Check if the last period extends beyond the end of the array
#         if current_period:
#             above_threshold_periods.append(current_period)

#         n_event=[len(x) for x in above_threshold_periods]
#         max_int=[max(x) for x in above_threshold_periods]
#         mean_int=[np.mean(x) for x in above_threshold_periods]
        
#         return n_event,max_int,mean_int


# h,b,n=_count_threshold_periods(sph_jja.sel(lon=7.75,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_jja.sel(lon=7.75,lat=45.25,method='nearest').mw.values,q=0.9))
# s,d,f=_count_threshold_periods(sph_jja.sel(lon=9,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_jja.sel(lon=9,lat=45.25,method='nearest').mw.values,q=0.9))
# e,r,t=_count_threshold_periods(sph_jja.sel(lon=10.25,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_jja.sel(lon=10.25,lat=45.25,method='nearest').mw.values,q=0.9))

# len(h),len(s),len(e)

# h,b,n=_count_threshold_periods(sph_djf.sel(lon=7.75,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_djf.sel(lon=7.75,lat=45.25,method='nearest').mw.values,q=0.9))
# s,d,f=_count_threshold_periods(sph_djf.sel(lon=9,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_djf.sel(lon=9,lat=45.25,method='nearest').mw.values,q=0.9))
# e,r,t=_count_threshold_periods(sph_djf.sel(lon=10.25,lat=45.25,method='nearest').mw.values,
#                                np.nanquantile(sph_djf.sel(lon=10.25,lat=45.25,method='nearest').mw.values,q=0.9))

# len(h),len(s),len(e)

# import pickle
# with open('/mnt/data/lcesarini/EVENTS/mw/SPHERA_len_events_90_JJA.pkl',"rb") as file:
#     len_per_above_threshold=pickle.load(file)

# import seaborn as sns

# plt.plot(np.sort(sph.sel(lon=7.75,lat=45.25,method='nearest').mw.values),label='Not gap')
# plt.plot(np.sort(sph.sel(lon=9,lat=45.25,method='nearest').mw.values),label='Gap')
# plt.plot(np.sort(sph.sel(lon=10.5,lat=45.25,method='nearest').mw.values),label='Not gap2')
# # plt.plot(np.sort(cpm.sel(lon=9,lat=45,method='nearest').mw.values),label='Gap')
# plt.scatter(x=87111*0.9,y=np.nanquantile(sph.sel(lon=7.75,lat=45.25,method='nearest').mw.values,q=0.9),label='90 NotGap')
# plt.scatter(x=87111*0.9,y=np.nanquantile(sph.sel(lon=9,lat=45.25,method='nearest').mw.values,q=0.9),label='90 Gap')
# # plt.plot(np.sort(cpm.sel(lon=7.75,lat=45,method='nearest').mw.values),label='Not gap')
# plt.scatter(x=87111*0.9,y=np.nanquantile(sph.sel(lon=10.25,lat=45.25,method='nearest').mw.values,q=0.9),label='90 NotGap2')
# plt.legend()
# plt.show()

# X=xr.merge([ds_cpm,oro])

# (ds_cpm.pr * xr.where((oro.orog > 500) & (oro.orog <= 1000),1,np.nan)).plot.pcolormesh()
# cpm.isel(time=1).mw.plot.pcolormesh()
# plt.savefig("XXX.png")
# plt.close()
cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()
ENV_VAR='mw'


PLOT_BOXPLOTS=True
if __name__ == "__main__":
    """RELATIVE BIAS WIND"""
    # THR=np.int8(sys.argv[1])
    # nm=str(sys.argv[2])
    THR=6

    oro_1000=xr.where(oro.orog <= 1000,1,np.nan)
    oro_2000=xr.where((oro.orog > 1000) & (oro.orog <= 2000),1,np.nan)
    oro_3000=xr.where((oro.orog > 2000) & (oro.orog <= 3000),1,np.nan)
    oro_4000=xr.where(oro.orog > 3000,1,np.nan)

    for lvl,ele in zip([1000,2000,3000,4000],[oro_1000,oro_2000,oro_3000,oro_4000]):
        array_heatmap=np.zeros(shape=(len(seasons),3))
        array_boxplot=np.zeros(shape=(42976,len(seasons)*3))
        counter=0
        for i,SEAS in enumerate(seasons):
            for j,(idx,metrics) in enumerate(zip(abbr_ms,list_ms)):

                ds_cpm=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/ENSEMBLE_{THR}_{idx}.nc") * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf) #xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc")
                ds_rea=xr.open_dataset(f"output/{SEAS}/{ENV_VAR}/SPHERA_{THR}_{idx}.nc")  #xr.open_dataset(f"output/{SEAS}/CMCC_VHR_{idx}.nc")
                ds_rea=ds_rea.assign_coords({"lon":ds_cpm.lon.values,"lat":ds_cpm.lat.values}) * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf)

                # ds_rea.lat.values == ds_cpm.lat.values
                # ds_rea.lon.values == ds_cpm.lon.values
                if idx == "f":
                    ds_cpm=xr.where(ds_cpm.mw < 0.2, np.nan, ds_cpm.mw).to_dataset(name="mw")
                    ds_rea=xr.where(ds_rea.mw < 0.2, np.nan, ds_rea.mw).to_dataset(name="mw")
                elif idx == "m":
                    ds_cpm=xr.where(ds_cpm.mw < 1, np.nan, ds_cpm.mw).to_dataset(name="mw")
                    ds_rea=xr.where(ds_rea.mw < 1, np.nan, ds_rea.mw).to_dataset(name="mw")
                elif idx == "q":
                    ds_cpm=xr.where(ds_cpm.mw < 2, np.nan, ds_cpm.mw).to_dataset(name="mw")
                    ds_rea=xr.where(ds_rea.mw < 2, np.nan, ds_rea.mw).to_dataset(name="mw")


                rel_bias= ( ds_cpm * ele  - ds_rea * ele ) / (ds_rea * ele)
                
                # rel_bias.mw.plot()
                # plt.show()
                if idx != "f":
                    iii=np.nanargmax(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan).values)
                # np.nanargmax(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan).values.reshape(-1))
                #     print(f"{metrics} {SEAS} {iii}")
                #     print(ds_cpm.mw.values.reshape(-1)[iii],ds_rea.mw.values.reshape(-1)[iii])
                # print(f"{lvl} {np.nanmean(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan)) * 100}")
                array_heatmap[i,j]=np.nanmean(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan)) * 100
                array_boxplot[:,counter]   = (rel_bias.mw * 100).values.reshape(-1)
                # array_boxplot[:,counter+1] = f"{metrics} {SEAS}"
                #print(f"{metrics} bias in {SEAS}: {np.nanmean(rel_bias.pr) * 100:.2f}%")
                np.save(f'output/array_boxplot_wind_{lvl}.npy',array_boxplot)
                np.save(f'output/array_cpm_{metrics}_{lvl}.npy',(ds_cpm * ele).mw.values.reshape(-1))
                np.save(f'output/array_rea_{metrics}_{lvl}.npy',(ds_rea * ele).mw.values.reshape(-1))
                counter+=1
                # print(f"Corr in {SEAS} for metric {idx}: {pattern_correlation(ds_cpm,ds_rea,type='centred')}")
        """
        PLOT SCATTER WITH ELEVATION
        """
        ll_df_cpm=[]
        ll_df_rea=[]

        for lvl in [1000,2000,3000,4000]:
            arr_cpm=np.load(f'output/array_cpm_{metrics}_{lvl}.npy')
            arr_rea=np.load(f'output/array_rea_{metrics}_{lvl}.npy')
            df_cpm=pd.DataFrame(arr_cpm,columns=[f'{metrics}_CPM'])
            df_rea=pd.DataFrame(arr_rea,columns=[f'{metrics}_REA'])
            # df_cpm=df_box.melt()
            # df_rea=df_box.melt()
            df_cpm['Elevation']=lvl
            # df_rea['Elevation']=lvl
            ll_df_cpm.append(df_cpm)
            ll_df_rea.append(df_rea)

        df_cpm_all=pd.concat(ll_df_cpm)
        df_rea_all=pd.concat(ll_df_rea)

        dffd=pd.concat([df_cpm_all,df_rea_all],axis=1)

        colors=["green","red","blue","brown"]
        fig,ax=plt.subplots(figsize=(14,8),nrows=2,ncols=2,sharex=True)
        ax=ax.flatten()
        for i in range(4):
            dffd=pd.concat([ll_df_cpm[i],ll_df_rea[i]],axis=1)
            # print(df_box.melt())
            sns.set(style="white")
            sns.scatterplot(
                        y="Heavy Winds_REA", 
                        x="Heavy Winds_CPM",#whis=float('inf'),
                        # hue='Elevation',
                        data=dffd, 
                        c=colors[i],
                        # palette=colors[i], 
                        #width=0.5,
                        alpha=0.8,
                        ax=ax[i])
            
            ax[i].plot([0,15],[0,15],linestyle='dashed',c='black',alpha=0.75)
            ax[i].grid(True, linestyle='--', alpha=0.55)
            if i != 3:
                ax[i].set_title(f"Bias up to {[1000,2000,3000][i]}m")
            else:
                ax[i].set_title(f"Bias from 3000 m")
        plt.show()
        # print(np.nanmax(array_boxplot,axis=0))
        np.savetxt(f"output/array_heatmap_wind_{lvl}.txt",array_heatmap) 
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
        array_heatmap=np.loadtxt(f"output/array_heatmap_wind_{lvl}.txt")
        df=pd.DataFrame(array_heatmap,columns=['Mean Speed','Above Threshold','Heavy Winds'],index=seasons)
        pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
        pcm.set(xlabel="", ylabel="")
        pcm.xaxis.tick_top()
        plt.title(f"Biases of CPM ensemble vs SPHERA at {lvl-1000}-{lvl}m")
        plt.savefig(f"figures/heatmap_bias_wind_{lvl}.png")
        plt.close()      
        if PLOT_BOXPLOTS:
            ll_df_box=[]
            for lvl in [1000,2000,3000,4000]:
                array_boxplot=np.load(f'output/array_boxplot_wind_{lvl}.npy')
                df_box=pd.DataFrame(array_boxplot,columns=[f'{m} {s}' for s in seasons for m in list_ms])
                df_box=df_box.melt()
                df_box['Elevation']=lvl
                ll_df_box.append(df_box)
                # df_box['col']='Heavy Prec, JJA'

            df_box=pd.concat(ll_df_box)
            fig=plt.figure(figsize=(14,8))
            ax=plt.axes()
            # print(df_box.melt())
            sns.set(style="white")
            sns.scatterplot(y="value", 
                        x="variable",#whis=float('inf'),
                        hue='Elevation',
                        data=df_box, 
                        palette=["green","red","blue","brown"], 
                        #width=0.5,
                        ax=ax)
            sns.boxplot(y="value", 
                        x="variable",whis=float('inf'),
                        hue='Elevation',
                        data=df_box, 
                        palette=["green","red","blue"], 
                        width=0.5,
                        ax=ax)
            # ax.hlines(y=-5, xmin=-1,xmax=12,linestyles='dashed',color='red')
            # ax.hlines(y=25, xmin=-1,xmax=12,linestyles='dashed',color='red')
            ax.set_xlabel(f"")
            ax.set_ylabel(f"Relative Bias [%]")
            ax.set_ylim(-100,100)
            ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
            plt.title(f"Relative bias of CPM vs SPHERA for wind at different elevation",pad=20,fontsize=20)
            plt.show()
            plt.savefig(f"figures/boxplot_bias_wind_{lvl}.png")
            # df_box=pd.DataFrame(array_boxplot,columns=[f'{m} {s}' for s in seasons for m in list_ms])
            # # df_box['col']='Heavy Prec, JJA'
            # fig=plt.figure(figsize=(14,8))
            # ax=plt.axes()
            # # print(df_box.melt())
            # sns.set(style="white")
            # sns.boxplot(y="value", 
            #             x="variable",whis=float('inf'),
            #             data=df_box.melt(), 
            #             palette=["green","red","blue"], 
            #             width=0.25,
            #             ax=ax)
            # ax.hlines(y=-5, xmin=-1,xmax=12,linestyles='dashed',color='red')
            # ax.hlines(y=25, xmin=-1,xmax=12,linestyles='dashed',color='red')
            # ax.set_xlabel(f"")
            # ax.set_ylabel(f"Relative Bias [%]")
            # ax.set_ylim(-100,100)
            # ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
            # plt.title(f"Relative bias of CPM vs SPHERA for wind at {lvl-1000}-{lvl}m",pad=20,fontsize=20)
            # plt.savefig(f"figures/boxplot_bias_wind_{lvl}.png")



    """RELATIVE BIAS PRECIPITATION"""
    # THR=np.int8(sys.argv[1])
    # nm=str(sys.argv[2])
    array_heatmap=np.zeros(shape=(len(seasons),3))
    array_boxplot=np.zeros(shape=(42976,len(seasons)*3))


    counter=0
    for i,SEAS in enumerate(seasons):
        for j,(idx,metrics) in enumerate(zip(['f','i','q'],['Frequency', 'Intensity', 'Heavy Precipitation'])):

            ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc") * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf) #xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{idx}.nc")
            ds_rea=xr.open_dataset(f"output/{SEAS}/SPHERA_{idx}.nc")  #xr.open_dataset(f"output/{SEAS}/CMCC_VHR_{idx}.nc")
            ds_rea=ds_rea.assign_coords({"lon":ds_cpm.lon.values,"lat":ds_cpm.lat.values}) * xr.where(sea_mask.sftlf == 0, np.nan, sea_mask.sftlf)

            # ds_rea.lat.values == ds_cpm.lat.values
            # ds_rea.lon.values == ds_cpm.lon.values

            rel_bias= ( ds_cpm     - ds_rea ) / ds_rea
            # rel_bias.mw.plot()
            # plt.show()

            if idx != "f":
                
                iii=np.nanargmax(xr.where(np.isfinite(rel_bias.pr),rel_bias.pr,np.nan).values)
            # np.nanargmax(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan).values.reshape(-1))
                print(f"{metrics} {SEAS} {iii}")
                print(ds_cpm.pr.values.reshape(-1)[iii],ds_rea.pr.values.reshape(-1)[iii])

            
            # np.nanargmax(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan).values)
            # np.nanargmax(xr.where(np.isfinite(rel_bias.mw),rel_bias.mw,np.nan).values.reshape(-1))

            # ds_cpm.mw.values.reshape(-1)[31055]/ds_rea.mw.values.reshape(-1)[31055]

            array_heatmap[i,j]=np.nanmean(xr.where(np.isfinite(rel_bias.pr),rel_bias.pr,np.nan)) * 100
            array_boxplot[:,counter]   = (rel_bias.pr * 100).values.reshape(-1)
            # array_boxplot[:,counter+1] = f"{metrics} {SEAS}"
            #print(f"{metrics} bias in {SEAS}: {np.nanmean(rel_bias.pr) * 100:.2f}%")
            counter+=1
            # print(f"Corr in {SEAS} for metric {idx}: {pattern_correlation(ds_cpm,ds_rea,type='centred')}")



    np.savetxt("output/array_heatmap_pr.txt",array_heatmap) 
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
    #make an heatmap with a discrete colorbar
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import matplotlib.colorbar as colorbar

    #data
    data = np.random.rand(10,10)

    #plot
    fig, ax = plt.subplots()
    cmap = plt.cm.jet
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    ax.imshow(data, interpolation='nearest', cmap=cmap)
    ax.set_title('title')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    # ax.grid(True)
    #colorbar
    #make the colorbar discrete
    bounds = np.linspace(0,1,11)
    cbar = fig.colorbar(scalarMap, ticks=bounds, boundaries=bounds, format='%.2f')

    cbar.set_label('colorbar label')

    #write the number of each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, '%.2f' % data[i, j],
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, ec='grey'))

    plt.show()
    array_heatmap=np.loadtxt(f"output/array_heatmap_pr.txt")
    df=pd.DataFrame(array_heatmap,columns=['Frequency','Intensity','Heavy Prec.'],index=seasons)
    pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
    pcm.set(xlabel="", ylabel="")
    pcm.xaxis.tick_top()
    plt.title("Biases of CPM ensmble vs SPHERA")
    # plt.show()
    plt.savefig(f"figures/heatmap_bias_prec.png")
    # plt.close()      
    if PLOT_BOXPLOTS:
        df_box=pd.DataFrame(array_boxplot,columns=[f'{m} {s}' for s in seasons for m in ['Frequency', 'Intensity', 'Heavy Precipitation']])
        # df_box['col']='Heavy Prec, JJA'
        fig=plt.figure(figsize=(14,8))
        ax=plt.axes()
        # print(df_box.melt())
        sns.set(style="white")
        sns.boxplot(y="value", 
                    x="variable",whis=float('inf'),
                    data=df_box.melt(), 
                    palette=["green","red","blue"], 
                    width=0.25,
                    ax=ax)
        ax.hlines(y=-5, xmin=-1,xmax=12,linestyles='dashed',color='red')
        ax.hlines(y=25, xmin=-1,xmax=12,linestyles='dashed',color='red')
        ax.set_xlabel(f"")
        ax.set_ylabel(f"Relative Bias [%]")
        ax.set_ylim(-100,100)

        # fig=plt.figure(figsize=(14,8))
        # ax=plt.axes()
        # # print(df_box.melt())
        # sns.set(style="darkgrid")
        # sns.boxplot(y="value", 
        #             x="variable",whis=float('inf'),
        #             data=df_box.melt(), 
        #             palette=["green","red","blue"], 
        #             width=0.25,
        #             ax=ax)
        # ax.hlines(y=-5, xmin=-0.5,xmax=11.5,linestyles='dashed',color='red')
        # ax.hlines(y=25, xmin=-0.5,xmax=11.5,linestyles='dashed',color='red')
        # ax.set_xlabel(f"")
        # ax.set_ylabel(f"Relative Bias [%]")
        # ax.set_ylim(-100,100)
        ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in ['Frequency', 'Intensity', 'Heavy Precipitation']],rotation=-25)
        plt.title("Relative bias of CPM vs SPHERA for precipitation",pad=20,fontsize=20)
        plt.savefig(f"figures/boxplot_bias_prec.png")

        # plt.show()          