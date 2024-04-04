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
import geopandas as gpd
import matplotlib as mpl
from random import sample
# import xarray.ufuncs as xu                            
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 

from resilience.utils import *

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

REF="SPHERA"
DEBUG=True
if DEBUG:
    ADJUST = 'VALIDATION'
    NQUANT=1000
    BOXPLOTS=True
    MAPS=True
    metrics='q'
    AREA='northern_italy'
    VAR='mw'
else:
    ADJUST = args.adjust_period
    NQUANT=10000
    BOXPLOTS=args.boxplots
    MAPS=args.maps
    metrics=args.metric
    AREA=args.area
    VAR=args.var




mask=xr.open_dataset("data/mask_stations_nan_common.nc")

if __name__=="__main__":
    
    SEASONS=['JJA'] 

    for seas in SEASONS: 
        
        ori=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/JJA/{VAR}/ENSEMBLE_q_biased_SPHERA_1000_SEQUENTIAL_VALIDATION_{AREA}.nc")
        qdm=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/JJA/{VAR}/ENSEMBLE_q_QDM_SPHERA_1000_SEQUENTIAL_VALIDATION_{AREA}.nc")
        eqm=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/JJA/{VAR}/ENSEMBLE_q_EQM_SPHERA_1000_SEQUENTIAL_VALIDATION_{AREA}.nc")
        sph=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/JJA/{VAR}/SPHERA_q_1000_SEQUENTIAL_VALIDATION_{AREA}.nc")

        xr.open_dataset(f"/home/lcesarini/2022_resilience/output/mw/JJA/HCLIMcom_q_EQM_SPHERA_1000_SEQUENTIAL_VALIDATION_{AREA}.nc")
        # M='q'
        # nc_to_csv(ds_qdm,f"Ensemble_{M}_{seas}_QDM",M=M) 
        # nc_to_csv(ds_eqm,f"Ensemble_{M}_{seas}_EQM",M=M) 
        # nc_to_csv(ds_ori,f"Ensemble_{M}_{seas}_RAW",M=M) 

        # sta_q_seas=xr.load_dataset("/home/lcesarini/2022_resilience/output/MAM/stations_q_VALIDATION_10000.nc")
        # nc_to_csv(sta_q_seas,f"Stations_{M}_{seas}",M=M) 

        # ds_sta=ds_tri.sel(correction=~ds_tri.correction.str.contains("biased|EQM|QDM"))

        # ds_rea=get_triveneto(xr.load_dataset(f"output/JJA/SPHERA_{metrics}.nc"),sta_val)
        # ds_vhr=get_triveneto(xr.load_dataset(f"output/JJA/CMCC_VHR_{metrics}.nc"),sta_val)


        # gripho=xr.load_dataset(f"output/JJA/GRIPHO_ORIGINAL_{metrics}.nc")
        # gripho=xr.load_dataset(f"/mnt/data/lcesarini/gripho_3km.nc")
        # gripho_jja=gripho.isel(time=gripho['time.season'].isin(["JJA"]))
        # gripho_jja_valid=gripho_jja.isel(time=gripho_jja['time.year'].isin(np.arange(2005,2010)))

        # g_jj_q=gripho_jja.quantile(q=0.999,dim='time')
        # g_jj_q2=gripho_jja_valid.quantile(q=0.999,dim='time')
        # plot_panel_rotated(
        #         figsize=(12,8),
        #         nrow=1,ncol=2,
        #         list_to_plot=[g_jj_q.pr,g_jj_q2.pr],
        #         name_fig=f"EVERYTHING",
        #         list_titles= ["GRIPHO 20001-2016","GRIPHO 2005-2010"],
        #         levels=[lvl_q,lvl_q],
        #         suptitle=f"",
        #         # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
        #         name_metric=["Heavy Prec","Heavy Prec"],
        #         SET_EXTENT=True,
        #         cmap=[cmap_q,cmap_q],
        #         transform=rot
        # )
        # # xr.open_dataset("/mnt/data/lcesarini/gripho_3km.nc").crs
        # rot=ccrs.LambertAzimuthalEqualArea(central_latitude=52,central_longitude=10,
        #                                    false_easting=4321000,false_northing=3210000)
        # # sta_val.where(np.isfinite(sta_val.pr))

        # x,y=np.where(np.isfinite(sta_val.pr.max(dim='time')))

        # X,Y=[],[]

        # for i in range(x.shape[0]):
        #     X.append(sta_val.isel(lon=y[i],lat=x[i]).lon.item())
        #     Y.append(sta_val.isel(lon=y[i],lat=x[i]).lat.item())

        # rlon=[rot.transform_point(x_,y_,ccrs.CRS("WGS84"))[0] for x_,y_ in zip(X,Y)]
        # rlat=[rot.transform_point(x_,y_,ccrs.CRS("WGS84"))[1] for x_,y_ in zip(X,Y)]
        
        # x1=[g_jj_q2.sel(x=rx,y=ry,method='nearest').x.item() for rx,ry in zip(rlon,rlat)]
        # y1=[g_jj_q2.sel(x=rx,y=ry,method='nearest').y.item() for rx,ry in zip(rlon,rlat)]

        # # gripho.where((gripho.x.isin(x)) & (gripho.y.isin(y)), gripho.pr, np.nan )
        # # ds_gri=gripho.sel(x=x1[0],y=y1[0])

        # # for i,(rx,ry) in enumerate(zip(x1,y1)):
        # #     print(i,gripho.sel(x=rx,y=ry).pr.item())
        
        # pr=[g_jj_q2.sel(x=rx,y=ry).pr.item() for rx,ry in zip(x1,y1)]
        # ln=[g_jj_q2.sel(x=rx,y=ry).lon.item() for rx,ry in zip(x1,y1)]
        # lt=[g_jj_q2.sel(x=rx,y=ry).lat.item() for rx,ry in zip(x1,y1)]

        # df=pd.DataFrame([ln,lt,pr]).transpose().rename({0:'lon',1:'lat',2:'pr'},axis=1)
        # os.getcwd()

        # df.to_csv("csv/gripho_q_JJA.csv")
        # ds_eqm.pr.mean() == ds_qdm.pr.mean()

        # bias_ori=((ds_ori.pr - ds_sta.pr) / ds_sta.pr * 100)

        # bias_ori.values.reshape(-1)[np.isfinite(bias_ori.values.reshape(-1))

       


        # def eval_model(model,name_mdl):
            
        #     ds_sta=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/DJF/{REF}_q_1000_SEQUENTIAL_VALIDATION.nc")
            
        #     if REF=="SPHERA":
        #         ds_sta=ds_sta.rename({"longitude":"lon","latitude":"lat"})
        #         ds_sta['lon']=ds.lon.values
        #         ds_sta['lat']=ds.lat.values
        #         ds_sta=get_triveneto(ds_sta,sta_val)

        #     bc_ori=EvaluatorBiasCorrection(ds_sta.pr,model.sel(correction=model.correction.str.contains("biased")).pr)
        #     bc_eqm=EvaluatorBiasCorrection(ds_sta.pr,model.sel(correction=model.correction.str.contains("EQM")).pr)
        #     bc_qdm=EvaluatorBiasCorrection(ds_sta.pr,model.sel(correction=model.correction.str.contains("QDM")).pr)
        

        #     plot_boxplot([bc_ori.PBias()[~np.isnan(bc_ori.PBias())],
        #                   bc_eqm.PBias()[~np.isnan(bc_eqm.PBias())],
        #                   bc_qdm.PBias()[~np.isnan(bc_qdm.PBias())]
        #                 ],
        #                 names_to_concatenate=["Original","EQM","QDM"],
        #                 title=f"PBias 3 methods {metrics} {seas}",
        #                 filename=f"PBias_{name_mdl}_{metrics}_{REF}_{ADJUST}_{seas}_{NQUANT}"
        #                 )

        # list_mdl=["ETH","CNRM","KNMI","ICTP","HCLIMcom","CMCC","KIT","MOHC"] #

        # for mdl in list_mdl:
        #     eval_model(ds_tri.sel(correction=ds_tri.correction.str.contains(mdl)),mdl)
        
        # eval_model(ds_tri.sel(correction=ds_tri.correction.str.contains("ENSEMBLE")),"ENSEMBLE")

        # xx=ds_tri.sel(correction=ds_tri.correction.str.contains(mdl))
        # xx.sel(correction=xx.correction.str.contains("biased")).pr

        bc_ori=EvaluatorBiasCorrection(sph.pr,ori.pr)
        bc_eqm=EvaluatorBiasCorrection(sph.pr,eqm.pr)
        bc_qdm=EvaluatorBiasCorrection(sph.pr,qdm.pr)
        

        if BOXPLOTS:
            plot_boxplot([bc_ori.PBias()[~np.isnan(bc_ori.PBias())],
                          bc_eqm.PBias()[~np.isnan(bc_eqm.PBias())],
                          bc_qdm.PBias()[~np.isnan(bc_qdm.PBias())]
                        ],
                        names_to_concatenate=["Original","EQM","QDM"],
                        title=f"PBias 3 methods {metrics} {seas}",
                        filename=f"PBias_ensemble_{metrics}_{ADJUST}_{seas}_{NQUANT}"
                        )
        if MAPS:
            plot_panel_rotated(
                figsize=(24,8),
                nrow=1,ncol=3,
                list_to_plot=[(( bc_ori.obs - bc_ori.mod ) / bc_ori.obs * 100).isel(correction=0),
                              (( bc_ori.obs - bc_eqm.mod ) / bc_ori.obs * 100).isel(correction=0),
                              (( bc_ori.obs - bc_qdm.mod ) / bc_ori.obs * 100).isel(correction=0)
                              ],
                name_fig=f"map_PBias_ensemble_{metrics}_{ADJUST}_{seas}_{NQUANT}",
                list_titles=["Original","EQM","QDM"],
                levels=[np.arange(-50,51,10),np.arange(-50,51,10),np.arange(-50,51,10)],
                suptitle=f"Heavy Precipitation (mm)",
                # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
                name_metric=["[%]","[%]","[%]"],
                SET_EXTENT=False,
                cmap=["RdBu","RdBu","RdBu"]
            )

        # c1="red"
        # c2="blue"
        # c3="green"
        # def plot_boxplot_general(list_to_concatenate,
        #          names_to_concatenate,
        #          title,filename):
        #     if hasattr(list_to_concatenate[0],'values'):
        #         df_box=pd.DataFrame(np.concatenate([mdl.values.reshape(-1,1) for mdl in list_to_concatenate],
        #                         axis=1),
        #                     columns=[name for name in names_to_concatenate])
        #     else:
        #         df_box=pd.DataFrame(np.concatenate([mdl.reshape(-1,1) for mdl in list_to_concatenate],
        #                                 axis=1),
        #                             columns=[name for name in names_to_concatenate])
        #     # df_box['col']='Heavy Prec, JJA'
        #     fig=plt.figure(figsize=(14,8))
        #     ax=plt.axes()
        #     # print(df_box.melt())
        #     sns.set(style="darkgrid")
        #     sns.boxplot(y="value", 
        #                 x="variable",
        #                 data=df_box.melt()[~pd.isnull(df_box.melt().value)], 
        #                 palette=["green","red",'blue'], 
        #                 notch=True,
        #                 width=0.25,
        #                 flierprops={"marker": "*"},
        #                 medianprops={"color": "coral"},
        #                 ax=ax)
        #     ax.hlines(y=-5, xmin=-1,xmax=3,linestyles='dashed',color='red')
        #     ax.hlines(y=25, xmin=-1,xmax=3,linestyles='dashed',color='red')
        #     ax.set_xlabel(f"")
        #     ax.set_ylabel(f"Relative Bias [%]")
        #     ax.set_ylim(-100,100)
        #     # ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
        #     plt.title(title,pad=20,fontsize=20)
        #     plt.savefig(f"figures/{filename}.png")
        #     plt.close()


        # plot_boxplot_general([ds_ori.pr.values.reshape(-1)[np.isfinite(ds_ori.pr.values.reshape(-1))],
        #                       ds_eqm.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))],
        #                       ds_qdm.pr.values.reshape(-1)[np.isfinite(ds_qdm.pr.values.reshape(-1))]],
        #                       names_to_concatenate=["Original","EQM","QDM"],
        #                       title=f"Heavy Prec {seas}",
        #                       filename=f"geenral_boxplot_ensemble_{metrics}_{ADJUST}_{seas}_{NQUANT}")
        sns.set(style="ticks")
        fig=plt.figure(figsize=(18,10))
        ax=plt.axes()
        ax.violinplot(ds_ori.pr.values.reshape(-1)[np.isfinite(ds_ori.pr.values.reshape(-1))],positions=[0])
        for i in range(8):
            x_=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|EQM|QDM")).isel(correction=i).pr.values.reshape(-1)
            x_=x_[np.isfinite(x_)]
            ax.boxplot(x_,positions=[0])
        ax.boxplot(ds_ori.pr.values.reshape(-1)[np.isfinite(ds_ori.pr.values.reshape(-1))],   positions=[0],sym='^')
        ax.violinplot(ds_eqm.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))],positions=[1])
        ax.boxplot(ds_eqm.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))],   positions=[1],sym='^')
        ax.violinplot(ds_qdm.pr.values.reshape(-1)[np.isfinite(ds_qdm.pr.values.reshape(-1))],positions=[2])
        ax.boxplot(ds_qdm.pr.values.reshape(-1)[np.isfinite(ds_qdm.pr.values.reshape(-1))],   positions=[2],sym='^')
        ax.violinplot(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))],positions=[3])
        ax.boxplot(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))],   positions=[3],sym='^')
        ax.violinplot(ds_rea.pr.values.reshape(-1)[np.isfinite(ds_rea.pr.values.reshape(-1))],positions=[4])
        ax.boxplot(ds_rea.pr.values.reshape(-1)[np.isfinite(ds_rea.pr.values.reshape(-1))],   positions=[4],sym='^')
        ax.violinplot(ds_vhr.pr.values.reshape(-1)[np.isfinite(ds_vhr.pr.values.reshape(-1))],positions=[5])
        ax.boxplot(ds_vhr.pr.values.reshape(-1)[np.isfinite(ds_vhr.pr.values.reshape(-1))],   positions=[5],sym='^')
        ax.violinplot(np.array(pr)[np.isfinite(np.array(pr))],positions=[6])
        ax.boxplot(np.array(pr)[np.isfinite(np.array(pr))],   positions=[6],sym='^')
        
        ax.violinplot(ds_eqm.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))],positions=[1])
        ax.boxplot(ds_eqm2.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))],   positions=[1],sym='^')
        ds_eqm2.pr.to_dataframe()
        ax.hlines(y=ds_sta.pr.median().item(),xmin=-0.5,xmax=6.5,linestyles='dashed',color='red')
        ax.set_xticklabels(["Original","EQM","QDM","OBSERVATIONS","SPHERA","VHR","GRIPHO"],fontdict={'fontsize': 21})
        ax.set_ylabel(f"Precipitation [mm/h]",fontdict={'fontsize': 21})
        ax.set_title(f"Distribution of heavy prec (99.9th) in {seas}",fontdict={'fontsize': 26})
        plt.show()


        plt.savefig(f"boxplot_ori_{metrics}_{ADJUST}_{seas}_{NQUANT}.png")
        plt.close()


        """
        QQ-Plot
        """
        fig=plt.figure(figsize=(8,8))
        ax=plt.axes()
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))]),
                np.sort(ds_ori.pr.values.reshape(-1)[np.isfinite(ds_ori.pr.values.reshape(-1))])[1:],   
                "*",label='BIASED')
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))]),
                np.sort(ds_eqm.pr.values.reshape(-1)[np.isfinite(ds_eqm.pr.values.reshape(-1))]),   
                "^",label='EQM')
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))]),
                np.sort(ds_qdm.pr.values.reshape(-1)[np.isfinite(ds_qdm.pr.values.reshape(-1))]),   
                "+",label='QDM')
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))]),
                np.sort(ds_rea.pr.values.reshape(-1)[np.isfinite(ds_rea.pr.values.reshape(-1))])[1:],   
                ".",label='SPHERA')
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))]),
                np.sort(ds_vhr.pr.values.reshape(-1)[np.isfinite(ds_vhr.pr.values.reshape(-1))])[1:],   
                ">",label='VHR')
        ax.plot(np.sort(ds_sta.pr.values.reshape(-1)[np.isfinite(ds_sta.pr.values.reshape(-1))])[5:168],
                np.sort(np.array(pr)[np.isfinite(np.array(pr))]),   
                ">",label='GRIPHO')
        ax.axline([5,5],[25,25],c='salmon')
        ax.legend()
        ax.set_xlabel(f"Observations",fontdict={'fontsize': 21})
        ax.set_ylabel(f"Modelled",fontdict={'fontsize': 21})
        ax.set_title(f"QQ-plot heavy prec (99.9th) in {seas}",fontdict={'fontsize': 26})
        plt.savefig(f"qq_{metrics}_{ADJUST}_{seas}_{NQUANT}.png")
        plt.close()

# find  -type d -exec chmod 755 {} \;
# find -type f -exec chmod 644 {} \; 