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
                    choices=["mean","thr","q"]  
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
DEBUG=False
metrics = ['mean', 'q', 'thr']
if DEBUG:
    ADJUST = 'VALIDATION'
    NQUANT=1000
    BOXPLOTS=True
    MAPS=True
    metrics='q'
else:
    ADJUST = args.adjust_period
    NQUANT=1000
    BOXPLOTS=args.boxplots
    MAPS=args.maps
    metrics=args.metric

mask=xr.open_dataset("data/mask_stations_nan_common.nc")
sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")

if __name__=="__main__":
    
    SEASONS=['SON','DJF','MAM', 'JJA'] 

    for seas in SEASONS: 
        
        # f"/home/lcesarini/2022_resilience/output/{seas}/*{metrics}_**{ADJUST}**{NQUANT}*"
        list_output_mdl=glob(f"/home/lcesarini/2022_resilience/output/mw/{seas}/*{metrics}_**{REF}**{NQUANT}**{ADJUST}*")
        list_output_mdl=list_output_mdl+glob(f"/home/lcesarini/2022_resilience/output/mw/{seas}/*{REF}**{metrics}_**{NQUANT}**{ADJUST}*")
        list_mdl=[os.path.basename(file).split("_")[0] for file in list_output_mdl]

        ds=[xr.open_dataset(file) for file in list_output_mdl]

        lon_min=ds[1]['lon'].min().item()-0.01
        lon_max=ds[1]['lon'].max().item()+0.01
        lat_min=ds[1]['lat'].min().item()-0.01
        lat_max=ds[1]['lat'].max().item()+0.01

        ds=[crop_to_extent(s,lon_min,lon_max,lat_min,lat_max) if s.lon.values.shape[0] > 100 else s for s in ds  ]

        sea_mask=crop_to_extent(sea_mask,lon_min,lon_max,lat_min,lat_max) 

        # len(ds)
        # for i in np.arange(0,22):
        #     print(ds[i].lat[0:1].values.item())
        #     print(ds[i].lon[0:1].values.item())

        # xr.concat(ds[-3:-1],"new_dim")

        if (not (ds[-1]['lon'].values==ds[1]['lon'].values).all().item() ) | (not (ds[-1]['lat'].values==ds[1]['lat'].values).all().item()):
            ds[-1]['lon']=ds[0].lon.values
            ds[-1]['lat']=ds[0].lat.values

        ds=xr.concat(ds,["_".join(os.path.basename(file).split("_")[:3]) if "stations" not in file else\
                        "_".join(os.path.basename(file).split("_")[:2]) for file in list_output_mdl]).\
                    rename({"concat_dim":"correction"})

        # ds_tri=crop_to_extent(ds[0],lon_min,lon_max,lat_min,lat_max)
        ds_tri=ds
        
        ds_ori=ds_tri.sel(correction=~ds_tri.correction.str.contains(f"{REF}|EQM|QDM")).mean(dim='correction')
        ds_eqm=ds_tri.sel(correction=~ds_tri.correction.str.contains(f"{REF}|biased|QDM")).mean(dim='correction')
        # ds_eqm2=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|biased|QDM"))
        ds_qdm=ds_tri.sel(correction=~ds_tri.correction.str.contains(f"{REF}|EQM|biased")).mean(dim='correction')


        ds_ref=ds_tri.sel(correction=~ds_tri.correction.str.contains("biased|EQM|QDM"))

        # (ds_ref.mw * sea_mask.sftlf).plot()
        # plt.show()

        bc_ori=EvaluatorBiasCorrection(ds_ref.mw * sea_mask.sftlf,ds_ori.mw * sea_mask.sftlf)
        if metrics == "thr":
            bc_eqm=EvaluatorBiasCorrection(ds_ref.mw * sea_mask.sftlf,ds_eqm.mw / 100 * sea_mask.sftlf)
            bc_qdm=EvaluatorBiasCorrection(ds_ref.mw * sea_mask.sftlf,ds_qdm.mw / 100 * sea_mask.sftlf)
        else:
            bc_eqm=EvaluatorBiasCorrection(ds_ref.mw * sea_mask.sftlf,ds_eqm.mw * sea_mask.sftlf)
            bc_qdm=EvaluatorBiasCorrection(ds_ref.mw * sea_mask.sftlf,ds_qdm.mw * sea_mask.sftlf)
        




        if BOXPLOTS:
            plot_boxplot([bc_ori.PBias()[~np.isnan(bc_ori.PBias())],
                          bc_eqm.PBias()[~np.isnan(bc_eqm.PBias())],
                          bc_qdm.PBias()[~np.isnan(bc_qdm.PBias())]
                        ],
                        names_to_concatenate=["Original","EQM","QDM"],
                        title=f"PBias 3 methods {metrics} {seas}",
                        filename=f"PBias_ensemble_{metrics}_{ADJUST}_{seas}_{NQUANT}_mw",
                        SAVE=True
                        )
        if MAPS:
            plot_panel_rotated(
                figsize=(24,8),
                nrow=1,ncol=3,
                list_to_plot=[(( bc_ori.obs - bc_ori.mod ) / bc_ori.obs * 100).isel(correction=0),
                              (( bc_ori.obs - bc_eqm.mod ) / bc_ori.obs * 100).isel(correction=0),
                              (( bc_ori.obs - bc_qdm.mod ) / bc_ori.obs * 100).isel(correction=0)
                              ],
                name_fig=f"map_PBias_ensemble_{metrics}_{ADJUST}_{seas}_{NQUANT}_mw",
                list_titles=["Original","EQM","QDM"],
                levels=[np.arange(-50,51,10),np.arange(-50,51,10),np.arange(-50,51,10)],
                suptitle=f"Heavy Precipitation (mm)",
                # name_metric=["[mm/h]","[mm/h]","[mm/h]"],
                name_metric=["[%]","[%]","[%]"],
                SET_EXTENT=False,
                cmap=["PuOr","PuOr","PuOr"],
                SAVE=True
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
        # sns.set(style="ticks")
        # fig=plt.figure(figsize=(18,10))
        # ax=plt.axes()
        # ax.violinplot(ds_ori.mw.values.reshape(-1)[np.isfinite(ds_ori.mw.values.reshape(-1))],positions=[0])
        # for i in range(4):
        #     x_=ds_tri.sel(correction=~ds_tri.correction.str.contains("stations|EQM|QDM")).isel(correction=i).mw.values.reshape(-1)
        #     x_=x_[np.isfinite(x_)]
        #     ax.boxplot(x_,positions=[0])
        # ax.boxplot(ds_ori.mw.values.reshape(-1)[np.isfinite(ds_ori.mw.values.reshape(-1))],   positions=[0],sym='^')
        # ax.violinplot(ds_eqm.mw.values.reshape(-1)[np.isfinite(ds_eqm.mw.values.reshape(-1))],positions=[1])
        # ax.boxplot(ds_eqm.mw.values.reshape(-1)[np.isfinite(ds_eqm.mw.values.reshape(-1))],   positions=[1],sym='^')
        # ax.violinplot(ds_qdm.mw.values.reshape(-1)[np.isfinite(ds_qdm.mw.values.reshape(-1))],positions=[2])
        # ax.boxplot(ds_qdm.mw.values.reshape(-1)[np.isfinite(ds_qdm.mw.values.reshape(-1))],   positions=[2],sym='^')
        # ax.violinplot(ds_ref.mw.values.reshape(-1)[np.isfinite(ds_ref.mw.values.reshape(-1))],positions=[3])
        # ax.boxplot(ds_ref.mw.values.reshape(-1)[np.isfinite(ds_ref.mw.values.reshape(-1))],   positions=[3],sym='^')
        
        # ax.hlines(y=ds_ref.mw.median().item(),xmin=-0.5,xmax=6.5,linestyles='dashed',color='red')
        # ax.set_xticklabels(["Original","EQM","QDM","OBSERVATIONS"],fontdict={'fontsize': 21})
        # ax.set_ylabel(f"Wind [m/s]",fontdict={'fontsize': 21})
        # ax.set_title(f"Distribution of extreme wind (99th) in {seas}",fontdict={'fontsize': 26})
        # # plt.show()


        # plt.savefig(f"boxplot_ori_{metrics}_{ADJUST}_{seas}_{NQUANT}_mw.png")
        # plt.close()


# find  -type d -exec chmod 755 {} \;
# find -type f -exec chmod 644 {} \; 