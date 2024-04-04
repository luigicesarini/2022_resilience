#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
"""
1. Spatial Variability
2. Spatial correlation
3. Taylor Diagram

"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import rasterio
import argparse
import subprocess
# import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature

os.chdir("/home/lcesarini/2022_resilience/")

from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-s','--season', 
                    required=True,default='JJA',
                    choices=['SON','DJF','MAM','JJA'], 
                    help='season to analyse')
parser.add_argument('-m','--metrics', 
                    required=True,default='q',
                    choices=['f','i','q'],
                    help='metrics to analyse')

args = parser.parse_args()

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
 
eth=fix_eth()
mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/pr/MOHC_ECMWF-ERAINT_*.nc").load()
ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/pr/ICTP_ECMWF-ERAINT_*.nc").load()
hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/pr/HCLIMcom_ECMWF-ERAINT_*.nc").load()
cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
knmi_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
cmcc_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CMCC/CPM/pr/CMCC_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
kit__rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KIT/CPM/pr/KIT_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()

gripho=xr.open_dataset("/mnt/data/lcesarini/gripho_3km.nc",chunks={'time':365})
gri_per=gripho.isel(time=gripho['time.year'].isin(np.arange(2000,2010)))
gri_per=gri_per.load()

ll1=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{yr}*") for yr in np.arange(2000,2010)]
ll2=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/CMCC/pr/*{yr}*") for yr in np.arange(2000,2010)]

def get_unlist(ll:list):
    ul=[]
    for sublist in ll:
        for file in sublist:
            ul.append(file)
    return ul

sphe_rg=xr.open_mfdataset(get_unlist(ll1)).load()
vhr__rg=xr.open_mfdataset(get_unlist(ll2)).load()
sta__rg=xr.open_mfdataset([f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/pr_st_{yr}.nc" for yr in np.arange(2000,2010)]).load()

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI','CMCC','KIT','SPHERA','VHR_CMCC','STATIONS','GRIPHO']
ds_models=[eth,mohc_rg,ictp_rg,hcli_rg,cnrm_rg,knmi_rg,cmcc_rg,kit__rg,sphe_rg,vhr__rg,sta__rg,gri_per]
dict_q99_hourly={}
for NAME,DS in tqdm(zip(name_models,ds_models),total=len(name_models)):
    for S in ["JJA"]:
        dict_0={f"{NAME}_{S}":compute_quantiles_by_hour(DS,0.999,S)}
        dict_q99_hourly.update(dict_0)


# diurnal_medio=sphe_rg.groupby(sphe_rg["time.hour"]).max()
# diurnal_medio_2=diurnal_medio.mean(dim=['latitude','longitude'])
# plt.plot(diurnal_medio_2.pr)
# plt.savefig(f"figures/max_sphera.png",dpi=300,bbox_inches="tight")
# plt.close()


ax=plt.axes()
[dict_q99_hourly[list(dict_q99_hourly.keys())[j]].\
 plot(label=name_models[i],ax=ax,
      alpha= 0.5 if name_models[i] != 'CMCC' else 1,
      linewidth= 1 if name_models[i] != 'CMCC' else 2,
      linestyle='-.' if name_models[i] != 'CMCC' else '-' ) for i,j in enumerate(range(8))]
xr.concat([dict_q99_hourly[list(dict_q99_hourly.keys())[j]] for j in range(8)],'model').\
    mean(dim='model').plot(label="Ensemble",ax=ax, linewidth=4,color='red')
dict_q99_hourly[f"SPHERA_{S}"].plot(label="SPHERA",ax=ax,linestyle='-', linewidth=3,color='blue')
dict_q99_hourly[f"VHR_CMCC_{S}"].plot(label="VHR CMCC",ax=ax,linestyle='-', linewidth=3,color='magenta')
dict_q99_hourly[f"STATIONS_{S}"].plot(label="STATIONS",ax=ax,marker='*', linewidth=3,color='green')
dict_q99_hourly[f"GRIPHO_{S}"].plot(label="GRIPHO",ax=ax,marker='+', linewidth=3,color='orange')
ax.set_title("Heavy precipitation (99.9th percentile) by hour JJA")
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Precipitation (mm)")
plt.legend()
plt.savefig(f"figures/q999_hourly.png",dpi=300,bbox_inches="tight")
plt.close()

# name_models=['CMCC','KIT']
# ds_models=[cmcc_rg,kit__rg]

# for SEAS in tqdm(['SON','DJF','MAM','JJA'],total=4):
#         for NAME,DS in zip(name_models,ds_models):
#             cmcc_f,cmcc_i,cmcc_v,cmcc_q=compute_metrics(get_season(DS,season=SEAS),meters=True,quantile=0.999)
            
#             cmcc_f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{NAME}_f.nc")
#             cmcc_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{NAME}_i.nc")
#             cmcc_v.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{NAME}_v.nc")
#             cmcc_q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{NAME}_q.nc")

# cmap_f,cmap_i,cmap_q=get_palettes()
# lvl_f,lvl_i,lvl_q=get_levels()

# plot_panel_rotated(
#     figsize=(13,3.5),
#     nrow=1,ncol=3,
#     list_to_plot=[cmcc_f*100,cmcc_i,cmcc_q],
#     name_fig=f"PANEL_KIT_{SEAS}",
#     list_titles=["Frequency","Intensity","Heavy Prec."],
#     levels=[lvl_f*100,lvl_i,lvl_q],
#     suptitle=f"CMCC's metrics for {SEAS}",
#     name_metric=["[%]","[mm/hr]","[mm/hr]"],
#     SET_EXTENT=False,
#     cmap=[cmap_f,cmap_i,cmap_q]
# )      


# q_01=sphe_rg.isel(time=sphe_rg.time.dt.season.isin(['JJA']))
# q_sp=q_01.groupby(q_01['time.hour']).quantile(q=0.99)
# for i in np.arange(24):
#     print(f"{i}:00 {q_sp.isel(hour=q_sp.hour.isin([i])).mean().pr.item()}")



# q99_02=sphe_rg.isel(time=sphe_rg.time.dt.hour.isin([16])).quantile(q=0.99)