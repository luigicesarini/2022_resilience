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
from scipy.signal import correlate2d
from cartopy import feature as cfeature

os.chdir("/home/lcesarini/2022_resilience/")

from resilience.utils import *

# paths=[
#     "/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr",
#     "/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr",
#     "/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr",
#     "/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Rcp85/FZJ-IDL-WRF381DA/CPM/pr",
#     "/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr",
# ]

# for path in paths:
#     ll_files=glob(f"{path}/*.nc")
#     for file in tqdm(ll_files):
#         x=xr.open_dataset(file).isel(time=slice(12,35)).load()
#         x.pr.quantile(q=0.8,dim='time').plot()
#         plt.title(f"{os.path.basename(file)}")
#         plt.show()
# x1=xr.open_dataset(f'{paths[1]}/pr_ALP-3_SMHI-EC-EARTH_historical_r12_FZJ-IBG3-WRF381CA_v1_1hr_200501010030-200512312330.nc').load()
# x2=xr.open_dataset(f'{paths[2]}/pr_ALP-3_SMHI-EC-EARTH_historical_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_200501010030-200512312330.nc').load()
# x3=xr.open_mfdataset(f'{paths[2]}/pr_ALP-3_SMHI-EC-EARTH_historical_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_*.nc').load()

# x1.pr.quantile(q=0.8,dim='time').plot()
# plt.suptitle(f"{x1.attrs['title']}")
# plt.title(f"{pd.to_datetime(x1.time.values[0])}")
# plt.show()

# x2.pr.quantile(q=0.8,dim='time').plot()
# plt.suptitle(f"{x2.attrs['title']}")
# plt.title(f"{pd.to_datetime(x2.time.values[0])}")
# plt.show()

# x3.pr.quantile(q=0.8,dim='time').plot()
# plt.suptitle(f"{x3.attrs['title']}")
# plt.title(f"{pd.to_datetime(x3.time.values[0])}")
# plt.show()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-s','--season', 
                    required=True,default="JJA",
                    choices=["SON","DJF","MAM","JJA"], 
                    help='season to analyse')
parser.add_argument('-m','--metrics', 
                    required=True,default="q",
                    choices=["f","i","q"],
                    help='metrics to analyse')

args = parser.parse_args()


PATH_SPHERA_OUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr"
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
PATH_OUTPUT=f"/home/lcesarini/2022_resilience/output/"

shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

shp_ita=gpd.read_file("/home/lcesarini/2022_resilience/gadm36_ITA.gpkg", layer="gadm36_ITA_0")
mask=xr.open_dataset("data/mask_stations_nan_common.nc")

SEAS=args.season
M=args.metrics
# M='q'
# SEAS='JJA'

na_stat=xr.load_dataset(f"{PATH_OUTPUT}/JJA/STATIONS_q.nc").isel(quantile=0)

ens=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_{M}.nc")
rea=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/SPHERA_{M}.nc")
vhr=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/CMCC_VHR_{M}.nc")
if M=='q':
    sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc").isel(quantile=0)
else:
    sta=xr.load_dataset(f"{PATH_OUTPUT}/{SEAS}/STATIONS_{M}.nc")



# pattern_correlation(ens.isel(lon=range(6),lat=range(6)),
#                     rea.isel(lon=range(6),lat=range(6)),type="uncentred")
xmin,xmax=get_range(na_stat.lon)
ymin,ymax=get_range(na_stat.lat)

ens_sta = clip_ds(ens * mask.mask,xmin,xmax,ymin,ymax)
rea_sta = clip_ds(rea * mask.mask,xmin,xmax,ymin,ymax)
vhr_sta = clip_ds(vhr * mask.mask,xmin,xmax,ymin,ymax)
# ens_sta=xr.where(np.isfinite(na_stat.pr),ens.isel(lon=ens.lon.isin(sta.lon),lat=ens.lat.isin(sta.lat)),np.nan)
# ens_sta=xr.where(np.isfinite(na_stat.pr),ens.isel(lon=ens.lon.isin(sta.lon),lat=ens.lat.isin(sta.lat)),np.nan)
sta=xr.where(np.isfinite(na_stat.pr),sta,np.nan)


np.sum(~np.isnan(sta.pr.values.reshape(-1)))
np.sum(~np.isnan(ens_sta.pr.values.reshape(-1)))

ens_rea_corr=pattern_correlation(ens_sta,rea_sta,type="centred")
ens_vhr_corr=pattern_correlation(ens_sta,vhr_sta,type="centred")
ens_sta_corr=pattern_correlation(ens_sta,sta,type="centred")
vhr_sta_corr=pattern_correlation(vhr_sta,sta,type="centred")
rea_sta_corr=pattern_correlation(rea_sta,sta,type="centred")

print(f"""
{SEAS} metrics: {M}
Correlation ENSEMBLE-SPHERA:{ens_rea_corr:.4f}
Correlation ENSEMBLE-STATIONS:{ens_sta_corr:.4f}
Correlation ENSEMBLE-VHR CMCC:{ens_vhr_corr:.4f}
Correlation SPHERA-STATIONS:{rea_sta_corr:.4f}
Correlation VHR CMCC-STATIONS:{vhr_sta_corr:.4f}
"""
)
      
# np.savetxt("ensemble.csv",ens_sta.pr.values.reshape(-1)[np.isfinite(ens_sta.pr.values.reshape(-1))], delimiter=",")

# np.savetxt("reanalysis.csv",rea_sta.pr.values.reshape(-1)[np.isfinite(rea_sta.pr.values.reshape(-1))], delimiter=",")


# pattern_correlation(ens,vhr)
# pattern_correlation(vhr,rea)

# pattern_correlation(ens.isel(lon=ens.lon.isin(sta.lon),
#                              lat=ens.lat.isin(sta.lat)),
#                              sta.isel(quantile=0))

# pattern_correlation(rea.isel(lon=ens.lon.isin(sta.lon),
#                              lat=ens.lat.isin(sta.lat)),
#                              sta.isel(quantile=0))

# pattern_correlation(vhr.isel(lon=ens.lon.isin(sta.lon),
#                              lat=ens.lat.isin(sta.lat)),
#                              sta.isel(quantile=0))

# ens.isel(lon=ens.lon.isin(sta.lon),lat=ens.lat.isin(sta.lat)).pr
# np.nansum(sta.isel(quantile=0).pr.values)



# np.nansum(ens_sta.pr)

# np.nansum(ens_sta.pr * sta.isel(quantile=0).pr) / (np.nansum(np.square(ens_sta.pr)))
# def nc_to_csv(ds:xr.Dataset,name:str):
#     df=ds[list(ds.data_vars)[0]].to_pandas()
#     df.reset_index(names=['lat']).\
#     melt(id_vars="lat",value_vars=df.columns,var_name="lon",value_name=M).\
#         to_csv(f"{name}.csv",index=False)

# nc_to_csv(ens_sta,f"ens_sta_{SEAS}_{M}")
# nc_to_csv(rea_sta,f"rea_sta_{SEAS}_{M}")
# nc_to_csv(vhr_sta,f"vhr_sta_{SEAS}_{M}")
# nc_to_csv(sta,f"sta_{SEAS}_{M}")


sv_ens_rea=(np.nanstd(ens_sta.pr)/np.nanmean(ens_sta.pr)) / (np.nanstd(rea_sta.pr)/np.nanmean(rea_sta.pr))
sv_ens_sta=(np.nanstd(ens_sta.pr)/np.nanmean(ens_sta.pr)) / (np.nanstd(sta.pr)/np.nanmean(sta.pr))
sv_ens_vhr=(np.nanstd(ens_sta.pr)/np.nanmean(ens_sta.pr)) / (np.nanstd(vhr_sta.pr)/np.nanmean(vhr_sta.pr))
sv_rea_sta=(np.nanstd(rea_sta.pr)/np.nanmean(rea_sta.pr)) / (np.nanstd(sta.pr)/np.nanmean(sta.pr))
sv_vhr_sta=(np.nanstd(vhr_sta.pr)/np.nanmean(vhr_sta.pr)) / (np.nanstd(sta.pr)/np.nanmean(sta.pr))

print(f"""
{SEAS} metrics: {M}
Spatial Variability ENSEMBLE-SPHERA:{sv_ens_rea:.4f}
Spatial Variability ENSEMBLE-STATIONS:{sv_ens_sta:.4f}
Spatial Variability ENSEMBLE-VHR CMCC:{sv_ens_vhr:.4f}
Spatial Variability SPHERA-STATIONS:{sv_rea_sta:.4f}
Spatial Variability VHR CMCC-STATIONS:{sv_vhr_sta:.4f}
"""
)
      
"""
print spatial correlation and spatial variability to text
"""
PRINT=False
if PRINT:
    dict_to_print={
        'season':SEAS,
        'metric':M,
        'EV':'pr',
        'SC ENS-SPH':ens_rea_corr,
        'SC ENS-STA':ens_sta_corr,
        'SC ENS-VHR':ens_vhr_corr,
        'SC SPH-STA':rea_sta_corr,
        'SC VHR-STA':vhr_sta_corr,
        'SV ENS-SPH':sv_ens_rea,
        'SV ENS-STA':sv_ens_sta,
        'SV ENS-VHR':sv_ens_vhr,
        'SV SPH-STA':sv_rea_sta,
        'SV VHR-STA':sv_vhr_sta
    }

    if not os.path.isfile(f'output/corr/df_corr.csv'):
        pd.DataFrame(dict_to_print,index=[0]).to_csv(f'output/corr/df_corr.csv',index=False)

    else:
        pd.DataFrame(dict_to_print,index=[0]).to_csv(f'output/corr/df_corr.csv',mode="a",index=False,header=False)

# (np.nanstd(ens_sta.pr)) / (np.nanstd(rea_sta.pr))
# (np.nanstd(ens_sta.pr)) / (np.nanstd(sta.pr))

"""
To plot Taylor I need:
- Values of spatial correlation and values of Spatial variability 

|   What            |  Sp. Corr         |  Sp. Variability  |
|-------------------|-------------------|-------------------|
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
"""


# geoms = shp_ita.geometry.values # list of shapely geometries
# geometry = geoms[0] # shapely geometry
# # transform to GeJSON format
# geoms = [mapping(geoms[0])]
# # extract the raster values values within the polygon 
# with rasterio.open(f"{PATH_OUTPUT}/{SEAS}/ENSEMBLE_q.nc") as src:
#      out_image, out_transform = mask(src, geoms, crop=False)



# ds_mask = xr.Dataset(
#     data_vars=dict(
#         corr=(["lat", "lon"],np.flip(out_image[0,:,:],axis=0)),
#     ),
#     coords=dict(
#         lon=ens_q.lon,
#         lat=ens_q.lat,
#     ),
#     attrs=ens_q.attrs,
# )

# ds_mask.to_netcdf("output/mask_study_area.nc")
# fig,ax=plt.subplots(1,2,figsize=(10,8))
# (ens_q.pr * ds_mask.corr).plot.pcolormesh(ax=ax[0])
# (rea_q.pr * ds_mask.corr).plot.pcolormesh(ax=ax[1])
# plt.savefig('test.png')
# plt.close()
# corr2D=correlate2d(ens_q.pr,ens_q.pr)

# np.isnan((ens_q.pr * ds_mask.corr).values.reshape(-1)).sum()

# x_ens,y_ens=np.where(np.isnan((ens_q.pr * ds_mask.corr)))
# x_rea,y_rea=np.where(np.isnan((rea_q.pr * ds_mask.corr)))

# xy_ens=np.concatenate([x_ens.reshape(-1,1),y_ens.reshape(-1,1)],axis=1)
# xy_rea=np.concatenate([x_rea.reshape(-1,1),y_rea.reshape(-1,1)],axis=1)

# for i in range(xy_ens.shape[0]):
#     if xy_ens[i,:] not in xy_rea:
#         print(i,"Ens not in Rea")   
#     elif ~np.any((xy_ens[:,0] == xy_rea[i,0]) & (xy_ens[:,1] == xy_rea[i,1])):
#         idx=np.where((xy_ens[:,0] == xy_rea[i,0]) & (xy_ens[:,1] == xy_rea[i,1]))
#         print(idx,"Rea not in Ens")

# np.where(xy_rea[i,0] in xy_ens[:,])

# xy_ens[21468]

# [e==r for e,r in zip(xy_ens,xy_rea)]
# np.corrcoef(ens_q.pr.values.reshape(-1),rea_q.pr.values.reshape(-1))

# corr_ens_sta=compute_coeff(ens_q.pr,ens_q.pr)

# ens_q_tri=ens_q.sel(lon=ens_q.lon.isin(sta_q.lon),lat=ens_q.lat.isin(sta_q.lat))
# rea_q_tri=rea_q.sel(lon=rea_q.lon.isin(sta_q.lon),lat=rea_q.lat.isin(sta_q.lat))



# mask_sta=np.where(np.isnan(out_image),np.nan,1)

# ens_masked = ens_q.pr * np.flip(mask_sta[0,:,:],axis=0)
# rea_masked = rea_q.pr * np.flip(mask_sta[0,:,:],axis=0)

# np.corrcoef(ens_masked,rea_masked).shape


# mask_sta[0,:,:][lat_na,lon_na]
# mask_rea=np.where(np.isnan(ens_masked.values) != np.isnan(rea_masked.values),np.nan,1)

# ens_masked = ens_q.pr * np.flip(mask_rea[:,:],axis=0)
# rea_masked = rea_q.pr * np.flip(mask_rea[:,:],axis=0)

# clean_na(ens_masked).shape
# clean_na(rea_masked).shape

# lat_na,lon_na=np.argwhere(np.isnan(ens_masked.values) != np.isnan(rea_masked.values))[:,0],np.argwhere(np.isnan(ens_masked.values) != np.isnan(rea_masked.values))[:,1]

# rea_masked.isel(lon=267,lat=89)

# fig,axs=plt.subplots(1,2,
#                      figsize=(12,4),
#                      subplot_kw={"projection":ccrs.PlateCarree()})
# ens_masked.plot.pcolormesh(ax=axs[0],add_colorbar=False)
# rea_masked.plot.pcolormesh(ax=axs[1],add_colorbar=False)
# # sta_q.pr.isel(quantile=0).plot.pcolormesh(ax=axs[2],add_colorbar=False)
# # shp_triveneto.boundary.plot(ax=axs[0],ec='red')
# # shp_triveneto.boundary.plot(ax=axs[1],ec='red')
# # shp_triveneto.boundary.plot(ax=axs[2],ec='red')
# shp_ita.boundary.plot(ax=axs[0],ec='red')
# shp_ita.boundary.plot(ax=axs[1],ec='red')
# # shp_ita.boundary.plot(ax=axs[2],ec='red')
# gl=axs[0].gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
# gl=axs[1].gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
# # gl=axs[2].gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
# # axs[2].coastlines()
# axs[0].set_extent([6.35,14,43.10,47.6])
# axs[1].set_extent([6.35,14,43.10,47.6])
# plt.savefig("figures/check_na.png")
# plt.close()



# ds_tc = xr.Dataset(
#     data_vars=dict(
#         corr=(["lat", "lon"],corr_ens_sta),
#     ),
#     coords=dict(
#         lon=ens_q.lon,
#         lat=ens_q.lat,
#     ),
#     attrs=ens_q.attrs,
# )

# f,ax=plt.subplots(1,1,figsize=(8,8),subplot_kw={"projection":ccrs.PlateCarree()})
# ds_tc.corr.plot.pcolormesh(ax=ax,cmap="Spectral",add_colorbar=True,
#             levels=np.arange(0.825,0.999999999,(0.999999999-0.825)/7))
# # shp_triveneto.boundary.plot(ax=axs[:2],ec='red')
# ax.coastlines()
# ax.set_extent([10.2,13.2,44.5,47.4])
# ax.set_title(f"Correlation according to Pearson coefficient")
# plt.savefig("figures/spat_corr_ens_rea.png")
# plt.close()


# ens_q_masked=(ens_q_tri.pr * mask_sta[0,:,:])
# rea_q_masked=(rea_q_tri.pr * mask_sta[0,:,:])

# np.all(np.isnan(ens_q_masked.values.reshape(-1)) == np.isnan(rea_q_masked.values.reshape(-1))).item()

# def clean_na(arr):
#     return arr.values.reshape(-1)[~np.isnan(arr.values.reshape(-1))]

# clean_na(ens_masked).shape
# clean_na(rea_masked).shape

# #MODEL-REANALYSIS
# np.corrcoef(ens_q_masked.values.reshape(-1)[~np.isnan(ens_q_masked.values.reshape(-1))],
#             rea_q_masked.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))])
# #MODEL-STATION
# np.corrcoef(ens_q_masked.values.reshape(-1)[~np.isnan(ens_q_masked.values.reshape(-1))],
#             sta_q.isel(quantile=0).pr.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))])

# ratio_ens_sta = ens_q_masked.values.reshape(-1)[~np.isnan(ens_q_masked.values.reshape(-1))] / sta_q.isel(quantile=0).pr.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))]

# np.std(ratio_ens_sta) / np.mean(ratio_ens_sta)

# #REANALYSIS-STATION
# np.corrcoef(rea_q_masked.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))],
#             sta_q.isel(quantile=0).pr.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))])

# ratio_rea_sta = rea_q_masked.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))] / sta_q.isel(quantile=0).pr.values.reshape(-1)[~np.isnan(rea_q_masked.values.reshape(-1))]

# np.std(ratio_rea_sta) / np.mean(ratio_rea_sta)


# # a=xr.where(ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.1,
# #            ds["tp" if "tp" in list(ds.data_vars) else "pr"],
# #            np.nan)

# # from scipy.stats import skew

# # skew_a=skew(a,axis=0,nan_policy="omit")

# # skew_a.shape

# # ds_skew = xr.Dataset(
# #     data_vars=dict(
# #         skew=(["lat", "lon"],skew_a),
# #     ),
# #     coords=dict(
# #         lon=ens_q.lon,
# #         lat=ens_q.lat,
# #     ),
# #     # attrs=ens_q.attrs,
# # )

# # ax=plt.axes(projection=ccrs.PlateCarree())
# # ds_skew.skew.plot.pcolormesh(ax=ax)
# # ax.coastlines()
# # plt.savefig("figures/skew.png")

# #write a PatternCorrelation class

