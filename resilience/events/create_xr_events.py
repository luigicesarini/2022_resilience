#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
"""
Create xarray from list of defined events
"""

import os
import sys
sys.path.append("/home/lcesarini/2022_resilience")
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from resilience.utils import get_unlist,get_palettes,plot_bin_hist,read_file_events

os.chdir("/home/lcesarini/2022_resilience")
sftlf=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KNMI/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
SEAS="JJA"

model=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2001)])
model_slice=model.isel(time=1).load()

# xr.where(sftlf.sftlf > 50,1,0)

idx_on_land=np.argwhere(xr.where(sftlf.sftlf > 50,1,0).values.reshape(-1)==1)

l_pr_sph,a_pr_sph,m_pr_sph=read_file_events(EV="pr",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")
l_mw_sph,a_mw_sph,m_mw_sph=read_file_events(EV="mw",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")
l_cb_sph,a_cb_sph,m_cb_sph=read_file_events(EV="combined",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")

l_pr_cpm,a_pr_cpm,m_pr_cpm=read_file_events(EV="pr",MDL='KNMI',THR=99,INDICES=idx_on_land,SEAS="JJA")
l_mw_cpm,a_mw_cpm,m_mw_cpm=read_file_events(EV="mw",MDL='KNMI',THR=99,INDICES=idx_on_land,SEAS="JJA")
l_cb_cpm,a_cb_cpm,m_cb_cpm=read_file_events(EV="combined",MDL='KNMI',THR=99,INDICES=idx_on_land,SEAS="JJA")


"""
VIOLIN PLOT AND 2D HISTOGRAM
"""
# df_1=pd.DataFrame({"Precipitation":get_unlist(l_pr_sph)}).melt()
# df_2=pd.DataFrame({"Wind":get_unlist(l_mw_sph)}).melt()
# df_3=pd.DataFrame({"Combined":get_unlist(l_cb_sph)}).melt()
# df_concat=pd.concat([df_1,df_2,df_3])

# import seaborn as sns
# ax= sns.violinplot(df_concat.rename(columns={"value":"Duration"}),x="variable",y="Duration",hue="variable");
# ax.set_title("Duration of Events")
# ax.set_ylabel("Duration (hours)")
# ax.set_xlabel("")
# ax.set_ylim(0,5)
# plt.show()


"""
MAPS
"""

# longitude=[]
# latitude=[]

# for i in tqdm(range(158)):
#     for j in range(272):
#         lon,lat=model_slice.isel(lat=i,lon=j).lon.item(),model_slice.isel(lat=i,lon=j).lat.item()

#         longitude.append(lon)
#         latitude.append(lat)

# ds_mw = xr.Dataset(
#     data_vars=dict(
#         n_event=(["lat", "lon"],np.array([len(x) for x in l_mw_sph]).reshape(158,272)),
#         d_event=(["lat", "lon"],np.array([np.max(x) for x in l_mw_sph]).reshape(158,272)),
#         avg_int=(["lat", "lon"],np.array([np.median(x) for x in a_mw_sph]).reshape(158,272)),
#         max_int=(["lat", "lon"],np.array([np.median(x) for x in m_mw_sph]).reshape(158,272)),
        
#     ),
#     coords=dict(
#         lon=np.unique(longitude),
#         lat=np.unique(latitude)
#     ),
#     attrs=model_slice.attrs,
# )


# cmap_f,cmap_i,cmap_q=get_palettes()

# fig,ax = plt.subplots(1,2, figsize=(16,6),subplot_kw={"projection":ccrs.PlateCarree()})

# ds_mw.n_event.plot(cmap=cmap_q,
#                    #levels=np.arange(10,200,25),#np.nanquantile(ds_mw.n_event,q=np.arange(0.1,0.999,0.11125)),
#                    ax=ax[0],add_colorbar=True)
# ds_mw.d_event.plot(cmap="RdBu",
#                 #levels=np.nanquantile(ds_mw.avg_int,q=np.arange(0.1,0.999,0.11125)),
#                 ax=ax[1],add_colorbar=True)

# [ax[_].coastlines() for _ in range(2)]
# [ax[_].add_feature(cfeature.BORDERS) for _ in range(2)]
# plt.suptitle("Wind");
# plt.show()


if __name__=="__main__":

    bins_pr_avg_inte=np.load("/home/lcesarini/bins_pr_avg_inte.npy")
    bins_pr_max_inte=np.load("/home/lcesarini/bins_pr_max_inte.npy")
    bins_pr_duration=np.load("/home/lcesarini/bins_pr_duration.npy")

    bins_mw_avg_inte=np.load("/home/lcesarini/bins_mw_avg_inte.npy")
    bins_mw_max_inte=np.load("/home/lcesarini/bins_mw_max_inte.npy")
    bins_mw_duration=np.load("/home/lcesarini/bins_mw_duration.npy")

    for MDL in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom","SPHERA"]: #"KNMI","CMCC","CNRM","KIT",

        # l_pr_sph,a_pr_sph,m_pr_sph=read_file_events(EV="pr",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")
        # l_mw_sph,a_mw_sph,m_mw_sph=read_file_events(EV="mw",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")
        # l_cb_sph,a_cb_sph,m_cb_sph=read_file_events(EV="combined",MDL='SPHERA',THR=99,INDICES=idx_on_land,SEAS="JJA")

        l_pr_cpm,a_pr_cpm,m_pr_cpm=read_file_events(EV="pr",MDL=MDL,THR=99,INDICES=idx_on_land,SEAS="JJA")
        l_mw_cpm,a_mw_cpm,m_mw_cpm=read_file_events(EV="mw",MDL=MDL,THR=99,INDICES=idx_on_land,SEAS="JJA")
        # l_cb_cpm,a_cb_cpm,m_cb_cpm=read_file_events(EV="combined",MDL=MDL,THR=99,INDICES=idx_on_land,SEAS="JJA")


        plot_bin_hist(l_pr_cpm,m_pr_cpm,a_pr_cpm,bins_pr_max_inte,bins_pr_avg_inte,bins_pr_duration,MDL,"Precipitation")
        plot_bin_hist(l_mw_cpm,m_mw_cpm,a_mw_cpm,bins_mw_max_inte,bins_mw_avg_inte,bins_mw_duration,MDL,"Wind")
        # plot_bin_hist(l_mw_sph,m_mw_sph,a_mw_sph,"SPHERA","Wind")
        # plot_bin_hist(l_pr_cpm,m_pr_cpm,a_pr_cpm,"KNMI","Precipitation")




# for MDL in ["CMCC","CNRM","KIT","KNMI","MOHC","ETH","ICTP","HCLIMcom","STATIONS","SPHERA"]:

#     #LOAD the lists
#     with open(f'/mnt/data/lcesarini/{MDL}_len_events.pkl', 'rb') as file:
#         len_per_above_threshold=pickle.load(file)

#     with open(f'/mnt/data/lcesarini/{MDL}_mean_events.pkl', 'rb') as file:
#         mean_per_periods=pickle.load(file)

#     with open(f'/mnt/data/lcesarini/{MDL}_max_events.pkl', 'rb') as file:
#         max_per_periods=pickle.load(file)

#     duration=get_unlist(len_per_above_threshold)
#     max_inte=get_unlist(max_per_periods)
#     avg_inte=get_unlist(mean_per_periods)

#     print(f"{MDL}\n{len(duration)/1e6},{np.mean(duration):.2f},{np.mean(max_inte):.2f},{np.mean(avg_inte):.2f}")


# len(len_per_above_threshold),len(mean_per_periods),len(max_per_periods)


# duration=get_unlist(len_per_above_threshold)
# max_inte=get_unlist(max_per_periods)
# avg_inte=get_unlist(mean_per_periods)
# len(duration)==len(max_inte)==len(avg_inte)

# H, yedges, xedges = np.histogram2d(max_inte,duration,bins=10);
# H2, yedges2, xedges2 = np.histogram2d(avg_inte,duration,bins=10);
# H=np.where(np.isfinite(np.log(H.astype(int))),np.log(H.astype(int)),0)
# H2=np.where(np.isfinite(np.log(H2.astype(int))),np.log(H2.astype(int)),0)
# # Plot histogram using pcolormesh
# fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=False)
# pcm=ax1.pcolormesh(xedges, yedges, (H), cmap='rainbow')
# ax1.set_ylim(np.min(max_inte), np.max(max_inte))
# ax1.set_xlim(np.min(duration), np.max(duration))
# ax1.set_ylabel('Peak Intensity')
# ax1.set_xlabel('Duration')
# ax1.set_title('Events')
# # ax1.grid()
# cbar = plt.colorbar(pcm)
# # Set colorbar label
# cbar.set_label('', rotation=270, labelpad=20)

# pcm2=ax2.pcolormesh(xedges2, yedges2, (H2), cmap='rainbow')
# ax2.set_ylim(np.min(avg_inte), np.max(avg_inte))
# ax2.set_xlim(np.min(duration), np.max(duration))
# ax2.set_ylabel('Mean Intensity')
# ax2.set_xlabel('Duration')
# ax2.set_title('Events')
# # ax1.grid()
# cbar = plt.colorbar(pcm2)
# # Set colorbar label
# cbar.set_label('', rotation=270, labelpad=20)

# #do a scatter plot of duration vs max intensity
# fig, ax = plt.subplots()
# ax.scatter(duration, max_inte, alpha=0.5)
# ax.set_xlabel('Duration')
# ax.set_ylabel('Peak Intensity')
# ax.set_title('Events')
# # ax.grid()
# plt.show()


# plt.imshow((np.where(np.isfinite(np.log(x1.astype(int))),np.log(x1.astype(int)),0)))



# model=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2001)])


# model_slice=model.isel(time=1).load()

# longitude=[]
# latitude=[]

# for i in tqdm(range(158)):
#     for j in range(272):
#         lon,lat=model_slice.isel(lat=i,lon=j).lon.item(),model_slice.isel(lat=i,lon=j).lat.item()

#         longitude.append(lon)
#         latitude.append(lat)


# ds = xr.Dataset(
#     data_vars=dict(
#         len_event=(["lat", "lon"], len_per_above_threshold),
        
#     ),
#     coords=dict(
#         lon=longitude,
#         lat=latitude
#     ),
#     attrs=model_slice.attrs,
# )