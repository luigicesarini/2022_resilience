#! /home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
"""
Create xarray from list of defined events
"""

import os
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience")
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from resilience.utils import get_unlist,get_palettes,plot_bin_hist,read_file_events

os.chdir("/mnt/beegfs/lcesarini/2022_resilience")
sftlf=xr.open_dataset("/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS/ECMWF-ERAINT"
SEAS="JJA"

model=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2001)])
model_slice=model.isel(time=1).load()

# xr.where(sftlf.sftlf > 50,1,0)

idx_on_land=np.argwhere(xr.where(sftlf.sftlf > 50,1,0).values.reshape(-1)==1)
sftlf.sftlf.values.reshape(-1).shape[0]



"""
READ THE FILES
"""

model=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2001)])


model_slice=model.isel(time=1).load()

longitude=[]
latitude=[]

import numpy as np
for i in range(158):
    for j in range(272):
        lon,lat=model_slice.isel(lat=i,lon=j).lon.item(),model_slice.isel(lat=i,lon=j).lat.item()

        longitude.append(lon)
        latitude.append(lat)

# bins_pr_avg_inte=np.load("/mnt/beegfs/lcesarini/bins_pr_avg_inte.npy")
# bins_pr_max_inte=np.load("/mnt/beegfs/lcesarini/bins_pr_max_inte.npy")
# bins_pr_duration=np.load("/mnt/beegfs/lcesarini/bins_pr_duration.npy")

bins_mw_avg_inte=np.load("/mnt/beegfs/lcesarini/bins_mw_avg_inte.npy")
bins_mw_max_inte=np.load("/mnt/beegfs/lcesarini/bins_mw_max_inte.npy")
bins_mw_duration=np.load("/mnt/beegfs/lcesarini/bins_mw_duration.npy")

#SPHERA
l_pr_sph,a_pr_sph,m_pr_sph=read_file_events(EV="pr",MDL='SPHERA',THR=90,INDICES=idx_on_land,SEAS="JJA")
l_mw_sph,a_mw_sph,m_mw_sph=read_file_events(EV="mw",MDL='SPHERA',THR=90,INDICES=idx_on_land,SEAS="JJA")
l_cb_sph,a_cb_sph,m_cb_sph=read_file_events(EV="combined",MDL='SPHERA',THR=90,INDICES=idx_on_land,SEAS="JJA")

#CPM
list_cpm_pr={}
list_xr_cpm_pr=[]
for i,MDL in enumerate(["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom","MOHC"]):
    l_pr_cpm,a_pr_cpm,m_pr_cpm=read_file_events(EV="pr",MDL=MDL,THR=90,INDICES=np.arange(len(latitude)),SEAS="JJA",WH=True)

    dict={MDL:{'length':l_pr_cpm,'avg':a_pr_cpm,'max':m_pr_cpm}} 
    list_cpm_pr.update(dict)


    ds_pr_cpm = xr.Dataset(
        data_vars={
            "n_event":(["lat", "lon"],np.array([len(x) for x in l_pr_cpm]).reshape(158,272)),
            "avg_int":(["lat", "lon"],np.array([np.median(x) for x in a_pr_cpm]).reshape(158,272)),
            "max_int":(["lat", "lon"],np.array([np.median(x) for x in m_pr_cpm]).reshape(158,272)),
        },
        coords={
            'lon':np.unique(longitude),
            'lat':np.unique(latitude)
        },
        attrs=model_slice.attrs,
    )

    list_xr_cpm_pr.append(ds_pr_cpm)



list_cpm_mw={}
list_xr_cpm_mw=[]
for i,MDL in enumerate(["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]):
    l_mw_cpm,a_mw_cpm,m_mw_cpm=read_file_events(EV="mw",MDL=MDL,THR=90,INDICES=np.arange(len(latitude)),SEAS="JJA")

    dict={MDL:{'length':l_mw_cpm,'avg':a_mw_cpm,'max':m_mw_cpm}} 
    list_cpm_mw.update(dict)

    ds_mw_cpm = xr.Dataset(
        data_vars={
            'n_event':(["lat", "lon"],np.array([len(x) for x in l_mw_cpm]).reshape(158,272)),
            'avg_int':(["lat", "lon"],np.array([np.median(x) for x in a_mw_cpm]).reshape(158,272)),
            'max_int':(["lat", "lon"],np.array([np.median(x) for x in m_mw_cpm]).reshape(158,272)),
        },
        coords={
            'lon':np.unique(longitude),
            'lat':np.unique(latitude)
        },
        attrs=model_slice.attrs,
    )

    list_xr_cpm_mw.append(ds_mw_cpm)

"""
2D HISTORGRAM

"""
duration=get_unlist(l_pr_sph)
avg_inte=get_unlist(a_pr_sph)
max_inte=get_unlist(m_pr_sph)
# H, yedges, xedges = np.histogram2d(max_inte,
#                                 duration,
#                                 bins=[
#                                     np.arange(np.min(max_inte).astype(np.int32),
#                                                 np.max(max_inte).astype(np.int32),
#                                                 (np.max(max_inte).astype(np.int32)-np.min(max_inte).astype(np.int32))/100),
#                                     np.arange(np.min(duration).astype(np.int32),
#                                                 np.max(duration).astype(np.int32)+1,1),

#                                     ]);

bins_pr_avg_inte=np.arange(np.min(avg_inte).astype(np.int32),
                                                np.max(avg_inte).astype(np.int32),
                                                (np.max(avg_inte).astype(np.int32)-np.min(avg_inte).astype(np.int32))/300)

bins_pr_max_inte=np.arange(np.min(max_inte).astype(np.int32),
                                                np.max(max_inte).astype(np.int32),
                                                (np.max(max_inte).astype(np.int32)-np.min(max_inte).astype(np.int32))/300)

bins_pr_duration=np.arange(np.min(duration).astype(np.int32),
                                                np.max(duration).astype(np.int32)+1,1)
import matplotlib as mpl
from resilience.utils import plot_bin_hist
plot_bin_hist(l_pr_sph,m_pr_sph,a_pr_sph,
              bins_pr_max_inte,bins_pr_avg_inte,bins_pr_duration,
              "SPHERA","Precipitation",
              SAVE=True,
              palette=mpl.colormaps['gist_earth_r']
              )
H_sph_max,H_sph_mean=plot_bin_hist(l_mw_sph,m_mw_sph,a_mw_sph,
              bins_mw_max_inte,bins_mw_avg_inte,bins_mw_duration,
              "SPHERA","Wind",SAVE=False,
              palette=mpl.colormaps['gist_earth_r']
              )

for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    print(np.nanmax(get_unlist(list_cpm_mw[NM]['max'])))

np.nanmax(get_unlist(m_mw_sph))

plot_bin_hist(list_cpm_mw['KNMI']['length'],list_cpm_mw['KNMI']['max'],list_cpm_mw['KNMI']['avg'],
              bins_mw_max_inte,bins_mw_avg_inte,bins_mw_duration,
              'KNMI',"Wind",SAVE=True,
              palette=mpl.colormaps['rainbow']
              )
#SPHERA
H_sph_pr, yedges, xedges = np.histogram2d(
                                        get_unlist(m_pr_sph),
                                        get_unlist(l_pr_sph),
                                        bins=[
                                            bins_pr_max_inte,
                                            bins_pr_duration,
                                            ]);


#CPM
list_H=[]
for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    H, yedges, xedges = np.histogram2d(get_unlist(list_cpm_pr[NM]['max']),
                                       get_unlist(list_cpm_pr[NM]['length']),
                                        bins=[
                                            bins_pr_max_inte,
                                            bins_pr_duration,
                                            ]);

    list_H.append(H)
list_H2=[]
for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    H, yedges, xedges = np.histogram2d(get_unlist(list_cpm_pr[NM]['avg']),
                                       get_unlist(list_cpm_pr[NM]['length']),
                                        bins=[
                                            bins_pr_avg_inte,
                                            bins_pr_duration,
                                            ]);

    list_H2.append(H)


H=np.stack(list_H,axis=0).mean(axis=0)
H2=np.stack(list_H2,axis=0).mean(axis=0)

"""
START BIAS PRECIPITATION
"""
import matplotlib as mpl
cmap = (mpl.colors.ListedColormap([
                                    '#760421',
                                    '#BC2D35', 
                                    '#E27B62',
                                    '#F7BA9D',
                                    # '#FAE9DF',
                                    '#F1F1F1',
                                    '#F1F1F1',
                                    '#E3EEF3',
                                    '#ABD1E5',
                                    '#62A7CD',
                                    '#2B73B2',
                                    '#0A3A70'
                                    ]))


BIAS_H=(np.log(H)-np.log(H_sph_pr))/np.log(H_sph_pr) * 100
BIAS_H=np.where(np.isfinite(BIAS_H),BIAS_H,np.nan)

boundaries = np.arange(-50, 51, 10)

# Create a colormap
# cmap = plt.cm.get_cmap('RdBu', len(boundaries) - 1)
from matplotlib.colors import BoundaryNorm

# Create a BoundaryNorm instance
norm = BoundaryNorm(boundaries, cmap.N, clip=True)
fig, ax2 = plt.subplots(ncols=1,figsize=(8,8), sharey=False)

pcm2=ax2.pcolormesh(bins_pr_duration, np.log(bins_pr_avg_inte), BIAS_H, 
                    cmap=cmap,#mpl.colormaps['turbo'],
                    norm=norm#mpl.colors.Normalize(vmin=0.0,vmax=0.04)
                    )
ax2.set_ylim(np.log(np.min(bins_pr_avg_inte)), np.log(np.max(bins_pr_avg_inte)))
ax2.set_xlim(np.min(bins_pr_duration), np.min([np.max(bins_pr_duration),8]))
ax2.set_ylabel('Mean Intensity')
ax2.set_xlabel('Duration')
ax2.set_title('')
# ax2.set_yticks(np.arange(0,100,2.5))
# ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
ax2.set_xticks(np.arange(1,11,2))
ax2.set_xticklabels(labels=np.arange(1,11,2))
ax2.set_yticks(np.log([5,10,20,50,100,170]))
ax2.set_yticklabels([5,10,20,50,100,170])
cbar = plt.colorbar(pcm2)
# Set colorbar label
cbar.set_label('Bias [%]', rotation=0, labelpad=20)
plt.title(f"Bias of CPM vs SPHERA in identifying the events")
plt.show()
plt.savefig("XX.png")
# plt.savefig(f"/mnt/beegfs/lcesarini/ENSEMBLE_precipitation.png")
plt.close()
"""
END BIAS PRECIPITATION
"""

fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)

pcm=ax1.pcolormesh(bins_pr_duration, np.log(bins_pr_max_inte), np.where(H==0,np.nan,H)/np.nansum(np.where(H==0,np.nan,H)), 
                    cmap=mpl.colormaps['gist_earth_r'],
                    norm=mpl.colors.Normalize(vmin=0.0,vmax=0.08)
                    )

ax1.set_ylim(np.log(np.min(bins_pr_max_inte)), np.log(np.max(bins_pr_max_inte)))
ax1.set_xlim(np.min(bins_pr_duration), np.min([np.nanmax(bins_pr_duration),8]))
ax1.set_ylabel('Peak Intensity')
ax1.set_xlabel('Duration')
ax1.set_title('')
ax1.set_xticks(np.arange(1,11,2))
ax1.set_xticklabels(np.arange(1,11,2))
ax1.set_yticks(np.log([5,10,20,50,100,170]))
ax1.set_yticklabels([5,10,20,50,100,170])
# ax1.grid()
cbar = plt.colorbar(pcm)
# Set colorbar label
cbar.set_label('', rotation=270, labelpad=20)

pcm2=ax2.pcolormesh(bins_pr_duration, np.log(bins_pr_avg_inte), np.where(H2==0,np.nan,H2)/np.nansum(np.where(H==0,np.nan,H)), 
                    cmap=mpl.colormaps['gist_earth_r'],
                    norm=mpl.colors.Normalize(vmin=0.0,vmax=0.04)
                    )
ax2.set_ylim(np.log(np.min(bins_pr_avg_inte)), np.log(np.max(bins_pr_avg_inte)))
ax2.set_xlim(np.min(bins_pr_duration), np.min([np.max(bins_pr_duration),8]))
ax2.set_ylabel('Mean Intensity')
ax2.set_xlabel('Duration')
ax2.set_title('')
# ax2.set_yticks(np.arange(0,100,2.5))
# ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
ax2.set_xticks(np.arange(1,11,2))
ax2.set_xticklabels(labels=np.arange(1,11,2))
ax2.set_yticks(np.log([5,10,20,50,100,170]))
ax2.set_yticklabels([5,10,20,50,100,170])
cbar = plt.colorbar(pcm2)
# Set colorbar label
cbar.set_label('', rotation=270, labelpad=20)
plt.suptitle(f"Ensemble for Precipitation")
plt.savefig(f"/mnt/beegfs/lcesarini/ENSEMBLE_precipitation.png")
plt.close()

"""
WIND
"""
#SPHERA
H_sph_mw, yedges, xedges = np.histogram2d(
                                        get_unlist(a_mw_sph),
                                        get_unlist(l_mw_sph),
                                        bins=[
                                            bins_mw_avg_inte,
                                            bins_mw_duration,
                                            ]);

list_H_w=[]
for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    H, yedges, xedges = np.histogram2d(get_unlist(list_cpm_mw[NM]['max']),
                                       get_unlist(list_cpm_mw[NM]['length']),
                                        bins=[
                                            bins_mw_max_inte,
                                            bins_mw_duration,
                                            ]);

    list_H_w.append(H)

H_mw=np.stack(list_H_w,axis=0).mean(axis=0)
list_H2_w=[]
# for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    # import seaborn as sns
    # sns.kdeplot(get_unlist(list_cpm_mw[NM]['max']),label=NM)
    # sns.kdeplot(get_unlist(list_cpm_mw['CMCC']['max']),label='CMCC')
    # sns.kdeplot(get_unlist(list_cpm_mw['CNRM']['max']),label='CNRM')
    # sns.kdeplot(get_unlist(list_cpm_mw['KIT']['max']),label='KIT')
    # sns.kdeplot(get_unlist(list_cpm_mw['ETH']['max']),label='ETH')
    # sns.kdeplot(get_unlist(list_cpm_mw['ICTP']['max']),label='ICTP')
    # sns.kdeplot(get_unlist(list_cpm_mw['HCLIMcom']['max']),label='HCLIMcom')
    # plt.legend()
    # plt.savefig(f"/mnt/beegfs/lcesarini/kde_wind.png")

for NM in ["KNMI","CMCC","CNRM","KIT","ETH","ICTP","HCLIMcom"]:
    H, yedges, xedges = np.histogram2d(get_unlist(list_cpm_mw[NM]['avg']),
                                       get_unlist(list_cpm_mw[NM]['length']),
                                        bins=[
                                            bins_mw_avg_inte,
                                            bins_mw_duration,
                                            ]);

    list_H2_w.append(H)


H_mw=np.stack(list_H_w,axis=0).mean(axis=0)
H2_mw=np.stack(list_H2_w,axis=0).mean(axis=0)

H_mw=np.where(H_mw==0,np.nan,H_mw)/np.nansum(np.where(H_mw==0,np.nan,H_mw))
H2_mw=np.where(H2_mw==0,np.nan,H2_mw)/np.nansum(np.where(H2_mw==0,np.nan,H2_mw))
H_sph_mw=np.where(H_sph_mw==0,np.nan,H_sph_mw)/np.nansum(np.where(H_sph_mw==0,np.nan,H_sph_mw))
BIAS_H=(H2_mw-H_sph_mw)/H_sph_mw * 100
BIAS_H=np.where(np.isfinite(BIAS_H),BIAS_H,np.nan)
BIAS_H=np.where(BIAS_H < 500,BIAS_H,500)
import seaborn as sns
sns.boxplot(BIAS_H.flatten())
plt.show()
    


pcm=plt.imshow(H_sph_mw)
plt.colorbar(pcm)
plt.show()

boundaries = [0,1,2,3]
boundaries = np.arange(-50, 51, 5)
# BIAS_H=H_mw
# Create a colormap
cmap = plt.cm.get_cmap('viridis', len(boundaries) - 1)
from matplotlib.colors import BoundaryNorm

# Create a BoundaryNorm instance
norm = BoundaryNorm(boundaries, cmap.N, clip=True)
fig, ax2 = plt.subplots(ncols=1,figsize=(8,8), sharey=False)

pcm2=ax2.pcolormesh(bins_mw_duration, bins_mw_max_inte, BIAS_H, 
                    cmap=cmap,#mpl.colormaps['turbo'],
                    # norm=norm#mpl.colors.Normalize(vmin=0.0,vmax=0.04)
                    )
ax2.set_ylim(np.min(bins_mw_max_inte), np.max(bins_mw_max_inte))
ax2.set_xlim(np.min(bins_mw_duration), np.min([np.max(bins_mw_duration),8]))
ax2.set_ylabel('Mean Intensity')
ax2.set_xlabel('Duration')
ax2.set_title('')
# ax2.set_yticks(np.arange(0,100,2.5))
# ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
# ax2.set_xticks(np.arange(1,11,2))
# ax2.set_xticklabels(labels=np.arange(1,11,2))
# ax2.set_yticks([5,10,20,50,100,170]))
# ax2.set_yticklabels([5,10,20,50,100,170])
cbar = plt.colorbar(pcm2)
# Set colorbar label
cbar.set_label('Bias [%]', rotation=0, labelpad=20)
plt.suptitle(f"Bias for Wind")
plt.show()
plt.savefig("XX_2.png")
# plt.savefig(f"/mnt/beegfs/lcesarini/ENSEMBLE_precipitation.png")
plt.close()



np.nanmax(np.where(H_mw==0,np.nan,H_mw)/np.nansum(np.where(H_mw==0,np.nan,H_mw)))
[print(np.nanmax(np.where(x==0,np.nan,x)/np.nansum(np.where(x==0,np.nan,x)))) for x in list_H_w]

import matplotlib as mpl


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)

pcm=ax1.pcolormesh(bins_mw_duration, (bins_mw_max_inte), np.where(H_mw==0,np.nan,H_mw)/np.nansum(np.where(H_mw==0,np.nan,H_mw)), 
                    cmap=mpl.colormaps['gist_earth_r'],
                    norm=mpl.colors.Normalize(vmin=0.0025,vmax=0.05)
                    )
ax1.set_ylim((np.min(bins_mw_max_inte)), (np.max(bins_mw_max_inte)))
# ax1.set_ylim((np.min(max_inte)), 20)
ax1.set_xlim(np.min(bins_mw_duration),  np.min([np.max(bins_mw_duration),8]))        # ax1.set_xlim(np.min(duration), 20)
ax1.set_ylabel('Peak Intensity')
ax1.set_xlabel('Duration')
ax1.set_title('')
# ax1.set_xticks(np.arange(0,10,2))
# ax1.set_xticklabels(np.arange(0,10,2))
# ax1.set_yticks(([5,10,20,50,100,170]))
# ax1.set_yticklabels([5,10,20,50,100,170])
# ax1.grid()
cbar = plt.colorbar(pcm)
# Set colorbar label
cbar.set_label('', rotation=270, labelpad=20)

pcm2=ax2.pcolormesh(bins_mw_duration, (bins_mw_avg_inte), np.where(H2_mw==0,np.nan,H2_mw)/np.nansum(np.where(H2_mw==0,np.nan,H2_mw)),
                    cmap=mpl.colormaps['gist_earth_r'],
                    norm=mpl.colors.Normalize(vmin=0.0025,vmax=0.05)
                    )
ax2.set_ylim((np.min(bins_mw_avg_inte)), (np.max(bins_mw_avg_inte)))
ax2.set_xlim(np.min(bins_mw_duration),  np.min([np.max(bins_mw_duration),8]))
ax2.set_ylabel('Mean Intensity')
ax2.set_xlabel('Duration')
ax2.set_title('')

cbar = plt.colorbar(pcm2)
# Set colorbar label
cbar.set_label('', rotation=270, labelpad=20)
plt.suptitle(f"Ensemble for Wind")
plt.savefig(f"/mnt/beegfs/lcesarini/ENSEMBLE_wind.png")
plt.close()

# duration=get_unlist(l_mw_sph)
# avg_inte=get_unlist(a_mw_sph)
# max_inte=get_unlist(m_mw_sph)




# np.save("/mnt/beegfs/lcesarini/bins_pr_avg_inte.npy",bins_pr_avg_inte)
# np.save("/mnt/beegfs/lcesarini/bins_pr_max_inte.npy",bins_pr_max_inte)
# np.save("/mnt/beegfs/lcesarini/bins_pr_duration.npy",bins_pr_duration)

# bins_mw_avg_inte=np.arange(np.min(avg_inte).astype(np.int32),
#                                                 np.max(avg_inte).astype(np.int32),
#                                                 (np.max(avg_inte).astype(np.int32)-np.min(avg_inte).astype(np.int32))/100)

# bins_mw_max_inte=np.arange(np.min(max_inte).astype(np.int32),
#                                                 np.max(max_inte).astype(np.int32),
#                                                 (np.max(max_inte).astype(np.int32)-np.min(max_inte).astype(np.int32))/100)

# bins_mw_duration=np.arange(np.min(duration).astype(np.int32),
#                                                 np.max(duration).astype(np.int32)+1,1)

# np.save("/mnt/beegfs/lcesarini/bins_mw_avg_inte.npy",bins_mw_avg_inte)
# np.save("/mnt/beegfs/lcesarini/bins_mw_max_inte.npy",bins_mw_max_inte)
# np.save("/mnt/beegfs/lcesarini/bins_mw_duration.npy",bins_mw_duration)

# fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)

# H, yedges, xedges = np.histogram2d(max_inte,
#                                 duration,
#                                 bins=[bins_mw_avg_inte,bins_mw_duration]);



# H2, yedges2, xedges2 = np.histogram2d(avg_inte,
#                                       duration,
#                                       bins=[bins_mw_max_inte,bins_mw_duration]);

# pcm=ax1.pcolormesh(xedges, (yedges), (np.where(H==0,np.nan,H)/len(duration)),
#                     cmap=mpl.colormaps['twilight'].resampled(9),
#                     norm=mpl.colors.Normalize(vmin=0.001,vmax=0.05))
# ax1.set_ylim((np.min(max_inte)), (np.max(max_inte)))
# # ax1.set_ylim((np.min(max_inte)), 20)
# ax1.set_xlim(np.min(duration),  np.min([np.max(duration),8]))        # ax1.set_xlim(np.min(duration), 20)
# ax1.set_ylabel('Peak Intensity')
# ax1.set_xlabel('Duration')
# ax1.set_title('')
# # ax1.set_xticks(np.arange(0,10,2))
# # ax1.set_xticklabels(np.arange(0,10,2))
# # ax1.set_yticks(([5,10,20,50,100,170]))
# # ax1.set_yticklabels([5,10,20,50,100,170])
# # ax1.grid()
# cbar = plt.colorbar(pcm)
# # Set colorbar label
# cbar.set_label('', rotation=270, labelpad=20)

# pcm2=ax2.pcolormesh(xedges2, (yedges2), (np.where(H2==0,np.nan,H2)/len(duration)),
#                     cmap=mpl.colormaps['twilight'].resampled(9),
#                     norm=mpl.colors.Normalize(vmin=0.001,vmax=0.05))
# ax2.set_ylim((np.min(avg_inte)), (np.max(avg_inte)))
# ax2.set_xlim(np.min(duration),  np.min([np.max(duration),8]))
# ax2.set_ylabel('Mean Intensity')
# ax2.set_xlabel('Duration')
# ax2.set_title('')
# # ax2.set_yticks(np.arange(0,100,2.5))
# # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
# # ax2.set_xticks(np.arange(0,10,2))
# # ax2.set_xticklabels(labels=np.arange(0,10,2))
# # ax2.set_yticks(([5,10,20,50,100,170]))
# # ax2.set_yticklabels([5,10,20,50,100,170])
# # ax2.grid()
# cbar = plt.colorbar(pcm2)
# # Set colorbar label
# cbar.set_label('', rotation=270, labelpad=20)
# plt.suptitle(f"SPHERA for Wind")
# plt.show()

print("FINISHED MAKE EVENTS")
"""
MAPS

"""

"""
VIOLIN PLOT

"""