#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
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
import matplotlib.ticker as mticker
from shapely.geometry import mapping
from cartopy import feature as cfeature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

from scripts.utils import *

PF="/mnt/data/RESTRICTED/CARIPARO"
shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

eth_or=xr.open_dataset(f"{PF}/datiDallan/CPM_ETH_Italy_2000-2009_pr_hour.nc").load()
eth_re=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/*.nc").load()
eth_nn=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/nn/*.nc").load()
# eth_la=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/laf/*.nc").load()
eth_or=xr.open_dataset(f"{PF}/datiDallan/CPM_ETH_Italy_2000-2009_pr_hour.nc").isel(time=12).load()
eth_re=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/*2000*.nc").isel(time=12).load()
uas_eth =xr.open_dataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200001010030_200012312330.nc").isel(time=12).load()
vas_eth =xr.open_dataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/vas/vas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200001010030_200012312330.nc").isel(time=12).load()


eth_or_tri=eth_or.where((eth_or.lon > 10.38) & (eth_or.lon < 13.1) & (eth_or.lat > 44.7) & (eth_or.lat < 47.1), drop=True)
eth_re_tri=eth_re.where((eth_re.lon > 10.38) & (eth_re.lon < 13.1) & (eth_re.lat > 44.7) & (eth_re.lat < 47.1), drop=True)
eth_nn_tri=eth_nn.where((eth_nn.lon > 10.38) & (eth_nn.lon < 13.1) & (eth_nn.lat > 44.7) & (eth_nn.lat < 47.1), drop=True)
# eth_la_tri=eth_la.where((eth_la.lon > 10.38) & (eth_la.lon < 13.1) & (eth_la.lat > 44.7) & (eth_la.lat < 47.1), drop=True)
del eth_re,eth_or,eth_nn#,eth_la_tri


"""METADATA Stations"""
meta_stations=pd.read_csv("meta_station_updated_col.csv",index_col=0)
meta_stations


"""MAPS HEAVY QUANTILE"""

q99_or=eth_or_tri.quantile(q=np.arange(0.9,1,0.01),dim='time')
q99_re=eth_re_tri.quantile(q=0.99,dim='time')
q99_nn=eth_nn_tri.quantile(q=0.999,dim='time')
q99_la=eth_la_tri.quantile(q=0.999,dim='time')

qq_or_cdo=xr.load_dataset("/mnt/data/lcesarini/q99_eth_eleonora.nc")
qq_or_cdo_JJA=xr.load_dataset("/mnt/data/lcesarini/q99_eth_eleonora_JJA.nc")
qq_re_cdo=xr.load_dataset("/mnt/data/lcesarini/q99_eth_merged_remap.nc")
qq_re_cdo_JJA=xr.load_dataset("/mnt/data/lcesarini/q99_eth_merged_remap_JJA.nc")


plot_panel_rotated(1,1,
                list_to_plot=[eth_or_tri.pr.isel(time=0)],
                name_fig="orig_slice",
                list_titles=['Original Grid'],
                levels=[lvl_q],
                suptitle="99.9th Quantile all period",
                name_metric=["mm/h"],
                SET_EXTENT=False,
                cmap=[cmap_q],
                proj=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170),
                transform=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170)
)

plot_panel_rotated(1,1,
                list_to_plot=[qq_or_cdo.pr.isel(time=0)],
                name_fig="99_quantile_orig",
                list_titles=['Original Grid'],
                levels=[lvl_q],
                suptitle="99.9th Quantile all period",
                name_metric=["mm/h"],
                SET_EXTENT=True,
                cmap=[cmap_q],
                proj=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170),
                transform=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170)
)

plot_panel_rotated(1,1,
                list_to_plot=[qq_re_cdo.pr.isel(time=0)],
                name_fig="99_quantile_remap",
                list_titles=['Remapped Grid'],
                levels=[lvl_q],
                suptitle="99th Quantile all period",
                name_metric=["mm/h"],
                SET_EXTENT=True,
                cmap=[cmap_q],
                proj=ccrs.PlateCarree(),
                transform=ccrs.PlateCarree()
)

plot_panel_rotated(1,1,
                list_to_plot=[qq_or_cdo_JJA.pr.isel(time=0)],
                name_fig="99_quantile_orig_JJA",
                list_titles=['Original Grid'],
                levels=[lvl_q],
                suptitle="99.9th Quantile JJA",
                name_metric=["mm/h"],
                SET_EXTENT=False,
                cmap=[cmap_q],
                proj=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170),
                transform=ccrs.RotatedPole(pole_latitude=43, pole_longitude=-170)
)

plot_panel_rotated(1,1,
                list_to_plot=[q99_nn.pr],
                name_fig="99_quantile_remapnn",
                list_titles=['Remapped Grid'],
                levels=[lvl_q],
                suptitle="99.9th Quantile nn all period",
                name_metric=["mm/h"],
                SET_EXTENT=False,
                cmap=[cmap_q],
                proj=ccrs.PlateCarree(),
                transform=ccrs.PlateCarree()
)

"""ECDF"""

q95=eth_or_tri.pr.quantile(q=0.95).item()

a=np.sort(eth_or_tri.pr.isel(time=eth_or_tri["time.year"].isin(2004) & eth_or_tri["time.month"].isin(9)).values.reshape(-1))
b=np.sort(eth_re_tri.pr.isel(time=eth_re_tri["time.year"].isin(2004) & eth_re_tri["time.month"].isin(9)).values.reshape(-1))
fig=plt.figure()
ax=plt.axes()
ax.plot(a[a>q95],
        (np.arange(0,a[a>q95].shape[0])+1)/a[a>q95].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax.plot(b[b>q95],
        (np.arange(0,b[b>q95].shape[0])+1)/b[b>q95].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Remapped',
        alpha=0.5)
ax.set_title(f"ECDFs entire are for 2004")
ax.legend()
plt.savefig(f"figures/ecdf_orig_vs_remap_2004_95.png")
plt.close()

q99=eth_or_tri.pr.quantile(q=0.99).item()
q99_1=eth_re_tri.pr.quantile(q=0.99).item()

timemax,latmax,lonmax=np.unravel_index(np.argwhere(eth_or_tri.pr.values > 100),shape=eth_or_tri.pr.shape)

np.argwhere(eth_or_tri.pr.values > 100)


"""
CHECK INDICES TAKEN BY ELEONORA

"""
ij=pd.read_table("griglia_ele.txt",header=None)
ij.columns=["name","i","j"]
ij.i=ij.i-1
ij.j=ij.j-1
ij

for idx in range(meta_station.lon.values.shape[0]):

	i_l,j_l=reg.transform_point(x=meta_station.lon.values[idx],y=meta_station.lat.values[idx],src_crs=rot)
	name_l=meta_station.name.values[idx]

	i_e,j_e=ij[ij.name==name_l].i,ij[ij.name==name_l].j

	if not np.all(eth_or.pr.sel(rlon=i_l,rlat=j_l,method='nearest')==eth_or.pr.isel(rlon=i_e,rlat=j_e)):
		print(f"{idx} not hte same cell")



"""
END OF CHECK
"""
meta_station.lon
x_or,y_or=convert_coords(meta_station.lon.values,meta_station.lat.values,reg=ccrs.RotatedPole(pole_latitude=43.0, pole_longitude=-170.0),rot=ccrs.CRS("WGS84"))

XY=reg.transform_points(ccrs.CRS("WGS84"),meta_station.lon,meta_station.lat)[:,:2]

a_99_stat=eth_or_tri.pr.sel(rlon=XY[:,0],rlat=XY[:,1],method='nearest').isel(time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1)
b_99_stat=eth_re_tri.pr.sel(lon=[x for x in meta_station.lon],lat=[y for y in meta_station.lat],method='nearest').isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1)
c_99_stat=eth_nn_tri.pr.sel(lon=[x for x in meta_station.lon],lat=[y for y in meta_station.lat],method='nearest').isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1)

prec_thr=70

f,ax=plt.subplots(figsize=(9,6))
ax.hist(a_99_stat[a_99_stat > prec_thr ],np.arange(min(a_99_stat[a_99_stat > prec_thr ]), max(a_99_stat[a_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(0.2,0.2,0,0.25),ec="green",label='Original')
ax.hist(b_99_stat[b_99_stat > prec_thr ],np.arange(min(b_99_stat[b_99_stat > prec_thr ]), max(b_99_stat[b_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(0,0,1,0.25),ec="blue",label='conservative')
ax.hist(c_99_stat[c_99_stat > prec_thr ],np.arange(min(c_99_stat[c_99_stat > prec_thr ]), max(c_99_stat[c_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(1,0,0,0.25),ec="red",label='NNeighbour')
ax.set_title(f"Histogram entire area")
ax.set_xlabel("Rainfall [mm/hr]")
ax.set_ylabel("Density")
ax.legend()
plt.savefig(f"figures/pdf_orig_vs_remap_only_station.png")
plt.close()

a_99_stat=np.sort(a_99_stat)
b_99_stat=np.sort(b_99_stat)
c_99_stat=np.sort(c_99_stat)
f,ax=plt.subplots(1,2,figsize=(9,6))
ax[0].plot(a_99_stat[a_99_stat>q99],
        (np.arange(0,a_99_stat[a_99_stat>q99].shape[0])+1)/a_99_stat[a_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax[0].plot(b_99_stat[b_99_stat>q99],
        (np.arange(0,b_99_stat[b_99_stat>q99].shape[0])+1)/b_99_stat[b_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax[0].plot(c_99_stat[c_99_stat>q99],
        (np.arange(0,c_99_stat[c_99_stat>q99].shape[0])+1)/c_99_stat[c_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax[0].set_title(f"ECDFs of the 174 cells containing stations")
ax[0].legend()
# plt.savefig(f"figures/ecdf_orig_vs_remap_99_station.png")
# plt.close()

ax[1].plot(a_99_stat[a_99_stat>q99],
        (np.arange(0,a_99_stat[a_99_stat>q99].shape[0])+1)/a_99_stat[a_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax[1].plot(b_99_stat[b_99_stat>q99],
        (np.arange(0,b_99_stat[b_99_stat>q99].shape[0])+1)/b_99_stat[b_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax[2].plot(c_99_stat[c_99_stat>q99],
        (np.arange(0,c_99_stat[c_99_stat>q99].shape[0])+1)/c_99_stat[c_99_stat>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax[1].set_title(f"Zoom on the extremes")
# ax.vlines(x=106.88115,ymin=0,ymax=1)
ax[1].legend()
ax[1].set_xlim((70,110))
ax[1].set_ylim((0.99998,1.000004))
plt.savefig(f"figures/ecdf_orig_vs_remap_99_station_with_zoom.png")
plt.close()


a_99=np.sort(eth_or_tri.pr.isel(time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b_99=np.sort(eth_re_tri.pr.isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
c_99=np.sort(eth_nn_tri.pr.isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
# d_99=np.sort(eth_la_tri.pr.isel(time=eth_la_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
np.mean(a_99[a_99 > 20][-1000:]-b_99[b_99 > 20][-1000:])
fig=plt.figure()
ax=plt.axes()
plt.hist(a_99[a_99 > 20],25, density=True,color="green",label='Original', alpha=0.5)
plt.hist(b_99[b_99 > 20],25, density=True,color="blue",label='remap', alpha=0.5)
ax.set_title(f"Histogram entire area")
ax.legend()
plt.savefig(f"figures/pdf_orig_vs_remap_99.png")
plt.close()

fig=plt.figure()
ax=plt.axes()
n_a, bins_a, patches_a = ax.hist(a_99,10, density=True)
mu_a = np.nanmean(a_99)  # mean of distribution
sigma_a = np.nanstd(a_99)
# add a 'best fit' line
y_a = ((1 / (np.sqrt(2 * np.pi) * sigma_a)) *
     np.exp(-0.5 * (1 / sigma_a * (bins_a - mu_a))**2))
ax.plot(bins_a,y_a,
        color="blue",label='Original',
        alpha=0.5)
n_b, bins_b, patches_b = ax.hist(b_99,10, density=True)
mu_b = np.nanmean(b_99)  # mean of distribution
sigma_b = np.nanstd(b_99)
# add a 'best fit' line
y_b = ((1 / (np.sqrt(2 * np.pi) * sigma_b)) *
     np.exp(-0.5 * (1 / sigma_b * (bins_b - mu_b))**2))
ax.hist(bins_b,y_b,
        color="green",label='Remap',
        alpha=0.5)
ax.set_title(f"Histogram entire area")
ax.legend()
plt.savefig(f"figures/pdf_orig_vs_remap_99.png")
plt.close()

fig=plt.figure()
ax=plt.axes()
ax.plot(a_99,
        (np.arange(0,a_99.shape[0])+1)/a_99.shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax.plot(b_99,
        (np.arange(0,b_99.shape[0])+1)/b_99.shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Remapped',
        alpha=0.5)
ax.set_title(f"ECDFs entire area")
ax.legend()
plt.savefig(f"figures/ecdf_orig_vs_remap_99.png")
plt.close()




fig=plt.figure()
ax=plt.axes()
ax.plot(a_99[a_99>q99],
        (np.arange(0,a_99[a_99>q99].shape[0])+1)/a_99[a_99>q99].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax.plot(b_99[b_99>q99],
        (np.arange(0,b_99[b_99>q99].shape[0])+1)/b_99[b_99>q99].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax.plot(c_99[c_99>q99],
        (np.arange(0,c_99[c_99>q99].shape[0])+1)/c_99[c_99>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax.plot(d_99[d_99>q99],
        (np.arange(0,d_99[d_99>q99].shape[0])+1)/d_99[d_99>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='LAF',
        alpha=0.5)
ax.set_title(f"ECDFs entire area")
ax.legend()
plt.savefig(f"figures/ecdf_orig_vs_remap_99.png")
plt.close()

np.mean(a_99[a_99 > q99][-1000000:]-b_99[b_99 > q99][-1000000:])

fig=plt.figure()
ax=plt.axes()
ax.plot(a_99[a_99>q99],
        (np.arange(0,a_99[a_99>q99].shape[0])+1)/a_99[a_99>q99].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax.plot(b_99[b_99>q99],
        (np.arange(0,b_99[b_99>q99].shape[0])+1)/b_99[b_99>q99].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax.plot(c_99[c_99>q99],
        (np.arange(0,c_99[c_99>q99].shape[0])+1)/c_99[c_99>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax.plot(d_99[d_99>q99],
        (np.arange(0,d_99[d_99>q99].shape[0])+1)/d_99[d_99>q99].shape[0],
        marker="d",markersize=5,
        color="brown",label='LAF',
        alpha=0.5)
ax.set_title(f"ECDFs entire area for entire period")
ax.vlines(x=106.88115,ymin=0,ymax=1)
ax.legend()
ax.set_xlim((70,110))
ax.set_ylim((0.99998,1.000004))
plt.savefig(f"figures/ecdf_orig_vs_remap_2004_99_zoom.png")
plt.close()




a=np.sort(eth_or_tri.pr.isel(time=eth_or_tri["time.year"].isin([2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b=np.sort(eth_re_tri.pr.isel(time=eth_re_tri["time.year"].isin([2004,2005,2006,2007,2008,2009])).values.reshape(-1))
a=a[~np.isnan(a)]
b=b[~np.isnan(b)]
fig=plt.figure()
ax=plt.axes()
ax.plot(a[-100:],
        (np.arange(0,a[-100:].shape[0])+1)/a[-100:].shape[0],
        marker="d",markersize=5,
        color="blue",label='Original',
        alpha=0.5)
ax.plot(b[-100:],
        (np.arange(0,b[-100:].shape[0])+1)/b[-100:].shape[0],
        marker="d",markersize=5,
        color="darkgreen",label='Remapped',
        alpha=0.5)
ax.set_title(f"ECDFs entire are for 2004,2005,2006")
ax.legend()
plt.savefig(f"figures/ecdf_orig_vs_remap_top100.png")
plt.close()



eth_or_tri.pr.isel(rlon=lonmax,rlat=latmax).lat.item()
eth_or_tri.pr.isel(rlon=69,rlat=52).max().item()

a_99_s=np.sort(eth_or_tri.pr.isel(rlon=32,rlat=52,time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b_99_s=np.sort(eth_re_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=32,rlat=52).lon.item(),lat=eth_or_tri.pr.isel(rlon=32,rlat=52).lat.item(),method='nearest').\
               isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
c_99_s=np.sort(eth_nn_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=32,rlat=52).lon.item(),lat=eth_or_tri.pr.isel(rlon=32,rlat=52).lat.item(), method='nearest').\
               isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
d_99_s=np.sort(eth_la_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=32,rlat=52).lon.item(),lat=eth_or_tri.pr.isel(rlon=32,rlat=52).lat.item(), method='nearest').\
               isel(time=eth_la_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))



fig=plt.figure()
ax=plt.axes()
ax.plot(a_99_s[a_99_s>q99],
        (np.arange(0,a_99_s[a_99_s>q99].shape[0])+1)/a_99_s[a_99_s>q99].shape[0],
        marker="^",markersize=5,
        color="blue",label='Original',
        alpha=0.85)
ax.plot(b_99_s[b_99_s>q99],
        (np.arange(0,b_99_s[b_99_s>q99].shape[0])+1)/b_99_s[b_99_s>q99].shape[0],
        marker="*",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax.plot(c_99_s[c_99_s>q99],
        (np.arange(0,c_99_s[c_99_s>q99].shape[0])+1)/c_99_s[c_99_s>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax.plot(d_99_s[d_99_s>q99],
        (np.arange(0,d_99_s[d_99_s>q99].shape[0])+1)/d_99_s[d_99_s>q99].shape[0],
        marker="d",markersize=5,
        color="brown",label='LAF',
        alpha=0.5)
ax.set_title(f"ECDFs entire area for entire period")
ax.legend()
ax.set_xlim((20,110))
ax.set_ylim((0.98,1.004))
plt.savefig(f"figures/ecdf_orig_vs_remap_2004_99_zoom_singlemax_station1.png")
plt.close()

fig=plt.figure()
ax=plt.axes()
ax.plot(a_99_s[a_99_s>q99],
        (np.arange(0,a_99_s[a_99_s>q99].shape[0])+1)/a_99_s[a_99_s>q99].shape[0],
        marker="^",markersize=5,
        color="blue",label='Original',
        alpha=0.85)
ax.plot(b_99_s[b_99_s>q99],
        (np.arange(0,b_99_s[b_99_s>q99].shape[0])+1)/b_99_s[b_99_s>q99].shape[0],
        marker="*",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax.plot(c_99_s[c_99_s>q99],
        (np.arange(0,c_99_s[c_99_s>q99].shape[0])+1)/c_99_s[c_99_s>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax.set_title(f"ECDFs entire area for entire period")
ax.legend()
ax.set_xlim((4.5,5.5))
ax.set_ylim((0.499,0.501))
plt.savefig(f"figures/ecdf_orig_vs_remap_2004_99_zoom_singlemax_station_mean1.png")
plt.close()

"""ANALYSIS PER STATION"""

lon_st=meta_stations.lon[12]
lat_st=meta_stations.lat[12]

extr_q=np.arange(0.9999,1,0.00001)
extr_Q=[0.5,0.7,0.85,0.9,0.95,0.99,0.99]

def get_nearest(remapped_ds,lon_st,lat_st):
	X_,Y_=convert_coords(X=lon_st,Y=lat_st,
		           		reg=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),
						rot=ccrs.CRS("WGS84"),
				)
	cell_re=remapped_ds.sel(lon=lon_st,lat=lat_st,method='nearest')
	cell_or=eth_or_tri.sel(rlon=X_,rlat=Y_,method='nearest')
    
	return cell_or,cell_re

np.nanmax(eth_or_tri.pr)
np.nanmax(eth_re_tri.pr)
np.nanmax(eth_nn_tri.pr)


extr_q_orig,extr_q_rcon,extr_q_r_nn=[],[],[]
extr_Q_orig,extr_Q_rcon,extr_Q_r_nn=[],[],[]
max_orig,max_rcon,max_r_nn=[],[],[]
for idx,(lon_st,lat_st) in tqdm(enumerate(zip(meta_stations.lon,meta_stations.lat))):

	original,rcon=get_nearest(eth_re_tri,lon_st,lat_st)
	r_nn=get_nearest(eth_nn_tri,lon_st,lat_st)[1]
	# rlaf=get_nearest(eth_la_tri,lon_st,lat_st)[1]



	extr_q_orig.append(np.quantile(original.pr.values.reshape(-1),q=extr_q))
	extr_q_rcon.append(np.quantile(rcon.pr.values.reshape(-1),q=extr_q))
	extr_q_r_nn.append(np.quantile(r_nn.pr.values.reshape(-1),q=extr_q))

	extr_Q_orig.append(np.quantile(original.pr.values.reshape(-1),q=extr_Q))
	extr_Q_rcon.append(np.quantile(rcon.pr.values.reshape(-1),q=extr_Q))
	extr_Q_r_nn.append(np.quantile(r_nn.pr.values.reshape(-1),q=extr_Q))

	max_orig.append(np.nanmax(original.pr.values.reshape(-1)))
	max_rcon.append(np.nanmax(rcon.pr.values.reshape(-1)))
	max_r_nn.append(np.nanmax(r_nn.pr.values.reshape(-1)))
	# extr_q_rlaf=np.quantile(rlaf.pr.values.reshape(-1),q=extr_q)
		
	# mean_squared_error(extr_q_orig,extr_q_rcon)
	# mean_squared_error(extr_q_orig,extr_q_r_nn)
	# mean_squared_error(extr_q_orig,extr_q_rlaf)
	# np.max((extr_q_orig - extr_q_rcon) /  extr_q_orig) * 100
	# np.mean((extr_q_orig - extr_q_rcon) /  extr_q_orig) * 100

	# np.mean(np.abs(extr_q_orig - extr_q_rcon) /  extr_q_orig)
	# np.max(np.abs(extr_q_orig - extr_q_rcon) /  extr_q_orig)



min_diagonal=np.floor(np.min([np.min(extr_q_orig),np.max(max_orig),np.min(max_rcon),np.max(max_r_nn)]))
max_diagonal=np.ceil(np.max([np.min(extr_q_orig),np.max(max_orig),np.min(max_rcon),np.max(max_r_nn)]))

max_bias_cons=np.nanmax(np.abs((np.array(extr_q_orig)-np.array((extr_q_rcon)))/np.array(extr_q_orig)))
max_bias_nn=np.nanmax(np.abs((np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)))

MAX_bias_cons=np.nanmax(np.abs((np.array(max_orig)-np.array((max_rcon)))/np.array(max_orig)))
MAX_bias_nn=np.nanmax(np.abs((np.array(max_orig)-np.array((max_r_nn)))/np.array(max_orig)))

mean_bias_cons=np.nanmean(np.abs((np.array(extr_q_orig)-np.array((extr_q_rcon)))/np.array(extr_q_orig)))
mean_bias_nn=np.nanmean(np.abs((np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)))


MEAN_bias_cons=np.nanmean(np.abs((np.array(max_orig)-np.array((max_rcon)))/np.array(max_orig)))
MEAN_bias_nn=np.nanmean(np.abs((np.array(max_orig)-np.array((max_r_nn)))/np.array(max_orig)))

from sklearn.metrics import mean_absolute_percentage_error
mae_1=mean_absolute_error(np.array(max_orig),np.array(max_rcon))
mae_2=mean_absolute_error(np.array(max_orig),np.array(max_r_nn))
mape_1=mean_absolute_percentage_error(np.array(max_orig),np.array(max_rcon))
mape_2=mean_absolute_percentage_error(np.array(max_orig),np.array(max_r_nn))
rmse_1=np.sqrt(mean_squared_error(np.array(max_orig),np.array(max_rcon)))
rmse_2=np.sqrt(mean_squared_error(np.array(max_orig),np.array(max_r_nn)))

np.unravel_index(np.argmax(np.abs((np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)),axis=None),shape=(174,10))

np.array(extr_q_r_nn).shape

(extr_q_orig[51][9]-extr_q_r_nn[51][9])/extr_q_orig[51][9]


fig,axs=plt.subplots(1,1,
                        figsize=(8,8),constrained_layout=True, squeeze=True)


ax=axs
ax.scatter(extr_q_orig,extr_q_rcon,
                # "-",
                marker="d",#markersize=5,
                color="blue",label='Conservative',
                alpha=0.5)
ax.scatter(extr_q_orig,extr_q_r_nn,
                # "-",
                marker="*",#markersize=5,
                color="red",label='NNeighbour',
                alpha=0.5)
# ax.scatter(extr_q_orig,extr_q_rlaf,
#                 # "-",
#                 marker="d",#markersize=5,
#                 color="brown",label='LAF',
#                 alpha=0.5)
ax.scatter(max_orig,max_rcon,
                # "-",
                marker="d",#markersize=5,
                color="cyan",label='Max Conservative',
                alpha=0.5)
ax.scatter(max_orig,max_r_nn,
                # "-",
                marker="*",#markersize=5,
                color="magenta",label='Max NNeighbour',
                alpha=0.5)
ax.text(x=20,y=70,
		s=
f"""
Max bias orig-cons:{max_bias_cons*100:.2f}%
Max bias orig-nn:{max_bias_nn*100:.2f}%
Mean bias m orig-cons:{mean_bias_cons*100:.2f}%
Mean bias orig-nn:{mean_bias_nn*100:.2f}%
MAX bias orig-cons:{MAX_bias_cons*100:.2f}%
MAX bias orig-nn:{MAX_bias_nn*100:.2f}%
MEAN orig-cons:{MEAN_bias_cons*100:.2f}%
MEAN orig-nn:{MEAN_bias_nn*100:.2f}%
""",
		backgroundcolor='azure')
ax.text(x=70,y=20,
		s=
f"""
Max bias orig-cons:{max_bias_cons*100:.2f}%
Max bias orig-nn:{max_bias_nn*100:.2f}%
Mean bias m orig-cons:{mean_bias_cons*100:.2f}%
Mean bias orig-nn:{mean_bias_nn*100:.2f}%
MAX bias orig-cons:{MAX_bias_cons*100:.2f}%
MAX bias orig-nn:{MAX_bias_nn*100:.2f}%
MEAN orig-cons:{MEAN_bias_cons*100:.2f}%
MEAN orig-nn:{MEAN_bias_nn*100:.2f}%
""",
		backgroundcolor='azure')
ax.plot([min_diagonal, max_diagonal], [min_diagonal, max_diagonal])
# ax.scatter(15, 15,color="red")
ax.set_title(f"quantile considered:\n{extr_q*100}\nand Max value")
ax.legend()
ax.set_xlabel("Original [mm/h]")
ax.set_ylabel("Remap [mm/h]")
plt.suptitle(f"QQ-plot of all stations")
plt.savefig(f"figures/qqplot_original_remapcon.png")
plt.close()


min_diagonal_Q=np.floor(np.min([np.min(extr_Q_orig),np.max(extr_Q_orig),np.min(extr_Q_rcon),np.max(extr_Q_r_nn)]))
max_diagonal_Q=np.ceil(np.max([np.min(extr_Q_orig),np.max(extr_Q_orig),np.min(extr_Q_rcon),np.max(extr_Q_r_nn)]))

fig,axs=plt.subplots(1,1,
                        figsize=(8,8),constrained_layout=True, squeeze=True)


ax=axs
ax.scatter(extr_Q_orig,extr_Q_rcon,
                # "-",
                marker="d",#markersize=5,
                color="blue",label='Conservative',
                alpha=0.5)
ax.scatter(extr_Q_orig,extr_Q_r_nn,
                # "-",
                marker="*",#markersize=5,
                color="red",label='NNeighbour',
                alpha=0.5)
# ax.scatter(max_orig,max_rcon,
#                 # "-",
#                 marker="d",#markersize=5,
#                 color="cyan",label='Max Conservative',
#                 alpha=0.5)
# ax.scatter(max_orig,max_r_nn,
#                 # "-",
#                 marker="*",#markersize=5,
#                 color="magenta",label='Max NNeighbour',
#                 alpha=0.5)
ax.plot([min_diagonal_Q, max_diagonal_Q], [min_diagonal_Q, max_diagonal_Q])
# ax.scatter(15, 15,color="red")
ax.set_title(f"quantile considered:\n{[x*100 for x in extr_Q]}")
ax.legend()
ax.set_xlabel("Original [mm/h]")
ax.set_ylabel("Remap [mm/h]")
plt.suptitle(f"QQ-plot of all stations")
plt.savefig(f"figures/QQplot_original_remapcon.png")
plt.close()


""""

PLOT GRID

"""
for idx,(lon_st,lat_st) in enumerate(zip(meta_stations.lon,meta_stations.lat)):
	xy_or,xy_re=get_nearest(lon_st,lat_st)
	line={
	"name_station":	meta_stations.name.iloc[idx],
	"dist_lon"    :degree_to_meters(xy_or.lon.item()-xy_re.lon.item()),
	"dist_lat"    :degree_to_meters(xy_or.lat.item()-xy_re.lat.item()),
	"err_medio"   :np.nanmean(np.abs(xy_or.pr-xy_re.pr)),
	"err_massimo" :np.nanmax(np.abs(xy_or.pr-xy_re.pr)),
	"err_999"     :np.nanquantile(np.abs(xy_or.pr-xy_re.pr),q=0.999),
	"err_max_ecdf":np.nanmax(np.abs(np.sort(xy_or.pr)-np.sort(xy_re.pr)))

	}

	pd.DataFrame(line,index=[0]).to_csv("df_check_dist.csv",mode='a',header=None,index=False)

l_rx,l_ry=[],[]
for rx,ry in zip(create_list_coords(eth_or_tri.rlon,eth_or_tri.rlat)[:,0],create_list_coords(eth_or_tri.rlon,eth_or_tri.rlat)[:,1]):


	x1_,y1_=convert_coords(rx,ry,rot=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),reg=ccrs.CRS("WGS84"))
	l_rx.append(x1_)
	l_ry.append(y1_)


l_rx=np.array(l_rx)
l_ry=np.array(l_ry)

x_original=create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0]
y_original=create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1]

x_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)]
y_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)]

slice_re=eth_re_tri.pr.isel(time=1)
slice_or=eth_or_tri.pr.isel(time=1)
ax = plt.subplot(projection=ccrs.PlateCarree())
xr.where(slice_re >= 0, 1,np.nan).plot.pcolormesh(x="lon",y="lat",ax=ax, add_colorbar=False,alpha=0.5,cmap="Blues")
# ax.scatter(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:10,0],
# 	   	   create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:10,1],
# 	       color='blue')
ax.scatter(x_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)],
		   y_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)],
	       color='blue')
xr.where(slice_or >= 0, 1,np.nan).plot.pcolormesh(x="rlon",y="rlat",ax=ax, transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),
												  add_colorbar=False,alpha=0.5,cmap="Greens")
# ax.scatter(create_list_coords(eth_or_tri.lon,eth_or_tri.lat)[:10,0],
# 	   	   create_list_coords(eth_or_tri.lon,eth_or_tri.lat)[:10,1],
# 	       color='green')
ax.scatter(l_rx,
	   	   l_ry,
	       color='green')
ax.set_extent([10.39,10.61,44.69,44.91])
gl=ax.gridlines(draw_labels=False,linewidth=.5, color='azure', alpha=0.5, linestyle='--')
gl2=ax.gridlines(draw_labels=False,linewidth=.5, color='indianred', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:10,0] + (0.02749919891357422/2))
gl.ylocator = mticker.FixedLocator(y_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)] - (0.02749919891357422/2))
gl2.xlocator = mticker.FixedLocator(l_rx[(l_rx >= 10.40) & (l_rx <= 10.6) & (l_ry >= 44.7) & (l_ry <= 44.9)] - (0.02749919891357422/2))
gl2.ylocator = mticker.FixedLocator(l_ry[(l_rx >= 10.40) & (l_rx <= 10.6) & (l_ry >= 44.7) & (l_ry <= 44.9)] + (0.02749919891357422/2))
plt.savefig("figures/scatter_grid.png")
plt.close()





import seaborn as sns

ax=plt.axes()
xy_or.where(xy_or.pr > 15).pr.plot.hist(ax=ax,bins=5,density=True,alpha=0.5,color="green")
# sns.kdeplot(xy_or.where(xy_or.pr > 5).pr.values, bw=0.5,ax=ax,color='green')
xy_re.where(xy_re.pr > 15).pr.plot.hist(ax=ax,bins=5,density=True,alpha=0.5,color="blue")
# sns.kdeplot(xy_re.where(xy_re.pr > 5).pr.values, bw=0.5,ax=ax,color='blue')
plt.savefig("hist_cell.png")
plt.close()


ax=plt.axes()
# xy_or.where(xy_or.pr > 5).pr.plot.hist(ax=ax,bins=10,density=True,alpha=0.5,color="green")
sns.kdeplot(eth_re_tri.pr.values, bw=0.5,ax=ax,color='green')
# xy_re.where(xy_re.pr > 5).pr.plot.hist(ax=ax,bins=10,density=True,alpha=0.5,color="blue")
sns.kdeplot(eth_or_tri.pr.values, bw=0.5,ax=ax,color='blue')
# ax.set_ylim([0,0.0005])
# ax.set_xlim([40,50])
plt.savefig("hist_cell.png")
plt.close()







"""
CHECK PR & WIND GRID
"""
eth_or_tri=eth_or.where((eth_or.lon > 10.38) & (eth_or.lon < 13.1) & (eth_or.lat > 44.7) & (eth_or.lat < 47.1), drop=True)
eth_re_tri=eth_re.where((eth_re.lon > 10.38) & (eth_re.lon < 13.1) & (eth_re.lat > 44.7) & (eth_re.lat < 47.1), drop=True)
eth_u__tri=uas_eth.where((uas_eth.lon > 10.38) & (uas_eth.lon < 13.1) & (uas_eth.lat > 44.7) & (uas_eth.lat < 47.1), drop=True)

# l_rx,l_ry=[],[]
# counter=0
# for rx,ry in zip(create_list_coords(eth_or_tri.rlon,eth_or_tri.rlat)[:,0],create_list_coords(eth_or_tri.rlon,eth_or_tri.rlat)[:,1]):


# 	x1_,y1_=convert_coords(rx,ry,rot=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),reg=ccrs.CRS("WGS84"))
# 	l_rx.append(x1_)
# 	l_ry.append(y1_)

# 	counter +=1

# 	if counter % 1000 == 0: 
# 		print(counter)

# l_rx_or_pr=np.array(l_rx)
# l_ry_or_pr=np.array(l_ry)

# l_rx_u,l_ry_u=[],[]
# for rx,ry in zip(create_list_coords(eth_u__tri.rlon,eth_u__tri.rlat)[:,0],create_list_coords(eth_u__tri.rlon,eth_u__tri.rlat)[:,1]):


# 	x1_,y1_=convert_coords(rx,ry,rot=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),reg=ccrs.CRS("WGS84"))
# 	l_rx_u.append(x1_)
# 	l_ry_u.append(y1_)


# l_rx_or_u=np.array(l_rx_u)
# l_ry_or_u=np.array(l_ry_u)

# x_re_pr=create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0]
# y_re_pr=create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1]

# def extract_area(x_arr, y_arr):
# 	return
# x_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)]
# y_original[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)]

# slice_re=eth_re_tri.pr.isel(time=1)
# slice_or=eth_or_tri.pr.isel(time=1)
# ax = plt.subplot(projection=ccrs.PlateCarree())
# xr.where(slice_re >= 0, 1,np.nan).plot.pcolormesh(x="lon",y="lat",ax=ax, add_colorbar=False,alpha=0.5,cmap="Blues")
# # ax.scatter(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:10,0],
# # 	   	   create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:10,1],
# # 	       color='blue')
# ax.scatter(x_re_pr[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)],
# 		   x_re_pr[(x_original >= 10.40) & (x_original <= 10.6) & (y_original >= 44.7) & (y_original <= 44.9)],
# 	       color='blue')
# xr.where(slice_or >= 0, 1,np.nan).plot.pcolormesh(x="rlon",y="rlat",ax=ax, transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),
# 												  add_colorbar=False,alpha=0.5,cmap="Greens")
# # ax.scatter(create_list_coords(eth_or_tri.lon,eth_or_tri.lat)[:10,0],
# # 	   	   create_list_coords(eth_or_tri.lon,eth_or_tri.lat)[:10,1],
# # 	       color='green')
# ax.scatter(l_rx_or_u,
# 	   	   l_rx_or_u,
# 	       color='green')




or_slice=eth_or_tri.where((eth_or_tri.lon > 10.468) & (eth_or_tri.lon < 10.535) &\
		         (eth_or_tri.lat > 44.715) & (eth_or_tri.lat < 44.780), drop=True) 

or_slice_u=eth_u__tri.where((eth_u__tri.lon > 10.468) & (eth_u__tri.lon < 10.535) &\
		         (eth_u__tri.lat > 44.715) & (eth_u__tri.lat < 44.780), drop=True) 


ax = plt.subplot(projection=ccrs.PlateCarree())

##PRECIPITAZIONE ORiGINALE
ax.scatter(create_list_coords(or_slice.rlon.values,or_slice.rlat.values)[:,0],
	   	   create_list_coords(or_slice.rlon.values,or_slice.rlat.values)[:,1],
		   transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),color='black', label="Center cell pr orig")


or_slice.pr.plot.pcolormesh(color='black',alpha=0.15,
			    			add_colorbar=False,
						    cmap="Blues",
						    levels=1,
							transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))
#VENTO ORIGINALE
ax.scatter(create_list_coords(or_slice_u.rlon.values,or_slice_u.rlat.values)[:,0],
	   	   create_list_coords(or_slice_u.rlon.values,or_slice_u.rlat.values)[:,1],
		   transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),color='red', alpha=0.5,label="Center cell u orig")

or_slice_u.uas.plot.pcolormesh(color='red',alpha=0.15,
							   add_colorbar=False,
							   levels=1,
							   transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))

# ax.set_extent([10.4680,10.535,44.715,44.780])
ax.set_extent([10.45,10.55,44.7,44.82])
# or_slice.pr.plot.scatter(ax=ax,transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))
# eth_re_tri.pr.plot.scatter(ax=ax,c='red')
##PRECIPITAZIONE RIMAPPATA

ax.scatter(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0],
	   	   create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1],
	       color='green',label="center cell pr conservative")
gl=ax.gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0] + (0.02749919891357422/2))
gl.ylocator = mticker.FixedLocator(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1] + (0.02749919891357422/2))
# gl2=ax.gridlines(draw_labels=True,linewidth=.5, color='indianred', alpha=0.5, linestyle='--')
# gl2.xlocator = mticker.FixedLocator(l_rx_or_pr  + (0.02749919891357422/2))
# gl2.ylocator = mticker.FixedLocator(l_ry_or_pr + (0.02749919891357422/2))
plt.title("")
plt.legend(loc="upper left",edgecolor='black',shadow=True)
plt.savefig("figures/scatter_grid_wind.png")
plt.close()



"""
CNRM
"""
cnrm_pr_or=xr.open_dataset(f"/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc").isel(time=12).load()
cnrm_pr_re=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr/*2000*.nc").isel(time=12).load()
cnrm_u_ori=xr.open_dataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc").isel(time=12).load()

cnrm_or_tri=cnrm_pr_or.where((cnrm_pr_or.lon > 10.38) & (cnrm_pr_or.lon < 13.1) & (cnrm_pr_or.lat > 44.7) & (cnrm_pr_or.lat < 47.1), drop=True)
cnrm_re_tri=cnrm_pr_re.where((cnrm_pr_re.lon > 10.38) & (cnrm_pr_re.lon < 13.1) & (cnrm_pr_re.lat > 44.7) & (cnrm_pr_re.lat < 47.1), drop=True)
cnrm_u_tri=cnrm_u_ori.where((cnrm_u_ori.lon > 10.38) & (cnrm_u_ori.lon < 13.1) & (cnrm_u_ori.lat > 44.7) & (cnrm_u_ori.lat < 47.1), drop=True)


cnrm_or_tri_slice=cnrm_or_tri.where((cnrm_or_tri.lon > 10.468) & (cnrm_or_tri.lon < 10.535) &\
		         (cnrm_or_tri.lat > 44.715) & (cnrm_or_tri.lat < 44.780), drop=True) 

cnrm_u_tri_slice=cnrm_u_tri.where((cnrm_u_tri.lon > 10.468) & (cnrm_u_tri.lon < 10.535) &\
		         (cnrm_u_tri.lat > 44.715) & (cnrm_u_tri.lat < 44.780), drop=True) 


ax = plt.subplot(projection=ccrs.PlateCarree())

cnrm_or_tri_slice.pr.plot.pcolormesh(color='black',alpha=0.15,
			    			add_colorbar=False,
						    cmap="Blues",
						    levels=1,
							# transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43)
							)

##PRECIPITAZIONE ORiGINALE
ax.scatter(create_list_coords(cnrm_or_tri_slice.x.values,cnrm_or_tri_slice.y.values)[:,0],
	   	   create_list_coords(cnrm_or_tri_slice.x.values,cnrm_or_tri_slice.y.values)[:,1],
		   transform=ccrs.LambertConformal(central_longitude=8.48,central_latitude=44.88),color='black', label="Center cell pr orig")

gl=ax.gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
plt.savefig("figures/scatter_grid_wind_cnrm.png")
plt.close()


or_slice.pr.plot.pcolormesh(color='black',alpha=0.15,
			    			add_colorbar=False,
						    cmap="Blues",
						    levels=1,
							transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))
#VENTO ORIGINALE
ax.scatter(create_list_coords(or_slice_u.rlon.values,or_slice_u.rlat.values)[:,0],
	   	   create_list_coords(or_slice_u.rlon.values,or_slice_u.rlat.values)[:,1],
		   transform=ccrs.RotatedPole(pole_longitude=-170,standard_parallel=44.88),color='red', alpha=0.5,label="Center cell u orig")

or_slice_u.uas.plot.pcolormesh(color='red',alpha=0.15,
							   add_colorbar=False,
							   levels=1,
							   transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))

# ax.set_extent([10.4680,10.535,44.715,44.780])
ax.set_extent([10.45,10.55,44.7,44.82])
# or_slice.pr.plot.scatter(ax=ax,transform=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43))
# eth_re_tri.pr.plot.scatter(ax=ax,c='red')
##PRECIPITAZIONE RIMAPPATA

ax.scatter(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0],
	   	   create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1],
	       color='green',label="center cell pr conservative")
gl=ax.gridlines(draw_labels=True,linewidth=.5, color='blue', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,0] + (0.02749919891357422/2))
gl.ylocator = mticker.FixedLocator(create_list_coords(eth_re_tri.lon,eth_re_tri.lat)[:,1] + (0.02749919891357422/2))
# gl2=ax.gridlines(draw_labels=True,linewidth=.5, color='indianred', alpha=0.5, linestyle='--')
# gl2.xlocator = mticker.FixedLocator(l_rx_or_pr  + (0.02749919891357422/2))
# gl2.ylocator = mticker.FixedLocator(l_ry_or_pr + (0.02749919891357422/2))
plt.title("")
plt.legend(loc="upper left",edgecolor='black',shadow=True)
plt.savefig("figures/scatter_grid_wind_cnrm.png")
plt.close()
