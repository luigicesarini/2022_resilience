#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
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

from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

# from scripts.utils import *

seasons=['DJF','JJA']
list_ms=['Frequency','Intensity','Heavy Prec.']
abbr_ms=['f','i','q']

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()

PATH_COMMON="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"
shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")
ele = xr.load_dataset(f"{PATH_COMMON}/ICTP/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc")
PLOT_BOXPLOTS=False
if __name__ == "__main__":
	#MODELS WITH BOTH WIND SPEED AND WIND GUSTS
	name_models=[#'HCLIMcom',
					'CNRM',
					#'KNMI'
					]
	model="KNMI"
	path_mw=f"{PATH_COMMON}/{model}/CPM/mw"
	path_wg=f"{PATH_COMMON}/{model}/CPM/wsgsmax"

	#mw=xr.open_mfdataset(f"{path_mw}/*nc").load() / 3600
	mw=xr.open_mfdataset(f"{path_mw}/*nc").load() 
	wg=xr.open_mfdataset(f"{path_wg}/*nc").load()


	mw_d=mw.resample(time="D").max()
	wg_d=wg.resample(time="D").max()

	timemax,latmax,lonmax=np.unravel_index(np.argmax(mw_d.mw.values,axis=None),mw_d.mw.shape)
	timemax2,latmax2,lonmax2=np.unravel_index(np.argmax(wg_d.wsgsmax.values,axis=None),wg_d.wsgsmax.shape)


	a=mw_d.mw.isel(lon=lonmax,lat=latmax).values.reshape(-1)
	b=wg_d.wsgsmax.isel(lon=lonmax,lat=latmax).values.reshape(-1)


	def pearson_coeff(a,b):
		return (len(a) * np.sum(a*b) - np.sum(a)*np.sum(b)) / (np.sqrt(len(a) * np.sum(a**2) - np.power(np.sum(a),2)) * np.sqrt(len(b) * np.sum(b**2) - np.power(np.sum(b),2))) 

	
	temporal_correlation=mw_d.mw.values
	temporal_correlation=np.zeros(shape=(158,272))*np.nan


	for lon in tqdm(range(272),total=272):
		for lat in range(158):
			temporal_correlation[lat,lon]=pearson_coeff(mw_d.mw.isel(lon=lon,lat=lat).values.reshape(-1),
													    wg_d.wsgsmax.isel(lon=lon,lat=lat).values.reshape(-1)) 


	plt.plot(mw_d.mw.isel(lon=lon,lat=lat,time=np.arange(50)).values.reshape(-1))
	plt.plot(wg_d.wsgsmax.isel(lon=lon,lat=lat,time=np.arange(50)).values.reshape(-1))
	plt.savefig("figures/fig1.png")
	plt.close()


	ds_tc = xr.Dataset(
		data_vars=dict(
			corr=(["lat", "lon"],temporal_correlation),
		),
		coords=dict(
			lon=mw_d.lon,
			lat=mw_d.lat,
		),
		attrs=mw_d.attrs,
	)

	f,ax=plt.subplots(1,1,figsize=(8,8),subplot_kw={"projection":ccrs.PlateCarree()})
	ds_tc.corr.plot.pcolormesh(ax=ax,cmap="RdBu",add_colorbar=True,
			    levels=np.arange(0.825,0.999999999,(0.999999999-0.825)/7))
	ax.coastlines()
	shp_triveneto.boundary.plot(ax=ax,ec='red')
	ax.set_extent([10.2,13.2,44.5,47.4])
	ax.set_title(f"Temporal correlation according to Pearson coefficient")
	plt.savefig("figures/temp_corr.png")
	plt.close()
	
	fig,axs=plt.subplots(1,1,
				figsize=(12,12),constrained_layout=True, squeeze=True)

	ax=axs
	a=mw_d.mw.isel(lon=lonmax,lat=latmax,time=np.arange(timemax-30,timemax+30)).values.reshape(-1)
	b=wg_d.wsgsmax.isel(lon=lonmax,lat=latmax,time=np.arange(timemax-30,timemax+30)).values.reshape(-1)
	ax.plot(mw_d.isel(time=np.arange(timemax-30,timemax+30)).time.values,a,
	 		# "-",
			marker="d",markersize=5,
			color="blue",label='Wind Speed',
			alpha=0.5)
	ax.plot(mw_d.isel(time=np.arange(timemax-30,timemax+30)).time.values,b,
	 		# "-",
			marker="d",markersize=5,
			color="darkgreen",label='Wind Gusts',
			alpha=0.5)
	ax.vlines(x=mw_d.isel(time=timemax).time.values,ymin=0,ymax=50,linestyles='dashed', color='pink', label="Max Wind Speed")
	# ax.vlines(x=mw_d.isel(time=timemax2).time.values,ymin=0,ymax=50,linestyles='dashed', color='orange', label="Max Wind gusts")
	ax.set_title(f"{model} 2 months of daily values of wind speed vs wind gusts centered aroung the max wind speed")
	ax.legend()
	plt.savefig(f"figures/ts_speed_vs_gusts_{model}_daily.png")
	plt.close()

	fig,axs=plt.subplots(1,1,
				figsize=(8,8),constrained_layout=True, squeeze=True)

	ax=axs
	# a=mw_d.mw.isel(lon=lonmax,lat=latmax,time=np.arange(timemax-300,timemax+300)).values.reshape(-1)
	# b=wg_d.wsgsmax.isel(lon=lonmax,lat=latmax,time=np.arange(timemax-300,timemax+300)).values.reshape(-1)
	# a=mw_d.mw.isel(lon=lonmax,lat=latmax).values.reshape(-1)
	# b=wg_d.wsgsmax.isel(lon=lonmax,lat=latmax).values.reshape(-1)
	#HOURLY
	e=np.tile(ele.orog.values.reshape(-1),100)
	a=mw.mw.isel(time=np.arange(2000,2100)).values.reshape(-1)
	b=wg.wsgsmax.isel(time=np.arange(2000,2100)).values.reshape(-1)

	df_winds=pd.DataFrame({"a":a,"b":b,"e":e})

	g=sns.scatterplot(df_winds,x="a",y="b",hue="e",palette="RdBu")
	# ax.scatter(a,b,
	#  		# "-",
	# 		marker="d",#markersize=5,
	# 		color="blue",label='',
	# 		alpha=0.5)
	# ax.plot([0, 1], [0, 1], transform=ax.transAxes)
	# ax.set_title(f"{model} scatter of speed vs gusts")
	# ax.legend()
	# ax.set_xlabel("Wind speed [m/s]")
	# ax.set_ylabel("Wind gusts [m/s]")
	g.set_title(f"{model} scatter of speed vs gusts")
	g.legend.set_title("Elevation [m]")
	g.set_axis_labels("Wind speed [m/s]", "Wind gusts [m/s]")
	g.set(xlim=(0,60),ylim=(0,60))
	g.axline((0, 0), slope=1, c=".2", ls="--", zorder=0)
	plt.savefig(f"figures/scatter_speed_vs_gusts_{model}_hourly_all_area_2000.png")
	plt.close()



	fig,axs=plt.subplots(1,1,
				figsize=(10,10),constrained_layout=True, squeeze=True)

	ax=axs
	# ax=axs.flatten()


	a=np.sort(mw_d.mw.isel(lon=lonmax,lat=latmax).values.reshape(-1))
	b=np.sort(wg_d.wsgsmax.isel(lon=lonmax,lat=latmax).values.reshape(-1))

	ax.plot(a,
			(np.arange(0,a.shape[0])+1)/a.shape[0],
			marker="d",markersize=5,
			color="blue",label='Wind Speed',
			alpha=0.5)
	ax.plot(b,
			(np.arange(0,b.shape[0])+1)/b.shape[0],
			marker="d",markersize=5,
			color="darkgreen",label='Wind Gusts',
			alpha=0.5)
	ax.set_title(f"{model} ECDFs of longitude:{mw_d.mw.isel(lon=lonmax,lat=latmax).lon.values.item():.2f} & latitude:{mw_d.mw.isel(lon=lonmax,lat=latmax).lat.values.item():.2f}")
	ax.legend()
	plt.savefig(f"figures/ecdf_speed_vs_gusts_{model}_daily.png")
	plt.close()


	heavy_mw_d=mw_d.quantile(0.999,dim='time') * sea_mask.sftlf
	heavy_wg_d=wg_d.quantile(0.999,dim='time').drop_vars("time_bnds") * sea_mask.sftlf.values

	heavy_mw=mw.quantile(0.999,dim='time') * sea_mask.sftlf
	heavy_wg=wg.quantile(0.999,dim='time').drop_vars("time_bnds") * sea_mask.sftlf

	#PLOT hourlay adn daily heavy SPEED
	plot_panel_rotated(
			figsize=(14,6),
			nrow=1,ncol=2,
			list_to_plot=[heavy_mw.mw,heavy_mw_d.mw],
			name_fig=f"WindSpeed_{model}",
			list_titles=["Heavy Speeds Hourly","Heavy Speeds Daily"],
			levels=[np.arange(1,21,2),np.arange(1,21,2)],
			suptitle=f"{model}",
			name_metric=["[m/s]","[m/s]"],
			SET_EXTENT=False,
			cmap=["RdYlGn","RdYlGn"]
	)
	#PLOT hourlay adn daily heavy GUSTS
	plot_panel_rotated(
			figsize=(14,6),
			nrow=1,ncol=2,
			list_to_plot=[heavy_wg.wsgsmax,heavy_wg_d.wsgsmax],
			name_fig=f"WindGust_{model}",
			list_titles=["Heavy Gusts Hourly","Heavy Gusts Daily"],
			levels=[np.arange(15,60,5),np.arange(15,60,5)],
			suptitle=f"{model}",
			name_metric=["[m/s]","[m/s]"],
			SET_EXTENT=False,
			cmap=["RdYlGn","RdYlGn"]
	)

	# plot_panel(
	# 		nrow=1,ncol=1,
	# 		list_to_plot=[heavy_mw.mw],
	# 		name_fig=f"WindSpeedCB_{model}",
	# 		list_titles=["Heavy Speeds"],
	# 		levels=[np.arange(0,12,2)],
	# 		suptitle=f"{model}",
	# 		name_metric=["[m/s]"],
	# 		SET_EXTENT=False,
	# 		cmap="RdYlGn"
	# )



	fig,axs=plt.subplots(3,3,
				figsize=(12,12),constrained_layout=True, squeeze=True)
	ax=axs.flatten()

	for idx in range(9):

		i,j=np.random.randint(0,272),np.random.randint(0,158)
		a=np.sort(mw.mw.isel(lon=i,lat=j).values.reshape(-1))
		b=np.sort(wg.wsgsmax.isel(lon=i,lat=j).values.reshape(-1))
		ax[idx].plot(a,
				(np.arange(0,a.shape[0])+1)/a.shape[0],
				marker="d",markersize=5,
				color="blue",label='Wind Speed',
				alpha=0.5)
		ax[idx].plot(b,
				(np.arange(0,b.shape[0])+1)/b.shape[0],
				marker="d",markersize=5,
				color="darkgreen",label='Wind Gusts',
				alpha=0.5)
		ax[idx].set_title(f"{model} ECDFs of longitude:{mw.lon[i].item():.2f} & latitude:{mw.lon[j].item():.2f}")
		ax[idx].legend()
	plt.savefig(f"figures/ecdf_speed_vs_gusts_{model}.png")
	plt.close()

	# fig,axs=plt.subplots(1,1,
	#             figsize=(12,12),constrained_layout=True, squeeze=True)
	# # ax=axs.flatten()

	# a=np.sort(mw.mw.isel(time=np.arange(24*2)).values.reshape(-1))
	# b=np.sort(wg.wsgsmax.isel(time=np.arange(24*2)).values.reshape(-1))

	# axs.plot(a,
	#         (np.arange(0,a.shape[0])+1)/a.shape[0],
	#         marker="d",markersize=5,
	#         color="blue",label='Wind Speed',
	#         alpha=0.5)
	# axs.plot(b,
	#         (np.arange(0,b.shape[0])+1)/b.shape[0],
	#         marker="d",markersize=5,
	#         color="darkgreen",label='Wind Gusts',
	#         alpha=0.5)
	# axs.set_title(f"ECDFs of year 2000")
	# axs.legend()
	# plt.savefig(f"figures/ecdf_speed_vs_gusts_all_area.png")
	# plt.close()

	"""
	HISTOGRAM
	"""

	a=mw_d.mw.values.reshape(-1)
	b=wg_d.wsgsmax.values.reshape(-1)
	fig,axs=plt.subplots(1,1,
			figsize=(8,8),constrained_layout=True, squeeze=True)
	# ax=axs.flatten()
	ax=axs
	
	ax.hist(a,
			bins=np.arange(0,60,2.5),
			color="blue",
			ec='white',
			density=True,
			label='Wind Speed',
			alpha=0.5)
	ax.hist(b,
			bins=np.arange(0,60,2.5),
			color="darkgreen",
			ec='white',
			density=True,
			label='Wind Gusts',
			alpha=0.5)
	ax.set_title(f"{model} whole area")
	ax.legend()

	# plt.xlim(1,25)
	plt.savefig(f"figures/hist_speed_vs_gusts_whole_area_{model}.png")
	plt.close()

	fig,axs=plt.subplots(3,3,
			figsize=(12,12),constrained_layout=True, squeeze=True)
	ax=axs.flatten()

	for idx in range(9):

		i,j=np.random.randint(0,272),np.random.randint(0,158)
		a=mw.mw.isel(lon=i,lat=j).values.reshape(-1)
		b=wg.wsgsmax.isel(lon=i,lat=j).values.reshape(-1)
		ax[idx].hist(a[a>0.1],
				bins=np.arange(0,20,0.1),
				color="blue",
				density=True,
				label='Wind Speed',
				alpha=0.5)
		ax[idx].hist(b,
				bins=np.arange(0,20,0.1),
				color="darkgreen",
				density=True,
				label='Wind Gusts',
				alpha=0.5)
		ax[idx].set_title(f"{model} at longitude:{mw.lon[i].item():.2f} & latitude:{mw.lon[j].item():.2f}")
		ax[idx].legend()
	plt.xlim(1,25)
	plt.savefig(f"figures/hist_speed_vs_gusts_{model}.png")
	plt.close()


	fig,axs=plt.subplots(1,1,
		figsize=(12,12),constrained_layout=True, squeeze=True)

	a=mw.mw.values.reshape(-1)
	b=wg.wsgsmax.values.reshape(-1)
	axs.hist(a[a>0.1],
			bins=np.arange(0,20,0.1),
			color="blue",
			density=True,
			label='Wind Speed',
			alpha=0.5)
	axs.hist(b[b>0.1],
			bins=np.arange(0,20,0.1),
			color="darkgreen",
			density=True,
			label='Wind Gusts',
			alpha=0.5)
	axs.set_title(f"{model} at longitude:{mw.lon[i].item():.2f} & latitude:{mw.lon[j].item():.2f}")
	axs.legend()
	plt.xlim(1,25)
	plt.ylim(0,0.6)
	plt.savefig(f"figures/hist_speed_vs_gusts_all_area_{model}.png")
	plt.close()