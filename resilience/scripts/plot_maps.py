#! /home/lcesarini/miniconda3/envs/colorbar/bin/python

import os
import numpy as np 
import xarray as xr 
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature

def compute_quantile(ds):
    return ds.pr.quantile(q=0.99,dim='time')

shp_triveneto = gpd.read_file("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

os.chdir("/mnt/data/RESTRICTED/CARIPARO/common/ECMWF-ERAINT/")
eth   = xr.open_dataset("ETH/CPM/pr/ETH_ECMWF-ERAINT_200001010030_200012312330.nc").load()
mohc  = xr.open_mfdataset("MOHC/CPM/pr/MOHC_ECMWF-ERAINT_2000*.nc").load()
hclim = xr.open_dataset("HCLIMcom/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_200001010030-200012312330_regrid.nc").load() 
ictp  = xr.open_dataset("ICTP/CPM/pr/ICTP_ECMWF-ERAINT_20000101000000-20010101000000.nc").load()
cnrm  = xr.open_dataset("CNRM/CPM/pr/CNRM_ECMWF-ERAINT_200001010030-200012312330.nc").load()

list_q = [compute_quantile(model) for model in [mohc,hclim,eth,ictp,cnrm]]

ens_mean = (list_q[0] + list_q[1] + list_q[2] + list_q[3] + list_q[4] ) / 5

cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        .with_extremes(over='0', under='0.75'))

cmap=mpl.cm.viridis

bounds = np.linspace(1,10,5)

norm = mpl.colors.BoundaryNorm(bounds, bounds.shape[0]+1)
titles=["MOHC","HCLIM","ETH","ICTP","CNRM","ENSEMBLE"]

fig,axs=plt.subplots(2,3,
                    figsize=(24,16),
                    subplot_kw={"projection":ccrs.PlateCarree()})

ax=axs.flatten()

for i,model in enumerate([mohc,hclim,eth,ictp,cnrm,ens_mean]):
    if i==1:
        pcm=(model.pr.quantile(q=0.99,dim='time') * 3600).plot.\
            pcolormesh(ax=ax[i],cbar_kwargs={"shrink":0.25},
            cmap=cmap,norm=norm,)
    elif i == 5:
        pcm=model.plot.pcolormesh(ax=ax[i],cbar_kwargs={"shrink":0.25},cmap=cmap,norm=norm)

    else:
        pcm=model.pr.quantile(q=0.99,dim='time').plot.pcolormesh(ax=ax[i],cbar_kwargs={"shrink":0.25},cmap=cmap,norm=norm)
    shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red')
    ax[i].add_feature(cfeature.BORDERS)
    ax[i].coastlines()
    ax[i].set_extent([10.2,13.15,44.6,47.15])
    ax[i].set_title(titles[i])

# fig.colorbar(pcm,ax=axs[1,:])
plt.savefig("/home/lcesarini/2022_resilience/figures/panel_env.png")
plt.close()
