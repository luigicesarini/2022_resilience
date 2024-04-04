#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
from resilience.utils import *
import rasterio
import argparse
import subprocess
# import rioxarray
import numpy as np 
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy import stats
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
from rasterio.mask import mask
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/lcesarini/2022_resilience/")
from resilience.utils import *

def crop_to_extent(xr,xmin=10.38,xmax=13.1,ymin=44.7,ymax=47.1):
    """
    Functions that select inside a given extent

    Parameters
    ----------
    xr : xarrayDataset, 
        xarray dataset to crop
    
    xmin,xmax,ymin,ymax: coordinates of the extent desired.    
    
    Returns
    -------
    
    cropped_ds: xarray dataset croppped

    Examples
    --------
    """
    xr_crop=xr.where((xr.lon > xmin) & (xr.lon < xmax) &\
                     (xr.lat > ymin) & (xr.lat < ymax), 
                     drop=True)

    return xr_crop

list_mdl=["ETH","CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT"] #
# for mdl in list_mdl:
#     sb.run(f"test -d /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw",shell=True)



if __name__=="__main__":

    for MODEL in tqdm(list_mdl,total=len(list_mdl)):

        if MODEL == "SPHERA":
            path_uas = f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/{MODEL}/uas/"
            path_vas = f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/{MODEL}/vas/"
        else:
            path_uas = f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{MODEL}/CPM/uas/"
            path_vas = f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{MODEL}/CPM/vas/"

        # uas=xr.open_mfdataset(glob(path_uas+"*")[0]) 
        # print(uas.time[0:2].values)

        for year in tqdm(np.arange(2000,2010)):
            

            if not os.path.exists(path_uas[:-4]+"mw"): os.makedirs(path_uas[:-4]+"mw")
            path_mw = path_uas[:-4]+"mw"

            if MODEL == "SPHERA":
                vas = xr.open_mfdataset(f"{path_vas}*{year}*")
                uas = xr.open_mfdataset(f"{path_uas}*{year}*")
                uas=uas.rename({"longitude":"lon","latitude":"lat","u10":"uas"})
                vas=vas.rename({"longitude":"lon","latitude":"lat","v10":"vas"})
            else:
                vas = xr.open_dataset(glob(f"{path_vas}*{year}*")[0])
                uas = xr.open_dataset(glob(f"{path_uas}*{year}*")[0])
            
            if MODEL == "SPHERA":
                name_file = f"mw_{year}_SPHERA.nc"
            else:
                name_file = f"mw{os.path.basename(glob(f'{path_uas}*{year}*')[0])[3:]}"
            
            # cropped_uas = crop_to_extent(uas)
            # cropped_vas = crop_to_extent(vas)

            mw = np.sqrt(np.power(uas.uas.values,2) + np.power(vas.vas.values,2))
            
            mw_d = xr.DataArray(mw, 
                        coords={'time':uas.time,
                                'lon': uas.lon, 
                                'lat': uas.lat
                                },
                        dims={'time':uas.time.shape[0],'lat':uas.lat.shape[0],'lon':uas.lon.shape[0]},
                        attrs = uas.attrs
            ) 
                        # dict(description = "mw stands for 'Module Wind",
                        #             unit = '[m*s-1]'))

            mw_ds=mw_d.to_dataset(name = 'mw', promote_attrs = True)        
        
            mw_ds.to_netcdf(f'{path_mw}/{name_file}',
                            encoding = {"mw": {"dtype": "float32"}})


        # plot_panel(
        #     nrow=1,ncol=1,
        #     list_to_plot=[uas_eth.uas.isel(time=12)],
        #     name_fig="mw_test",
        #     list_titles=["uas ETH"],
        #     levels=[8],
        #     suptitle="",
        #     name_metric="[mm/s]",
        #     SET_EXTENT=False,
        #     cmap="RdYlGn"
        # )



    # mw_ds=xr.open_mfdataset(f'/mnt/data/lcesarini/tmp/mw_*').load()

    # q_w=mw_ds.quantile(q=[0.95,0.99,0.999],dim='time')

    # from scripts.utils import *
    # plot_panel(
    #     1,3,
    #     [q_w.isel(quantile=i).mw for i in range(0,3)],
    #     "q_wind","3 yrs HCLIMcom",suptitle='Wiiinds',name_metric="[m * s^-1]",
    #     cmap="RdYlGn"
    #     )
    # proj = ccrs.PlateCarree()
    # if hasattr(mw_ds,"Lambert_Conformal"):
    #     rot = ccrs.LambertConformal(central_longitude=16, central_latitude=45.5, 
    #                                 false_easting=1349205.5349238443, false_northing=732542.657192843)
    # else:
    #     rot = ccrs.RotatedPole(pole_longitude=-170.0, 
    #                         pole_latitude=43.0, 
    #                         central_rotated_longitude=0.0, 
    #                         globe=None)
    

    # fig,ax = plt.subplots(nrows=1,
    #                   ncols=3,#int(len(list_to_plot) / 2),
    #                   figsize=(24,10),constrained_layout=True, squeeze=True,
    #                   subplot_kw={"projection":ccrs.PlateCarree()}
    #                   )

    # ax=ax.flatten()

    # cmap = plt.cm.rainbow

    # bounds = np.linspace(0.1,30,10)#np.array([18,19,20,21,22,23,24])
    
    # norm = mpl.colors.BoundaryNorm(bounds.round(2), bounds.shape[0]+1, extend='both')

    # merged=xr.merge([cropped_uas,cropped_vas])

    # resample = merged.isel(time=12,x=slice(None, None, 4),
    #                           y=slice(None, None, 4))

    # vars_name=["uas","vas","mw"]

    # for i,wind in enumerate([cropped_uas,cropped_vas,mw_ds]):
    #     pcm=(wind[vars_name[i]]).max(dim='time').plot.contourf(ax=ax[i],alpha=1,
    #                 transform=rot,
    #                 add_colorbar=True,
    #                 cmap=cmap, norm=norm,
    #                 cbar_kwargs={"shrink":0.7,
    #                             "orientation":"horizontal",
    #                             "label":"Woops (m/s)"
    #                             }
    #                 )
    #     if i == 3:            
    #         quiver = resample.plot.quiver(x='x', y='y', u='uas', v='vas', ax=ax[i],
    #                         transform=rot, 
    #                         scale=80
    #                         )
            
    #     ax[i].coastlines()
    #     gl = ax[i].gridlines(
    #         draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',xlocs=[11],ylocs=[45,47]
    #     )
    #     ax[i].add_feature(cfeature.BORDERS, linestyle='--')
    #     ax[i].set_title(f"{vars_name[i]}")


    # # fig,ax = plt.subplots(nrows=1,
    # #                 ncols=3,#int(len(list_to_plot) / 2),
    # #                 figsize=(24,10),constrained_layout=True, squeeze=True,
    # #                 subplot_kw={"projection":ccrs.PlateCarree()}
    # #                 )

    # # for i,j in enumerate([70,80,90]):
    # #     quiver = resample.plot.streamplot(x='x', y='y', u='uas', v='vas', ax=ax[i],
    # #                             transform=rot, 
    # #                             # scale=j
    # #                             )
    # plt.savefig(f"figures/test_wind.png")
    # plt.close()



