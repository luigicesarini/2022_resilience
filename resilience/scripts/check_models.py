#! /home/lcesarini/miniconda3/envs/detectron/bin/python

import os
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

from utils import *

import warnings
warnings.filterwarnings('ignore')
path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"

os.chdir("/home/lcesarini/2022_resilience/")


"""
What I have:
- Metadata of each station
- file with name code of each station containing the observed precipitation records
  .
- Gridded data of modeled precipitation for the periods:
    * historical 1996-2005: "CPM_ETH_MPI_historical_Italy_1996-2005_pr_hour.nc"
    * historical 2000-2009: "CPM_ETH_Italy_2000-2009_pr_hour.nc"
    * rcp8.5     2041-2050: "CPM_ETH_MPI_rcp85_Italy_2041-2050_pr_hour.nc"
    * rcp8.5     2090-2099: "CPM_ETH_MPI_rcp85_Italy_2090-2099_pr_hour.nc"
    * rcp8.5     2092-2101: "CPM_ETH_ECMWF-ERAINT_PGW-MPI-rcp85_Italy_2092-2101_pr_hour.nc"

Steps of things to do:
    pr_m: modeled precipitation, additional suffix refers to period
    pr_o: observed precipitation, additional suffix may refer to feature of station (e.g., region)

# For bias correction (independently from methods)

EXTRACTION OF DATA
  for i-th station:
    - Get vector of tp_o 
    - Get vector of tp_m for i-th station
TRANSFORMATION OF DATA
    - Which exploratory analysis to apply?
    - Which BC method to apply?
LOAD OF RESULTS
    - What output do we want?
    still ealry to decide I guess

"""


if __name__=="__main__":
    for model in ["CPM_ETH_Italy_2000-2009"]:
        for nome in ["AA_0220"]:
            # nome = "VE_0059" #AA_0220
            prec_o, date = get_observed(name_station=nome)

            ds=xr.open_dataset(path_model+f"/{model}_pr_hour.nc")
            # ds_1=xr.open_dataset(path_model+ "/CPM_ETH_Italy_2000-2009_pr_hour.nc")

            min_over_date, max_over_date = get_overlapping_dates(date,ds.time)

            print(findNA(date,min_over_date,max_over_date))

            proj = ccrs.PlateCarree()

            rot = ccrs.RotatedPole(pole_longitude=-170.0, 
                                pole_latitude=43.0, 
                                central_rotated_longitude=0.0, 
                                globe=None)

            prec_m=get_mod_at_obs(ds,name_station=nome,rotated_cartopy=rot,proj_cartopy=proj).pr.values
            index_overlap=np.where((date >= min_over_date) & (date <= max_over_date))[0]


            np.where(date == np.datetime64('1996-01-01T00:00:00'))

            print(f"Difference between shape of model and overlap {prec_m.shape[0] - index_overlap.shape[0]}")
            if (prec_m.shape[0] - index_overlap.shape[0]) == 1:
                prec_o   = prec_o[np.concatenate([np.array([index_overlap[0]-1]) ,index_overlap],axis=0)]
                date_ref = date[np.concatenate([np.array([index_overlap[0]-1]) ,index_overlap],axis=0)]
            elif (prec_m.shape[0] - index_overlap.shape[0]) == 2:
                prec_o   = prec_o[np.concatenate([np.array([index_overlap[0]-1]),
                                                index_overlap,
                                                np.array([index_overlap[-1]+1])],axis=0)]
                date_ref = date[np.concatenate([np.array([index_overlap[0]-1]),
                                                index_overlap,
                                                np.array([index_overlap[-1]+1])],axis=0)]
            else: 
                print("Problem with overlap check dates of overlapping period")

            
            if prec_o.shape[0] == prec_m.shape[0] == date_ref.shape[0]:
                pass
                #print(f"shapes of modelled and observations are equal {prec_o.shape[0]},{prec_m.shape[0]},{date_ref.shape[0]}")
            else:
                print(f"shapes of modelled and observations differ {prec_o.shape[0]},{prec_m.shape[0]},{date_ref.shape[0]}")
                

            error = (prec_m[:]-prec_o[:,0]) 
            print(f"""
                RUN:{model}
                MAE:{mean_absolute_error(prec_m[:],prec_o[:,0]):.2f}
                MSE:{mean_squared_error(prec_m[:],prec_o[:,0]):.2f}
                R2 :{r2_score(prec_m[:],prec_o[:,0]):.2f}
            """)
            idx = np.where(np.abs(error) == np.abs(error).max())


            rot_lon, rot_lat = rot.transform_point(12.41013,46.46165,proj)

            ds.sel(time=date_ref[idx],rlon=rot_lon,rlat=rot_lat,method='nearest').pr.item()
            # ds_1.sel(time=date_ref[idx],rlon=rot_lon,rlat=rot_lat,method='nearest').pr.item()
            # print(ds.sel(time=date_ref[idx],rlon=rot_lon,rlat=rot_lat , method='nearest').pr)

            PLOT_STATION=True
            if PLOT_STATION:
                meta_station = pd.read_csv("meta_station_updated_col.csv")

                gds=gpd.GeoDataFrame(meta_station,geometry=gpd.points_from_xy(meta_station[["lon"]],
                                        meta_station["lat"], 
                                        crs="EPSG:4326"))
                plt.figure(figsize=(16,12))
                ax = plt.axes(projection=proj)
                
                xr_plot=ds.sel(time=date_ref[idx],method='nearest').isel(time=0).pr
                
                xr.where(xr_plot > 0, 1000, np.nan).plot.pcolormesh(
                                                    ax=ax, transform=rot, x="rlon", y="rlat", 
                                                    add_colorbar=False, cmap= "Reds",alpha=0.5
                                                )
                xr.where(xr_plot > 1, 1000, np.nan).plot.pcolormesh(
                                            ax=ax, transform=rot, x="rlon", y="rlat", 
                                            add_colorbar=False, cmap= "Blues",alpha=0.5
                                        )
                ds.isel(time=np.where(ds.time==xr_plot.time)[0].item()).pr.plot.pcolormesh(
                                                    ax=ax, transform=rot, x="rlon", y="rlat", 
                                                    add_colorbar=False, cmap= "Greys", alpha=0.5
                                                )
                ax.coastlines()
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle='--')
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                # ax.add_feature(cfeature.STATES)
                ax.add_feature(cfeature.RIVERS)
                ax.set_xlim([10, 13])
                ax.set_ylim([44.5,47.2])
                gds[gds.name==nome].plot(ax=ax, column = "max_tp",markersize=370,marker='*', color="green")
                plt.annotate(text=gds[gds.name==nome].name.item(),
                            xy=(gds[gds.name==nome].lon.item(),gds[gds.name==nome].lat.item()),
                            size =50)
                gl = ax.gridlines(
                    draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--'
                )
                plt.title(f"Time: {ds.isel(time=np.where(ds.time==xr_plot.time)[0].item()).time.values}", 
                        fontsize=25)
                plt.suptitle(f"Station: {nome}", fontsize=25)
                plt.savefig(f"test_grid_{model}.png")



            PLOT=True
            if PLOT:
                plt.figure(figsize=(9,6))
                plt.plot(date_ref,prec_o,'or', label='Observed')
                plt.plot(date_ref,prec_m,'og', label='Modelled')
                plt.plot(date_ref,(prec_m[:]-prec_o[:,0]) ,'-b', label='Error') #/ prec_o[:,0]
                plt.ylabel("Precipitation [mm]")
                plt.legend()
                plt.xlim(('2001-08-29T09:00:00','2005-08-29T14:00:00'))
                plt.xticks(rotation=30)
                plt.savefig(f"timeseries_{model}.png")
                plt.close()

            # ds.sel(time=slice(min_over_date, max_over_date))


            # i_lon = 0
            # i_lat = 0

            # print(f"coords: {ds.lon[i_lon,i_lat].item()},{ds.lat[i_lon,i_lat].item()}")
            # print(f"coords: {ds.lon[i_lon,i_lon].item()},{ds.lat[i_lat,i_lat].item()}")
            #print(convert_coord(xy=(-2.5999999046325684,-3.380000114440918), pole=(-170,43), to_rotated=False))
            # lon,lat=convert_coord(xy=(ds.rlon[i_lon].item(),ds.rlat[i_lat].item()), pole=(-170,43), to_rotated=False)
            # print(lon,lat)


            # df = pd.read_csv("meta_station_updated_col.csv")
            # print(df.head())
            # print(rot.transform_point(df.lon[1],df.lat[1],proj))
            
            # plot_slice_model(ds,gds,proj,rot)