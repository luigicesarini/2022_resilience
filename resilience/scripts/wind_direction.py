#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import rioxarray
import subprocess
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
from windrose import WindroseAxes 
from shapely.geometry import mapping
from cartopy import feature as cfeature

from resilience.utils import *

PATH_WIND="/mnt/data/lcesarini/wind"
# meta_station=pd.read_csv("data/meta_stazioni_vento_4326.csv",sep=",")

meta_station=pd.read_csv(f"{PATH_WIND}/meta_stazioni_vento_4326.csv")
speed_10=glob("/mnt/data/lcesarini/wind/10m/*")
name_sta=[pd.read_csv(speed_10[i],encoding="utf-8").STAZIONE[0] for i in np.arange(22)]

vel_10=glob("/mnt/data/lcesarini/wind/10m/*Vel*")
dir_10=glob("/mnt/data/lcesarini/wind/10m/*dir*")


# [print(x,y) for x,y in zip(vel_10,dir_10)]
# for nc in name_sta:
#     if np.any([nc in x for x in vel_10]) and np.any([nc in x for x in dir_10]):
#         print("OK")
#     else:
#         print(f"{nc} non presente")

        

mw_sph=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/*.nc").load()
dir_sph=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/wind_dir/*.nc").load()

dir_sph=dir_sph.rename({"u10":"winddir",'longitude':'lon','latitude':'lat'})

list_mdl=["CMCC","KIT","CNRM","ICTP","ETH","KNMI"] #"CNRM","ICTP","ETH","KNMI",manca winddir in "HCLIMcom"
# mdl='HCLIMcom'
# for mdl in tqdm(list_mdl,total=len(list_mdl)):
#     PATH_U=f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/uas/"
#     PATH_V=f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/vas/"
#     PATH_TMP="/mnt/data/lcesarini/tmp/"
#     PATH_OUT=f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/wind_dir"

#     if not os.path.exists(PATH_OUT): os.makedirs(PATH_OUT)

#     list_uas=glob(f"{PATH_U}/*")
#     list_vas=glob(f"{PATH_V}/*")


#     for i in range(len(list_uas)):
#         base_out_dir=os.path.basename(list_uas[i]).replace("uas","dir")
#         base_out_o=os.path.basename(list_uas[i]).replace("uas","o")
#         subprocess.run(f"cdo atan2 {list_uas[i]} {list_vas[i]} {PATH_TMP}{base_out_dir}",shell=True)
#         subprocess.run(f"cdo -divc,3.14159265359 {PATH_TMP}{base_out_dir} {PATH_TMP}{base_out_o}",shell=True)
#         subprocess.run(f"cdo mulc,180. {PATH_TMP}{base_out_o} {PATH_TMP}{base_out_dir}",shell=True)
#         subprocess.run(f"cdo addc,180. {PATH_TMP}{base_out_dir} {PATH_TMP}{base_out_o}",shell=True)
#         subprocess.run(f"cdo setvar,winddir {PATH_TMP}{base_out_o} {PATH_OUT}/{base_out_dir}",shell=True)

#     subprocess.run(f"rm –f {PATH_TMP}{base_out_o}",shell=True)


# for mdl in tqdm(list_mdl,total=len(list_mdl)):
#      for name in glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/wind_dir/*"):
#          subprocess.run(f'mv {name} {name.replace("wind_dird","d")}',shell=True)
    

# for mdl in tqdm(list_mdl,total=len(list_mdl)):

#     list_dir=glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/wind_dir/*")

#     mw=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*").load()
#     uas=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/uas/*").load()
#     vas=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/vas/*").load()
#     dir=xr.open_mfdataset(list_dir).load()


    # mw.time
    # (uas.time.values==vas.time.values).all()
    # (uas.time.values==mw.time.values).all()
    # (vas.time.values==mw.time.values).all()
    # wind= xr.merge([uas.drop_vars("height"),vas.drop_vars("height"),mw])


    # m=mw.isel(time=slice(1,10))
    # u=uas.isel(time=slice(1,10))
    # v=vas.isel(time=slice(1,10))

    # combined=xr.merge([u.drop_vars("height"),v.drop_vars("height"),m])

    # stride = 6

    # xx=get_palettes()

    # ax=plt.axes(projection=ccrs.PlateCarree())
    # combined.isel(time=1).mw.plot(ax=ax,cmap="Greens",levels=10,robust=True)
    # combined.isel(lon=np.arange(0,combined.lon.shape[0],stride),
    #             lat=np.arange(0,combined.lat.shape[0],stride),
    #             time=1).plot.quiver(x="lon",y="lat",u="uas",v="vas",
    #                                 robust=True,
    #                                 ax=ax)
    # plt.show()
    # plt.savefig("/mnt/ssd/lcesarini/ss.png")
    # plt.close()

ll_mw=[]
ll_dir=[]
for mdl in tqdm(list_mdl,total=len(list_mdl)):

    list_dir=glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/wind_dir/*")

    # mw=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*").load()
    # dir=xr.open_mfdataset(list_dir).load()
    # uas=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/uas/*").load()
    # vas=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/vas/*").load()
    mw=xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*")
    dir=xr.open_mfdataset(list_dir)
    
    for idx,nc in enumerate(name_sta):

        if nc=="Adria - Bellombra":
            continue

        m_1=mw.sel(lon=meta_station[meta_station.Nome == nc].lon.item(),
                lat=meta_station[meta_station.Nome == nc].lat.item(),method='nearest').mw.values.reshape(-1,)
        d_1=dir.sel(lon=meta_station[meta_station.Nome == nc].lon.item(),
                    lat=meta_station[meta_station.Nome == nc].lat.item(),method='nearest').winddir.values.reshape(-1,)


        np.save(f"/mnt/data/lcesarini/wind/10m/arrays/arr_mw_{nc}_{mdl}.npy",m_1)
        np.save(f"/mnt/data/lcesarini/wind/10m/arrays/arr_dir_{nc}_{mdl}.npy",d_1)
    
        # ll_mw.append(m_1)
        # ll_dir.append(d_1)

for idx,nc in enumerate(name_sta):

    if nc=="Adria - Bellombra":
        continue

    ens_mw=np.array([np.load(ff) for ff in glob(f"/mnt/data/lcesarini/wind/10m/arrays/arr_mw_{nc}_*.npy")]).mean(axis=0)
    ens_dir=np.array([np.load(ff) for ff in glob(f"/mnt/data/lcesarini/wind/10m/arrays/arr_dir_{nc}_*.npy")]).mean(axis=0)
   
    m_1_sph=mw_sph.sel(lon=meta_station[meta_station.Nome == nc].lon.item(),
                lat=meta_station[meta_station.Nome == nc].lat.item(),method='nearest').mw.values.reshape(-1,)
    d_1_sph=dir_sph.sel(lon=meta_station[meta_station.Nome == nc].lon.item(),
                        lat=meta_station[meta_station.Nome == nc].lat.item(),method='nearest').winddir.values.reshape(-1,)


    x,y=np.histogram(d_1_sph);
    plt.hist(d_1_sph)


        # d_1=dir.winddir.values.reshape(-1,)
        # m_1=mw.mw.values.reshape(-1,)

        # d_1=d_1[~np.isnan(d_1)]
        # m_1=m_1[~np.isnan(m_1)]
    """
    CPM
    """
    ax1 = WindroseAxes.from_ax()
    ax1.bar(ens_dir, ens_mw, nsector=16, 
        #    bins=np.arange(15,25,2.5), 
        opening=0.6, edgecolor='white', normed=True)
    ax1.set_legend()
    ax1.set_title(f"{nc} CPM")
    # plt.show()
    plt.savefig(f"{PATH_WIND}/figures/rosa_ensemble_{nc}.png")

    """
    SPHERA
    """
    ax1 = WindroseAxes.from_ax()
    ax1.bar(d_1_sph, m_1_sph, nsector=16, 
        #    bins=np.arange(15,25,2.5), 
        opening=0.6, edgecolor='white', normed=True)
    ax1.set_legend()
    ax1.set_title(f"{nc} SPHERA")
    # plt.show()
    plt.savefig(f"{PATH_WIND}/figures/rosa_SPHERA_{nc}.png")

    # print(np.all(np.isfinite(d_1)))
    # print(np.all(np.isfinite(m_1)))

    # print(np.any(~np.isnan(d_1)))
    # print(np.any(~np.isnan(m_1)))

    """STATION"""

    vel_sta=pd.read_csv(np.array(vel_10)[np.array([nc in x for x in vel_10])].item(),encoding="utf-8")
    dir_sta=pd.read_csv(np.array(dir_10)[np.array([nc in x for x in dir_10])].item(),encoding="utf-8")

    # date_sequence_period = pd.date_range(start="2000-03-01" ,end='2022-01-01', freq='H')

    # dates_vel=np.array([pd.to_datetime(f"{vel_sta.DATA.values[_]} {vel_sta.ORA.values[_]}", format="%Y-%m-%d %H:%M") for _ in range(vel_sta.ORA.values.shape[0])])
    # dates_dir=np.array([pd.to_datetime(f"{dir_sta.DATA.values[_]} {dir_sta.ORA.values[_]}", format="%Y-%m-%d %H:%M") for _ in range(dir_sta.ORA.values.shape[0])])
    

    # x=np.isin(date_sequence_period,dates_vel)
    # missing_vel=date_sequence_period[~np.isin(date_sequence_period,dates_vel)]
    # missing_dir=date_sequence_period[~np.isin(date_sequence_period,dates_dir)]

    df_merge=pd.merge(vel_sta,dir_sta,on=["STAZIONE","DATA","ORA"],how='left').dropna()

    ax2 = WindroseAxes.from_ax()
    ax2.bar(np.array(df_merge.VALORE_y), np.array(df_merge.VALORE_x), nsector=16, 
            # bins=np.arange(5,25,5), 
            opening=0.6, edgecolor='white', normed=True)
    ax2.set_legend()
    ax2.set_title(f"{nc} STAZIONE")
    # plt.show()
    plt.savefig(f"{PATH_WIND}/figures/rosa_stations_{nc}.png")





    # ds.plot.quiver(x="x", y="y", u="A", v="B", col="w", row="z", scale=4)

