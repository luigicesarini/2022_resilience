#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
"""
0. Copy the original file adding the extension grb2
1. Remap the original
2. Decumulate the precipitation 
3. Save to the common folder

"""
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import argparse
import subprocess
# import rioxarray
import numpy as np 
import xarray as xr 
import xarray.ufuncs as xu
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

PATH_GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"
PATH_SPHERA="/mnt/data/RESTRICTED/SPHERA"
PATH_SPHERA_OUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr"

os.chdir("/home/lcesarini/2022_resilience")

from resilience.utils import *
from cfgrib.xarray_to_grib import to_grib

NAME_VAR="tp"
SHORT_VAR="t"

ls_files=pd.read_csv(f"/mnt/data/lcesarini/SPHERA/{NAME_VAR}/new_list_sphera_{NAME_VAR}.txt",header=None)

# Define multiple patterns using the | (or) operator
patterns = [
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2000**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2001**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2002**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2003**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2004**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2005**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2006**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2007**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2008**zoom_remapped.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2009**zoom_remapped.grb2*', 
            ]
patterns_original = [
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2000**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2001**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2002**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2003**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2004**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2005**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2006**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2007**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2008**zoom.grb2*', 
    f'/mnt/data/lcesarini/SPHERA/{NAME_VAR}/*2009**zoom.grb2*', 
            ]

# Use glob to find files matching any of the specified patterns
matching_files = []
for pattern in patterns_original:
    matching_files.extend(glob(pattern))


np.array(matching_files)[['idx' not in xx for xx in matching_files]]


for file in tqdm(ls_files.iterrows()):
    """
    STEP 0
    """
    print(file[1][0])
    command=f"grib_copy -w shortName={SHORT_VAR} {PATH_SPHERA}/{file[1][0]} /mnt/data/lcesarini/SPHERA/{NAME_VAR}/{file[1][0]}.grb2"
    subprocess.run(command,shell=True)

    # for file in glob("/mnt/data/lcesarini/SPHERA/2t/*"):
    #     subprocess.run(f"grib_copy -w shortName={SHORT_VAR} {file} {file}.grb2", shell=True)
    for file in glob("/mnt/data/lcesarini/SPHERA/2t/*.grb2"):
        subprocess.run(f"cdo remapycon,{PATH_GRID} {file} {file.replace('zoom','remapped')}",shell=True)
        subprocess.run(f"grib_to_netcdf -o {file.replace('zoom','remapped').replace('grb2','nc')} {file.replace('zoom','remapped')}",shell=True)

    for file in glob("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/vas/*.grb2"):
        subprocess.run(f"grib_to_netcdf -o {file.replace('zoom','remapped').replace('grb2','nc')} {file}",shell=True)
    """
    STEP 1
    """
    subprocess.run(f"cdo remapycon,{PATH_GRID} /mnt/data/lcesarini/SPHERA/tp/{file[1][0]}.grb2 /mnt/data/lcesarini/SPHERA/tp/{file[1][0]}_remapped.grb2",shell=True)

    
    """
    STEP 2
    """

    ll_files=np.array(matching_files)[[('idx' not in xx) and ('2020' not in xx) for xx in matching_files]]
    # ds_cfgrib=xr.open_dataset(f'/mnt/data/lcesarini/SPHERA/tp/{file[1][0]}_remapped.grb2',engine="cfgrib")
    for FILE in tqdm(ll_files,total=ll_files.shape[0]):
        try:
            FILE="/mnt/data/lcesarini/SPHERA/tp/200912_ten3_Tdeep_tpH_zoom.grb2"
            ds_cfgrib=[xr.open_mfdataset(ll,engine="cfgrib") for ll in ll_files]
            ds_cfgrib=xr.open_mfdataset(ll_files[7],engine="cfgrib")
            ds_cfgrib=xr.open_dataset(f'/mnt/data/lcesarini/SPHERA/tp/200001_mid1_Tdeep_tpH_zoom_remapped.grb2',engine="cfgrib").load()

            q99_ori=ds_cfgrib.groupby(ds_cfgrib['time.hour']).quantile(q=0.999,dim='time')

            xr.where(ds_cfgrib)

            ds_cfgrib.isel(time=24,step=0).tp.values

            plt.imshow(np.nanmax(ds_cfgrib.isel(time=1).tp.values,axis=0))
            plt.show()
            q99_hourly=ds_cfgrib.groupby(ds_cfgrib['time.hour']).quantile(q=0.999)
            ll_sphera=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS//reanalysis/SPHERA/pr/*{year}*") for year in np.arange(2003,2018)]
            ds_rg=xr.open_mfdataset(item for list in ll_sphera for item in list).load()            
            q99_remap=ds_rg.groupby(ds_rg['time.hour']).quantile(q=0.99,dim='time')
            
            ds_rg_jja=get_season(ds_rg,'JJA')

            ds_rg_jja_wh=xr.where(ds_rg_jja.pr > 0.1,ds_rg_jja.pr,np.nan)

            mean_intensity_diurnal=ds_rg_jja_wh.groupby(ds_rg_jja_wh['time.hour']).mean(dim='time',skipna=True)

            ax=plt.axes()
            plt.plot(mean_intensity_diurnal.mean(dim=['longitude','latitude']),marker='*')
            plt.suptitle(f"""
                        MEAN INTENSITY
                        """)
            ax.set_ylim((0.5,3))
            plt.show()

            for _ in range(20):
                random_lon=np.random.randint(low=0,high=q99_remap.longitude.shape[0])
                random_lat=np.random.randint(low=0,high=q99_remap.latitude.shape[0])
                plt.plot(q99_remap.pr.isel(longitude=random_lon,latitude=random_lat))
                # plt.plot(q99_remap.pr.mean(dim=['longitude','latitude']))
                plt.suptitle(f"""
                            {q99_remap.pr.isel(longitude=random_lon,latitude=random_lat).longitude.item():.2f}-{q99_remap.pr.isel(longitude=random_lon,latitude=random_lat).latitude.item():.2f}
                            """)
                plt.show()
            new_array=np.zeros(shape=(ds_cfgrib.time.shape[0],ds_cfgrib.y.shape[0],ds_cfgrib.x.shape[0])) * np.nan
            new_array2=np.zeros(shape=(ds_cfgrib.time.shape[0],ds_cfgrib.y.shape[0],ds_cfgrib.x.shape[0])) * np.nan

            ll_sphera=[ds_cfgrib.tp.values[time,step,:,:] for step,time in zip(np.tile(np.arange(24),20),np.arange(ds_cfgrib.time.shape[0]))]

            arr=np.array(ll_sphera)
            # pcm=plt.imshow(np.flip(np.quantile(arr,q=0.95,axis=0),axis=0))
            # plt.colorbar(pcm,shrink=0.6)
            # plt.show()

            # ax=plt.axes()
            # plt.plot(ds_cfgrib.time.values,arr[:,111,111],marker='*')
            # plt.plot(new_array[238:266,12,12],marker='^')
            # [plt.axvline(x=l,linestyle='--',color='r') for l in ds_cfgrib.time.values[::24]]
            # ax.set_xticklabels(ds_cfgrib.time.values,rotation=90);
            # plt.show()


            f_step=[i for _ in np.arange((ds_cfgrib.time.shape[0] / 24)) for i in np.arange(0,24)]

            for i,j in zip(f_step,np.arange(ds_cfgrib.time.shape[0])):
                if i == 0:
                    # print(j)
                    j_value=arr[j,:,:]
                    new_array2[j,:,:]=j_value
                else:
                    j_value=arr[j,:,:]
                    j_t_1=arr[j-1,:,:]
                    new_array2[j,:,:]=j_value-j_t_1

            new_ds = xr.DataArray(new_array2, 
                coords={'time':ds_cfgrib.time,
                        'lon': ds_cfgrib.x.values, 
                        'lat': ds_cfgrib.y.values
                        },
                # coords=dict(UAS.uas.coords),
                dims={'time':ds_cfgrib.time.shape[0],
                        'lat':ds_cfgrib.y.shape[0],
                        'lon':ds_cfgrib.x.shape[0]},
                # dims=UAS.uas.dims,
                # attrs = UAS.uas.attrs
                ) 
            
            q99_test=new_ds.groupby(new_ds['time.hour']).quantile(q=0.99,dim='time',skipna=True)

            ax=plt.axes()
            plt.plot(q99_test.mean(dim=['lon','lat']),marker='*')
            plt.suptitle(f"""
                        TEST ON ORIGINAL
                        """)
            ax.set_ylim((0.5,3))
            plt.show()
            # ax=plt.axes()
            # plt.plot(ds_cfgrib.time.values,arr[:,111,111],marker='*')
            # plt.plot(ds_cfgrib.time.values,new_array2[:,111,111],marker='^')
            # [plt.axvline(x=l,linestyle='--',color='r') for l in ds_cfgrib.time.values[::24]]
            # ax.set_xticklabels(ds_cfgrib.time.values,rotation=90);
            # plt.legend(["original","decumulated"])
            # plt.show()




            for i,j in zip(f_step,np.arange(ds_cfgrib.time.shape[0])):
                if i == 0:
                    j_value=ds_cfgrib.isel(step=i,time=j).tp.values
                    new_array[j,:,:]=j_value
                else:
                    j_value=ds_cfgrib.isel(step=i,time=j).tp.values
                    j_t_1=ds_cfgrib.isel(step=i-1,time=j-1).tp.values
                    new_array[j,:,:]=j_value-j_t_1

            ax=plt.axes()
            plt.plot(ds_cfgrib.time.values,arr[:,111,111],marker='*')
            # plt.plot(ds_cfgrib.time.values,
            #          ds_rg.sel(time=ds_rg.time.isin(ds_cfgrib.time)).pr.values[:,111,111],
            #          marker='x')
            plt.plot(ds_cfgrib.time.values,new_array2[:,111,111],marker='^')
            plt.plot(ds_cfgrib.time.values,new_array[:,111,111],marker='+')
            [plt.axvline(x=l,linestyle='--',color='r') for l in ds_cfgrib.time.values[::24]]
            ax.set_xticklabels(ds_cfgrib.time.values.astype('datetime64[s]'),rotation=45);
            plt.legend(["original","nc in cariparo","decumulated","old decumulated"])
            plt.show()

            if np.nanmax(new_array2-new_array) > 0:
                print(f"{i}: {np.nanmax(new_array2-new_array):.2f}")

        except Exception as e:
            print(f"An error occurred while processing {matching_files[i]}: {e}")

    ds_dropped=ds_cfgrib.drop_dims("step")
    ds_3d=ds_dropped.assign(pr=(["time", "latitude", "longitude"],new_array))

    """
    STEP 3
    """
    ds_3d.to_netcdf(f'{PATH_SPHERA_OUT}/{file[1][0]}.nc')

list_sphera=\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2000*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2001*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2002*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2003*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2004*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2005*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2006*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2007*')+\
glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2008*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2009*')


import subprocess

for file in list_sphera:
    subprocess.run(f"cdo remapycon,{PATH_GRID} {file} /mnt/data/lcesarini/SPHERA/tp/{os.path.basename(file)}_remapped.nc",shell=True)

# ds=xr.open_dataset('/mnt/data/lcesarini/tmp/200111_ten3_Tdeep_tpH_zoom.grb2',engine="cfgrib")
# ds_cfgrib=xr.open_dataset('/mnt/data/lcesarini/tmp/200111_ten3_Tdeep_tpH_zoom_remap.grb2',engine="cfgrib")

# new_array=np.zeros(shape=(ds_cfgrib.time.shape[0],ds_cfgrib.latitude.shape[0],ds_cfgrib.longitude.shape[0])) * np.nan


# f_step=[i for _ in np.arange((ds_cfgrib.time.shape[0] / 24)) for i in np.arange(0,24)]

# for i,j in zip(f_step,np.arange(ds_cfgrib.time.shape[0])):
#     if i == 0:
#         j_value=ds_cfgrib.isel(step=i,time=j).tp.values
#         new_array[j,:,:]=j_value
#     else:
#         j_value=ds_cfgrib.isel(step=i,time=j).tp.values
#         j_t_1=ds_cfgrib.isel(step=i-1,time=j-1).tp.values
#         new_array[j,:,:]=j_value-j_t_1


# ds_dropped=ds_cfgrib.drop_dims("step")
# ds_3d=ds_dropped.assign(pr=(["time", "latitude", "longitude"],new_array))

# ds_3d.pr.max()
# ds_3d.to_netcdf('/mnt/data/lcesarini/tmp/decumulated_prec.nc')
# to_grib(ds_3d,"/mnt/data/lcesarini/tmp/decumulated_prec.grb2",grib_keys={"edition":2})

# ax=plt.axes(projection=ccrs.PlateCarree())
# ds_3d.pr.quantile(q=0.99,dim='time').plot.pcolormesh(cmap='RdBu',ax=ax)
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
# plt.savefig("figures/map_shera.png")
# plt.close()

# # cdo remapycon,/home/lcesarini/2022_resilience/scripts/newcommongrid.txt 200111_ten3_Tdeep_tpH_zoom.grb2 200111_ten3_Tdeep_tpH_zoom_remap.grb2


# df=pd.read_csv("/mnt/data/lcesarini/tmp/data.txt",sep=" ", header=0)
# df.Value[0:10]

# ls_value=[]

# for _ in np.arange(240*260*214):
#     with open("/mnt/data/lcesarini/tmp/data.txt") as file:
#             ls_value.append(float(file.readline()[19:35]))


# np.array(ls_value).max()

# glob([f'{PATH_SPHERA_OUT}/*{year}*' for year in np.arange(2000,2010)])
# list_sphera=[]
# for year in np.arange(2000,2010):
#     list_sphera.append(glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/{year}*') )

# list_sphera=\
# glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2000*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2001*')+\
# glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2002*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2003*')+\
# glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2004*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2005*')+\
# glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2006*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2007*')+\
# glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2008*')+glob(f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*2009*')

# subprocess.run("ls /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{2000,2001}*",shell=True)


# ds_sphera=xr.open_mfdataset(list_sphera)
# cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()


# sphera_in_cnrm=[t_sphera in cnrm_rg.time_bnds[:,1] for t_sphera in ds_sphera.time.values]
# cnrm_in_sphera=[t_cnrm in ds_sphera.time for t_cnrm in cnrm_rg.time_bnds[:,1].values]


# np.all(sphera_in_cnrm)
# cnrm_rg.time.values[np.argwhere(np.array(cnrm_in_sphera)==0)]


# sp_2000=xr.open_dataset("/mnt/data/RESTRICTED/SPHERA/200001_ten3_Tdeep_tpH_zoom")



# 87672-87543
# cnrm_rg


# "2000-01-30T00:30:00.000000000"

list_sphera=\
glob(f'/mnt/data/lcesarini/SPHERA/tp/*2000*zoom.grb2*')+glob(f'/mnt/data/lcesarini/SPHERA/tp/*2001*zoom.grb2*')+\
glob(f'/mnt/data/lcesarini/SPHERA/tp/*2002*zoom.grb2*')+glob(f'/mnt/data/lcesarini/SPHERA/tp/*2003*zoom.grb2*')+\
glob(f'/mnt/data/lcesarini/SPHERA/tp/*2004*zoom.grb2*')+glob(f'/mnt/data/lcesarini/SPHERA/tp/*2005*zoom.grb2*')+\
glob(f'/mnt/data/lcesarini/SPHERA/tp/*2006*zoom.grb2*')+glob(f'/mnt/data/lcesarini/SPHERA/tp/*2007*zoom.grb2*')+\
glob(f'/mnt/data/lcesarini/SPHERA/tp/*2008*zoom.grb2*')+glob(f'/mnt/data/lcesarini/SPHERA/tp/*2009*zoom.grb2*')


len(list_sphera)

ds_cfgrib=xr.open_mfdataset(glob(f'/mnt/data/lcesarini/SPHERA/tp/*2000*2001*2002*zoom.grb2*'),engine="cfgrib").load()

jja=ds_cfgrib.isel(time=ds_cfgrib["time.month"].isin([6,7,8]))

xxx=np.nanmean(ds_cfgrib.tp,axis=(0,2,3))
x_max=ds_cfgrib.quantile(q=0.99,dim='step')

xx=x_max.groupby(jja['time.hour']).quantile(q=0.99)
xx=jja.groupby(jja['time.hour']).max()
xx.mean()
plt.plot(xx.tp.mean(dim=["x","y"]),"-*")
plt.savefig("figures/orig_sphera_diurnal.png")
plt.close()

