#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
from resilience.utils import *
import rasterio
import argparse
# import rioxarray
import numpy as np 
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from xclim import sdba
import subprocess as sb
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

"""


1) Compute the 99.9th percentile for the model adjustment period (on a file)
2) find the precipitaiton of the model in the calibration period corresponding to the 99.9th quantile of the adjustment
3) find the precipitaiton of the observations corresponding to the quantiles of the training which is equal to the 99.9 in the adjustment
4) Also, 99.9 of model train on file
5) 99.9 of observations train on file
6) 99.9 of observations adjustment on file



"""

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/data/lcesarini/BIAS_CORRECTED/" 
PATH_TEST="/mnt/ssd/lcesarini/test_djf/"
REF = "STATIONS"

if not os.path.exists(PATH_TEST): 
    os.makedirs(PATH_TEST)

yrs_train = np.arange(2000,2005)
yrs_valid = np.arange(2005,2010)

mask=xr.open_dataset("data/mask_stations_nan_common.nc")
sta_val=xr.open_mfdataset([f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()

list_ref_sphera=[glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{year}*") for year in np.arange(2000,2010)]
sphera=xr.open_mfdataset([item for list in list_ref_sphera for item in list]).load()

sphera["longitude"]=mask.lon.values
sphera["latitude"]=mask.lat.values
sphera=sphera.rename({"longitude":"lon","latitude":"lat"})

sta_cal_all=sta_val.sel(time=sta_val['time.year'].isin(yrs_train))
sta_adj_all=sta_val.sel(time=sta_val['time.year'].isin(yrs_valid))

sph_cal_all=sphera.sel(time=sphera['time.year'].isin(yrs_train))
sph_adj_all=sphera.sel(time=sphera['time.year'].isin(yrs_valid))

"""
CHECK ON THE CLOSE STATIONS WITH 
"""
SEAS='DJF'
mdl='ICTP'
mod_cal = xr.open_mfdataset(get_unlist([glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/pr/*{yr}*") for yr in yrs_train]))
mod_cal = mod_cal.sel(time=mod_cal['time.season'].isin(SEAS)).load()

mod_adj = xr.open_mfdataset(get_unlist([glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/pr/*{yr}*") for yr in yrs_valid])[1:])
mod_adj = mod_adj.sel(time=mod_adj['time.season'].isin(SEAS)).load()


sta_cal=sta_cal_all.sel(time=sta_cal_all['time.season'].isin("DJF"))
sta_adj=sta_adj_all.sel(time=sta_adj_all['time.season'].isin("DJF"))

sph_cal=sph_cal_all.sel(time=sph_cal_all['time.season'].isin("DJF"))
sph_adj=sph_adj_all.sel(time=sph_adj_all['time.season'].isin("DJF"))

sta_cal_1=sta_cal.sel(lon=11.66985,lat=45.3394,method='nearest')
sta_cal_2=sta_cal.sel(lon=11.69735,lat=45.31244,method='nearest')
sta_adj_1=sta_adj.sel(lon=11.66985,lat=45.3394,method='nearest')
sta_adj_2=sta_adj.sel(lon=11.69735,lat=45.31244,method='nearest')

sph_cal_1=sph_cal.sel(lon=11.66985,lat=45.3394,method='nearest')
sph_cal_2=sph_cal.sel(lon=11.69735,lat=45.31244,method='nearest')
sph_adj_1=sph_adj.sel(lon=11.66985,lat=45.3394,method='nearest')
sph_adj_2=sph_adj.sel(lon=11.69735,lat=45.31244,method='nearest')

mod_cal_1=mod_cal.sel(lon=11.66985,lat=45.3394,method='nearest')
mod_cal_2=mod_cal.sel(lon=11.69735,lat=45.31244,method='nearest')
mod_adj_1=mod_adj.sel(lon=11.66985,lat=45.3394,method='nearest')
mod_adj_2=mod_adj.sel(lon=11.69735,lat=45.31244,method='nearest')

sta_cal_1.pr.plot(label='sta_cal_1')
sph_cal_1.pr.plot(label='sph_cal_1')
sta_cal_2.pr.plot(label='sta_cal_2')
sph_cal_2.pr.plot(label='sph_cal_2')
sta_adj_1.pr.plot(label='sta_adj_1')
sph_adj_1.pr.plot(label='sph_adj_1')
sta_adj_2.pr.plot(label='sta_adj_2')
sph_adj_2.pr.plot(label='sph_adj_2')
mod_adj_1.pr.plot(label='mod_adj_1')
mod_adj_2.pr.plot(label='mod_adj_2')
mod_cal_1.pr.plot(label='mod_cal_1')
mod_cal_2.pr.plot(label='mod_cal_2')
plt.legend()
plt.show()



plt.boxplot([
            sta_cal_1.pr.values[sta_cal_1.pr.values > 0.2],
            sta_cal_2.pr.values[sta_cal_2.pr.values > 0.2],
            sta_adj_1.pr.values[sta_adj_1.pr.values > 0.2],
            sta_adj_2.pr.values[sta_adj_2.pr.values > 0.2],
            sph_cal_1.pr.values[sph_cal_1.pr.values > 0.1],
            sph_cal_2.pr.values[sph_cal_2.pr.values > 0.1],
            sph_adj_1.pr.values[sph_adj_1.pr.values > 0.1],
            sph_adj_2.pr.values[sph_adj_2.pr.values > 0.1],
            mod_cal_1.pr.values[mod_cal_1.pr.values > 0.1],
            mod_cal_2.pr.values[mod_cal_2.pr.values > 0.1],
            mod_adj_1.pr.values[mod_adj_1.pr.values > 0.1],
            mod_adj_2.pr.values[mod_adj_2.pr.values > 0.1]
            ],
            labels=[
                "sta_cal_1","sta_cal_2","sta_adj_1","sta_adj_2",
                "sph_cal_1","sph_cal_2","sph_adj_1","sph_adj_2",
                "mod_cal_1","mod_cal_2","mod_adj_1","mod_adj_2"
                ])
plt.legend()
plt.show()



stats.percentileofscore(a=sta_cal_1.pr.values,score=0.2,kind='strict')
np.nanquantile(sph_cal_1.pr.values,q=0.9262536873156343)
stats.mstats.mquantiles(sph_cal_1.pr.values[np.isfinite(sph_cal_1.pr.values)], prob=[0.92],alphap=0.4,betap=0.4)
[
            sta_cal_1.pr.values[sta_cal_1.pr.values > 0.12].shape[0],
            sta_cal_2.pr.values[sta_cal_2.pr.values > 0.2].shape[0],
            sta_adj_1.pr.values[sta_adj_1.pr.values > 0.2].shape[0],
            sta_adj_2.pr.values[sta_adj_2.pr.values > 0.2].shape[0],
            sph_cal_1.pr.values[sph_cal_1.pr.values > 0.13212109].shape[0],
            sph_cal_2.pr.values[sph_cal_2.pr.values > 0.13212109].shape[0],
            sph_adj_1.pr.values[sph_adj_1.pr.values > 0.13212109].shape[0],
            sph_adj_2.pr.values[sph_adj_2.pr.values > 0.13212109].shape[0],
            mod_cal_1.pr.values[mod_cal_1.pr.values > 0.1].shape[0],
            mod_cal_2.pr.values[mod_cal_2.pr.values > 0.1].shape[0],
            mod_adj_1.pr.values[mod_adj_1.pr.values > 0.1].shape[0],
            mod_adj_2.pr.values[mod_adj_2.pr.values > 0.1].shape[0]
            ]

fig=plt.figure(figsize=(15,10))
ax=plt.axes()
sns.ecdfplot(sta_cal_1.pr.values[sta_cal_1.pr.values > 0.2],ax=ax,label='sta_cal_1',linestyle="dashed",color='red'  )
sns.ecdfplot(sta_cal_2.pr.values[sta_cal_2.pr.values > 0.2],ax=ax,label='sta_cal_2',linestyle="dashed",color='grey' )
sns.ecdfplot(sta_adj_1.pr.values[sta_adj_1.pr.values > 0.2],ax=ax,label='sta_adj_1',linestyle="dashed",color='green')
sns.ecdfplot(sta_adj_2.pr.values[sta_adj_2.pr.values > 0.2],ax=ax,label='sta_adj_2',linestyle="dashed",color='blue' )
# sns.ecdfplot(sph_cal_1.pr.values[sph_cal_1.pr.values > 0.1],ax=ax,label='sph_cal_1',color='red'  )
# sns.ecdfplot(sph_cal_2.pr.values[sph_cal_2.pr.values > 0.1],ax=ax,label='sph_cal_2',color='grey' )
# sns.ecdfplot(sph_adj_1.pr.values[sph_adj_1.pr.values > 0.1],ax=ax,label='sph_adj_1',color='green')
# sns.ecdfplot(sph_adj_2.pr.values[sph_adj_2.pr.values > 0.1],ax=ax,label='sph_adj_2',color='blue' )
sns.ecdfplot(mod_cal_1.pr.values[mod_cal_1.pr.values > 0.1],ax=ax,label='mod_cal_1',linestyle="solid",color='red'  )
sns.ecdfplot(mod_cal_2.pr.values[mod_cal_2.pr.values > 0.1],ax=ax,label='mod_cal_2',linestyle="solid",color='grey' )
sns.ecdfplot(mod_adj_1.pr.values[mod_adj_1.pr.values > 0.1],ax=ax,label='mod_adj_1',linestyle="solid",color='green')
sns.ecdfplot(mod_adj_2.pr.values[mod_adj_2.pr.values > 0.1],ax=ax,label='mod_adj_2',linestyle="solid",color='blue' )

plt.scatter(np.quantile(sta_cal_1.pr.values[sta_cal_1.pr.values > 0.2],0.999),0.999,color='red'  ,label='sta_cal_1',marker='+')
plt.scatter(np.quantile(sta_cal_2.pr.values[sta_cal_2.pr.values > 0.2],0.999),0.999,color='grey' ,label='sta_cal_2',marker='+')
plt.scatter(np.quantile(sta_adj_1.pr.values[sta_adj_1.pr.values > 0.2],0.999),0.999,color='green',label='sta_adj_1',marker='+')
plt.scatter(np.quantile(sta_adj_2.pr.values[sta_adj_2.pr.values > 0.2],0.999),0.999,color='blue' ,label='sta_adj_2',marker='+')
# plt.scatter(np.quantile(sph_cal_1.pr.values[sph_cal_1.pr.values > 0.1],0.999),0.999,color='red'  ,label='sph_cal_1')
# plt.scatter(np.quantile(sph_cal_2.pr.values[sph_cal_2.pr.values > 0.1],0.999),0.999,color='grey' ,label='sph_cal_2')
# plt.scatter(np.quantile(sph_adj_1.pr.values[sph_adj_1.pr.values > 0.1],0.999),0.999,color='green',label='sph_adj_1')
# plt.scatter(np.quantile(sph_adj_2.pr.values[sph_adj_2.pr.values > 0.1],0.999),0.999,color='blue' ,label='sph_adj_2')
plt.scatter(np.quantile(mod_cal_1.pr.values[mod_cal_1.pr.values > 0.1],0.999),0.999,color='red'  ,label='mod_cal_1',marker='*')
plt.scatter(np.quantile(mod_cal_2.pr.values[mod_cal_2.pr.values > 0.1],0.999),0.999,color='grey' ,label='mod_cal_2',marker='*')
plt.scatter(np.quantile(mod_adj_1.pr.values[mod_adj_1.pr.values > 0.1],0.999),0.999,color='green',label='mod_adj_1',marker='*')
plt.scatter(np.quantile(mod_adj_2.pr.values[mod_adj_2.pr.values > 0.1],0.999),0.999,color='blue' ,label='mod_adj_2',marker='*')


ax.set_ylim(0.99,1.01)
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.show()
"""
END CHECK
"""

# plt.hist(sta_cal_999.pr.values.ravel(),label="cal")
# plt.hist(sta_adj_999.pr.values.ravel(),label="adj")
# plt.legend()
# plt.savefig("sss.png")

# sta_cal_999.to_netcdf(f"{PATH_TEST}sta_cal_999.nc")
# sta_adj_999.to_netcdf(f"{PATH_TEST}sta_adj_999.nc")

max_sta=np.nanmax(sta_val.pr.values[:,:,:],axis=2)

x,y=np.where((max_sta > 0))
x_,y_=np.where(np.isnan(max_sta))

xy=np.concatenate([y.reshape(-1,1),x.reshape(-1,1)],axis=1)
xy_=np.concatenate([y_.reshape(-1,1),x_.reshape(-1,1)],axis=1)
assert xy.shape[0] == 172, f"Number of cells, different from number {xy.shape[0]}"


list_mdl=["CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","MOHC"] #
for SEAS in tqdm(["SON","MAM","DJF","JJA"],total=4):
    sta_cal=sta_cal_all.sel(time=sta_cal_all['time.season'].isin(SEAS))
    sta_adj=sta_adj_all.sel(time=sta_adj_all['time.season'].isin(SEAS))

    sph_cal=sph_cal_all.sel(time=sph_cal_all['time.season'].isin(SEAS))
    sph_adj=sph_adj_all.sel(time=sph_adj_all['time.season'].isin(SEAS))

    sta_cal_999=sta_cal.quantile(q=0.999,dim="time")
    sta_adj_999=sta_adj.quantile(q=0.999,dim="time")

    sph_cal_999=get_triveneto(sph_cal.quantile(q=0.999,dim="time"),sta_cal_999)
    sph_adj_999=get_triveneto(sph_adj.quantile(q=0.999,dim="time"),sta_cal_999)

    for mdl in list_mdl:
        heavy_mod_adjustment = xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_q_biased_STATIONS_1000_SEQUENTIAL_VALIDATION.nc")
        heavy_mod_adjustment_eqm = get_triveneto(xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_q_EQM_STATIONS_1000_SEQUENTIAL_VALIDATION.nc"),sta_cal_999)
        
        mod_cal = xr.open_mfdataset(get_unlist([glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/pr/*{yr}*") for yr in yrs_train]))
        mod_cal = mod_cal.sel(time=mod_cal['time.season'].isin(SEAS)).load()
        mod_cal_tri=get_triveneto(mod_cal.pr,sta_cal_999)
        heavy_mod_adjustment_tri=get_triveneto(heavy_mod_adjustment.pr,sta_cal_999)

        for id_coord in xy:

            score_tr=stats.percentileofscore(a=mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1],
                                            score=heavy_mod_adjustment_tri.isel(lon=id_coord[0],lat=id_coord[1]).values.item(), 
                                            nan_policy='omit')

            # from statsmodels.distributions.empirical_distribution import ECDF

            # x=ECDF(mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1])
            # x(heavy_mod_adjustment_tri.isel(lon=id_coord[0],lat=id_coord[1]).values.item())

            q_mt=np.nanquantile(mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1],min(score_tr/100,1))
            q_mt_scipy=stats.mstats.mquantiles(mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1],min(score_tr/100,1),alphap=0.4,betap=0.4).item()
            q_mt_999=np.nanquantile(mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1],q=0.999)
            q_ot=np.nanquantile(sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1])[sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1]) > 0.2],min(score_tr/100,1))  
            q_ot_scipy=stats.mstats.mquantiles(sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1])[sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1]) > 0.2],min(score_tr/100,1),alphap=0.4,betap=0.4).item()
            q_ot_999=np.nanquantile(sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1])[sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1]) > 0.2],q=0.999)
            q_oa_999=np.nanquantile(sta_adj.pr.isel(lon=id_coord[0],lat=id_coord[1])[sta_adj.pr.isel(lon=id_coord[0],lat=id_coord[1]) > 0.2],q=0.999)
            q_ma=heavy_mod_adjustment_tri.isel(lon=id_coord[0],lat=id_coord[1]).values.item()

            heavy_mod_adjustment_eqm.pr.isel(lon=id_coord[0],lat=id_coord[1])
            heavy_mod_adjustment_tri.isel(lon=id_coord[0],lat=id_coord[1])
            
            dict_df={
                "model":mdl,
                "seas":SEAS,
                "id_lon":id_coord[0],
                "id_lat":id_coord[1],
                "hv_mdl_adj":np.float64(q_ma).round(2),
                "hv_mdl_adj_eqm":np.round(heavy_mod_adjustment_eqm.pr.isel(lon=id_coord[0],lat=id_coord[1]).item(),2),
                "hv_mdl_cal_at_adj":q_mt.round(2),
                "hv_obs_cal_at_adj":q_ot.round(2),
                "hv_obs_cal_at_adj_scipy":np.float64(q_ot_scipy).round(2),
                "hv_mod_cal":q_mt_999.round(2),
                "hv_obs_cal":q_ot_999.round(2),
                "hv_obs_adj":q_oa_999.round(2)
            }
            # dict_2={
            #     "model":mdl,
            #     "Reference":REF,
            #     "seas":SEAS,
            #     "id_lon":id_coord[0],
            #     "id_lat":id_coord[1],
            #     "N_WH_mod_cal":mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1].values.shape[0],
            #     "N_WH_obs_cal":sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1])[sta_cal.pr.isel(lon=id_coord[0],lat=id_coord[1]) > 0.2].values.shape[0]
            # }
                # sb.run(f"echo {mdl} DJF {id_coord[0]} {id_coord[1]} {q_ma:.2f} {q_mt:.2f} {q_ot:.2f} {q_mt_999:.2f} {q_ot_999:.2f} {q_oa_999:.2f} >> output/bias_increases.txt",shell=True)

            # os.remove(f"{PATH_TEST}bias_increases.csv")
            if not os.path.exists(f"{PATH_TEST}bias_increases.csv"):
                pd.DataFrame(dict_df,index=[0],columns=dict_df.keys()).to_csv(f"{PATH_TEST}bias_increases.csv",index=False,mode="w")
            else:
                pd.DataFrame(dict_df,index=[0],columns=dict_df.keys()).to_csv(f"{PATH_TEST}bias_increases.csv",index=False,mode="a",header=None)

            # if not os.path.exists(f"{PATH_TEST}plotting_position.csv"):
            #     pd.DataFrame(dict_2,index=[0],columns=dict_2.keys()).to_csv(f"{PATH_TEST}plotting_position.csv",index=False,mode="w")
            # else:
            #     pd.DataFrame(dict_2,index=[0],columns=dict_2.keys()).to_csv(f"{PATH_TEST}plotting_position.csv",index=False,mode="a",header=None)


            x=mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1])[mod_cal_tri.isel(lon=id_coord[0],lat=id_coord[1]) > 0.1].values

            np.nanquantile(x,0.9987261146496816).round(2)

            stats.mstats.mquantiles(x, prob=[0.9987261146496816],alphap=0.4,betap=0.4)


            # np.sort(x)


             


# scp luigi.cesarini@eucentre.loc@172.16.20.2:/home/giorgia.fosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200701010030-200712312330.nc lcesarini@192.167.76.81:/mnt/data/lcesarini/
# scp luigi.cesarini@eucentre.loc@172.16.20.2:/home/giorgia.fosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200801010030-200812312330.nc lcesarini@192.167.76.81:/mnt/data/lcesarini/
# scp luigi.cesarini@eucentre.loc@172.16.20.2:/home/giorgia.fosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200901010030-200912312330.nc lcesarini@192.167.76.81:/mnt/data/lcesarini/