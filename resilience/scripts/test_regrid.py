#! /home/lcesarini/miniconda3/envs/detectron/bin/python
import os
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import norm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from shapely.geometry import mapping
from cartopy import feature as cfeature
from math import pi,sin,cos, asin, atan2
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,mean_absolute_percentage_error)

import warnings
warnings.filterwarnings('ignore')

from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

from scripts.utils import *
PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"



#PROBLEM WITH TIME VALUES OF 2009 for ETH
eth__rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/ETH/CPM/pr/ETH_ECMWF-ERAINT_{yr}01010030_{yr}12312330.nc" for yr in np.arange(2000,2010)]).load()
mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/pr/MOHC_ECMWF-ERAINT_*.nc").load()
ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/pr/ICTP_ECMWF-ERAINT_*.nc").load()
hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/pr/HCLIMcom_ECMWF-ERAINT_*.nc").load()
cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
knmi_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()

eth__slice=eth__rg.isel(time=eth__rg["time.year"].isin(np.arange(2000,2009)))
del eth__rg
mohc_slice=mohc_rg.isel(time=mohc_rg["time.year"].isin(np.arange(2000,2009)))
del mohc_rg
ictp_slice=ictp_rg.isel(time=ictp_rg["time.year"].isin(np.arange(2000,2009)))
del ictp_rg
hcli_slice=hcli_rg.isel(time=hcli_rg["time.year"].isin(np.arange(2000,2009)))
del hcli_rg
cnrm_slice=cnrm_rg.isel(time=cnrm_rg["time.year"].isin(np.arange(2000,2009)))
del cnrm_rg
knmi_slice=knmi_rg.isel(time=knmi_rg["time.year"].isin(np.arange(2000,2009)))
del knmi_rg

eth__slice.update({'time':mohc_slice.time})

ens_raw=(eth__slice.pr+mohc_slice.pr+ictp_slice.pr+hcli_slice.pr+cnrm_slice.pr) / 5
ds_ensembl=ens_raw.to_dataset(name='pr')


"""
RUN A CHECK ON QUANTILE OF THE MEAN OR MEAN OF THE QUANTILE
"""

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI']

array_model=[eth__slice,mohc_slice,ictp_slice,hcli_slice,cnrm_slice,knmi_slice]

dict_metrics={}

for name,mdl in tqdm(zip(name_models,array_model), total=len(array_model)):
    
    dict_0={name:compute_metrics(mdl,meters=True,quantile=0.999)}
    
    dict_metrics.update(dict_0)

mean_q=(dict_metrics['ETH'][2]+dict_metrics['MOHC'][2]+dict_metrics['ICTP'][2]+dict_metrics['HCLIMcom'][2]+dict_metrics['CNRM'][2]+dict_metrics['KNMI'][2]) / 6

eth__rg.update({'time':mohc_rg.time})

ens_raw=(eth__rg.pr+mohc_rg.pr+ictp_rg.pr+hcli_rg.pr+cnrm_rg.pr) / 5
ds_ensembl=ens_raw.to_dataset(name='pr')
#Check random cell
np.random.seed(1)
i,j=np.random.randint(0,150),np.random.randint(0,150)
print(i,j)
#i,j=67,80
i,j=52,122
# st,fi=1406,1465
st,fi=0,87672
# st,fi=1000,1100
QQ=[0.50,.999]
m1=eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)
m2=mohc_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)
m3=ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)
m4=cnrm_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)
m5=hcli_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)

avg1=eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
    where(eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr).mean()
avg2=mohc_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
    where(mohc_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr).mean()
avg3=ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
    where(ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr).mean()
avg4=cnrm_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
    where(cnrm_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr).mean()
avg5=hcli_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
    where(hcli_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr).mean()

mx=ds_ensembl.isel(time=np.arange(st,fi),lon=i,lat=j).pr.quantile(q=QQ)
avg_mx=ds_ensembl.isel(time=np.arange(st,fi),lon=i,lat=j).pr.\
       where(ds_ensembl.isel(time=np.arange(st,fi),lon=i,lat=j).pr > 0.1).mean()

my=(m1+m2+m3+m4+m5)/5
avg_my=(avg1+avg2+avg3+avg4+avg5) / 5

lon,lat=eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).lon.item(),eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).lat.item()

plt.plot(np.sort(eth__rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker="d",markersize=5,color="grey",label='ETH',
         alpha=0.5)
plt.scatter(m1,QQ,marker="d",s=50,color="blue")
plt.plot(np.sort(mohc_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker="o",markersize=5,color="grey",label='MOHC',
         )
plt.scatter(m2,QQ,marker="o",s=50,color="blue")
plt.plot(np.sort(ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker="x",markersize=5,color="grey",label='ICTP',
         )
plt.scatter(m3,QQ,marker="x",s=50,color="blue")
plt.plot(np.sort(cnrm_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker="^",markersize=5,color="grey",label='CNRM',
         )
plt.scatter(m4,QQ,marker="^",s=50,color="blue")
plt.plot(np.sort(hcli_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker="1",markersize=5,color="grey",label='HCLIMcom',
         )
plt.scatter(m5,QQ,marker="1",s=50,color="blue")
plt.plot(np.sort(ds_ensembl.isel(time=np.arange(st,fi),lon=i,lat=j).pr),
         (np.arange(0,np.arange(st,fi).shape[0])+1)/np.arange(st,fi).shape,marker='*',markersize=5,color="red",label='ensemble',
         )
plt.scatter(mx,QQ,marker='*',s=400,color="red",label=f'Quantile of models avg   {mx[1].item():.3f}[mm/h]')

plt.scatter(my,QQ,marker='+',s=400,color="green",label=f"Avg of model's quantile {my[1].item():.3f}[mm/h]")
# plt.plot(mean_q.isel(time=np.arange(1400,1450),lon=i,lat=j).pr,color="grey50",label='ETH')
# plt.suptitle(f"cell at coords lon:{lon:.2f} lat:{lat:.2f}")
plt.title(f"""
          Avg. Ensemble distance from the mean: {(mx[1]-avg_mx).item():.2f}
          Avg of Qs distance from the mean: {(my[1]-avg_my).item():.2f}
          """)
plt.legend()
plt.savefig("figures/edcfs_models.png")
plt.close()

plt.hist(ds_ensembl.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='black')
plt.hist(ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='red')
plt.hist(mohc_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='green')
plt.hist(hcli_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='blue')
plt.hist(cnrm_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='yellow')
plt.hist(ictp_rg.isel(time=np.arange(st,fi),lon=i,lat=j).pr,bins=np.arange(0.1,15,0.1),color='white',edgecolor='magenta')
plt.vlines(avg_my.item(),ymin=0,ymax=350,color='red',linestyles='dashed')
plt.vlines(avg_mx.item(),ymin=0,ymax=350,color='red')
plt.vlines(mx[1].item(),ymin=0,ymax=350,color='green')
plt.vlines(my[1].item(),ymin=0,ymax=350,color='green',linestyles='dashed')
plt.vlines(mx[0].item(),ymin=0,ymax=350,color='blue')
plt.vlines(my[0].item(),ymin=0,ymax=350,color='blue',linestyles='dashed')
# plt.xlim(0,5)
plt.ylim(0,100)
plt.savefig("figures/histogram.png")
plt.close()

plt.plot("MOHC",m2[1],marker="o",color="grey")
plt.plot("ICTP",m3[1],marker="x",color="grey")
plt.plot("CNRM",m4[1],marker="^",color="grey")
plt.plot("HCLI",m5[1],marker="1",color="grey")
plt.plot("qavg",mx[1],marker='*',color="red",label='Quantile of models avg')
plt.plot("avgq",my[1],marker='+',color="green",label="Avg of moodel's quantile")
# plt.plot(mean_q.isel(time=np.arange(1400,1450),lon=i,lat=j).pr,color="grey50",label='ETH')
plt.legend()
plt.savefig("figures/q99_model_ensemble.png")
plt.close()






# xr_ens=xr.DataArray(ens_raw,
#                     coords={
#                         'time':mohc_slice.time,
#                         'lat':mohc_slice.lat,
#                         'lon':mohc_slice.lon,
#                         },
#                     dims={
#                         "time":mohc_slice.time.shape[0],
#                         "lat":mohc_slice.lat.shape[0],
#                         "lon":mohc_slice.lon.shape[0],
#                     }
#                     )


# np.unravel_index(ictp_slice.pr.values.argmax(), eth__slice.pr.values.shape)

"""
#CHECK DISPARITEIS BETWEEN MODELS
ictp_og='/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ICTP/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_ICTP-RegCM4-7_fpsconv-x2yn2-v1_1hr_20030101000000-20040101000000.nc'
mohc_og='/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_200301010030-200312312330.nc'
ict_orig=xr.open_dataset(ictp_og).sel(time='2003-12-03T12:30:00')
cnr_rig=xr.open_dataset(mohc_og).sel(time='2003-12-03T12:29:59.999997184')
cnr_rig.sel(time=cnr_rig["time.hour"].isin(12)).time.values
type(ict_orig.attrs)
ict_orig.where((ict_orig.lon > 7.9) & (ict_orig.lat > 46.5) &\
               (ict_orig.lon < 8)   & (ict_orig.lat < 46.7),
               drop=True).pr.max() * 3600

cnr_rig.where((cnr_rig.lon > 7.9) & (cnr_rig.lat > 46.5) & (cnr_rig.lon < 8)   & (cnr_rig.lat < 46.7),
               drop=True).pr.max() * 3600
ictp_slice.pr.isel(time=34380,lon=52,lat=122)
knmi_slice.pr.isel(time=34380,lon=52,lat=122)

ictp_rg.pr.values[34380,122,52]
eth__rg.pr.values[34380,122,52]
mohc_rg.pr.values[34380,122,52]
hcli_rg.pr.values[34380,122,52]
cnrm_rg.pr.values[34380,122,52]
knmi_rg.pr.values[34380,122,52]
ens_raw.values[34380,122,52]




i,j=np.random.randint(0,150),np.random.randint(0,150)
i,j=111,25
ams_1=eth__slice.isel(lon=i,lat=j).pr
ams_2=mohc_slice.isel(lon=i,lat=j).pr
ams_3=ictp_slice.isel(lon=i,lat=j).pr
ams_4=hcli_slice.isel(lon=i,lat=j).pr
ams_5=cnrm_slice.isel(lon=i,lat=j).pr
ams_6=ens_raw.isel(lon=i,lat=j)  

k=np.random.randint(0,35555)
ams_1=eth__slice.pr.quantile(q=0.999,dim='time')#.isel(time=k)
ams_2=mohc_slice.pr.quantile(q=0.999,dim='time')#.isel(time=k)
ams_3=ictp_slice.pr.quantile(q=0.999,dim='time')#.isel(time=k)
ams_4=hcli_slice.pr.quantile(q=0.999,dim='time')#.isel(time=k)
ams_5=cnrm_slice.pr.quantile(q=0.999,dim='time')#.isel(time=k)
ams_6=ens_raw.quantile(q=0.999,dim='time')#.isel(time=k)   
supposed_ens=(ams_1+ams_2+ams_3+ams_4+ams_5) / 5

plot_panel(
    nrow=2,ncol=3,
    list_to_plot=[ams_1,ams_2,ams_3,ams_4,ams_5,ams_6],
    name_fig="frequency_models",
    list_titles=name_models,
    levels=10,#np.arange(0.04,0.29,0.03),
    suptitle="Frequency of wet hours for JJA",
    name_metric="[fraction]",
    SET_EXTENT=False
)
# plt.plot(ams_1,color='grey',alpha=0.25,label="eth")
# plt.plot(ams_2,color='grey',alpha=0.25,label="mohc")
# plt.plot(ams_3,color='grey',alpha=0.25,label="ictp")
# plt.plot(ams_4,color='grey',alpha=0.25,label="hcli")
# plt.plot(ams_5,color='grey',alpha=0.25,label="cnrm")
plt.plot(ams_6,'x',markersize=2,alpha=0.5,label='ensemble')
plt.plot(supposed_ens,'o',alpha=0.5,label='supposed ensemble')
plt.legend()
plt.title(f"lon {i} lat {j}")
plt.savefig("figures/hist.png")
plt.close()

"""

ds_ens=ens_raw.to_dataset(name='pr')

name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI','Ensemble']

array_model=[eth__slice,mohc_slice,ictp_slice,hcli_slice,cnrm_slice]


dict_metrics={}

for name,mdl in tqdm(zip(name_models,array_model), total=len(array_model)):
    
    dict_0={name:compute_metrics(get_season(mdl,season='JJA'),meters=True,quantile=0.999)}
    
    dict_metrics.update(dict_0)


from scripts.utils import *
"""
FREQUENCY
"""
mean_f=(dict_metrics['ETH'][0]+dict_metrics['MOHC'][0]+dict_metrics['ICTP'][0]+dict_metrics['HCLIMcom'][0]+dict_metrics['CNRM'][0]+dict_metrics['KNMI'][0]) / 6
mean_i=(dict_metrics['ETH'][1]+dict_metrics['MOHC'][1]+dict_metrics['ICTP'][1]+dict_metrics['HCLIMcom'][1]+dict_metrics['CNRM'][1]+dict_metrics['KNMI'][1]) / 6
mean_q=(dict_metrics['ETH'][2]+dict_metrics['MOHC'][2]+dict_metrics['ICTP'][2]+dict_metrics['HCLIMcom'][2]+dict_metrics['CNRM'][2]+dict_metrics['KNMI'][2]) / 6

dict_0={'Ensemble':(mean_f,mean_i,mean_q)}

dict_metrics.update(dict_0)


cmap_freq = (mpl.colors.ListedColormap(['#B1DFFA',
                                        '#36BCFF', 
                                        '#508D5E',
                                        '#55CB70',
                                        '#E5E813',
                                        '#E8AB13',
                                        '#E85413',
                                        '#E82313'
                                        ])
        .with_extremes(over='#AB0202', under='#D8EEFA'))
plot_panel(
    nrow=3,ncol=3,
    list_to_plot=[dict_metrics[i][0] for i in name_models],
    name_fig="frequency_models",
    list_titles=name_models,
    levels=np.arange(0.04,0.29,0.03),
    suptitle="Frequency of wet hours for JJA",
    name_metric="[fraction]",
    SET_EXTENT=False,
    cmap=cmap_freq
)

"""
INTENSITY
"""

cmap_inte = (mpl.colors.ListedColormap(['#ECF7FE',
                                        '#B1DFFA',
                                        '#36BCFF', 
                                        '#508D5E',
                                        '#55CB70',
                                        '#88F7A1',
                                        '#E5E813',
                                        '#E8AB13',
                                        '#E85413',
                                        '#E82313'
                                        ])
        .with_extremes(over='#AB0202', under='#D8EEFA'))
plot_panel(
    nrow=3,ncol=3,
    list_to_plot=[dict_metrics[i][1] for i in name_models],
    name_fig="intensity_models",
    list_titles=name_models,
    levels=np.arange(0.3,3.31,0.3),
    suptitle="Intensity of wet hours [mm/h] for JJA",
    name_metric="[mm/h]",
    SET_EXTENT=False,
    cmap=cmap_inte
)

"""
HEAVY PREC
"""

cmap_q = (mpl.colors.ListedColormap(['#ECF7FE',
                                     '#B1DFFA',
                                     '#36BCFF', 
                                     '#508D5E',
                                     '#55CB70',
                                     '#88F7A1',
                                     '#E5E813',
                                     '#E8AB13',
                                     '#E85413',
                                     '#E82313'
                                     ])
        .with_extremes(over='#AB0202', under='#D8EEFA'))
# list_to_plot=[dict_metrics[i][2] for i in name_models]
# list_to_plot=[mean_q]
# for i in name_models[1:]:
#     list_to_plot.append(dict_metrics[i][2])
# len(list_to_plot)

plot_panel(
    nrow=3,ncol=3,
    list_to_plot=[dict_metrics[i][2] for i in name_models],
    name_fig="quantile_models",
    list_titles=name_models,
    levels=np.arange(2,19,2),
    suptitle="Heavy Prec q99.9 [mm/h] for JJA",
    name_metric="[mm/h]",
    SET_EXTENT=False,
    cmap=cmap_q
)




plot_panel(
    nrow=2,ncol=3,
    list_to_plot=[dict_metrics[i][2] for i in name_models],
    name_fig="quantile_models",
    list_titles=name_models,
    levels=np.arange(2,19,2),
    suptitle="Heavy Prec q99.9 [mm/h] for JJA",
    name_metric="[mm/h]",
    SET_EXTENT=False,
    cmap=cmap_q
)


name_metric=['freq','int','q']
for nmdl in name_models:
    for i in range(0,3):
        print(dict_metrics[nmdl][i].to_dataset(name=name_metric[i]).to_netcdf(f"/home/lcesarini/2022_resilience/output/{nmdl}_2000_2008_{name_metric[i]}.nc"))



# proj = ccrs.PlateCarree()
# if hasattr(xr_orig,"Lambert_Conformal"):
#     rot = ccrs.LambertConformal(central_longitude=16, central_latitude=45.5, 
#                                 false_easting=1349205.5349238443, false_northing=732542.657192843)
# else:
#     rot = ccrs.RotatedPole(pole_longitude=-170.0, 
#                         pole_latitude=43.0, 
#                         central_rotated_longitude=0.0, 
#                         globe=None)



# fig,ax = plt.subplots(nrows=2,
#                       ncols=2,#int(len(list_to_plot) / 2),
#                       figsize=(18,18),constrained_layout=True, squeeze=True,
#                       subplot_kw={"projection":ccrs.PlateCarree()}
#                       )

# ax=ax.flatten()

# title=["Original","Bicubic","Distance 4","Bilinear"]
# # fig.subplots_adjust(bottom=0.01, top=0.99,right=0.99, left=0.01)
# cmap = plt.cm.rainbow

# for i,metric in enumerate([xr_orig,xr_bicu,xr_dis4,xr_bili]):#test.pr.isel(time=19).plot(ax=ax,alpha=0.95)

#     print((metric.pr.quantile(q=[0.5,0.7,0.85,0.95,0.99,0.99999]) * 3600).values)
#     print(metric.pr.max() * 3600)


#     # if i in [0,3]:
#     #     bounds = np.array([0.02,0.06,0.12,0.16,0.22])
#     # elif i in [1,4]:
#     #     bounds = np.array([0.2,0.4,0.6,0.75,0.9])
#     # elif i in [2,5]:
    
#     bounds = np.linspace(0.1,21,20)#np.array([18,19,20,21,22,23,24])
    
#     norm = mpl.colors.BoundaryNorm(bounds.round(2), bounds.shape[0]+1, extend='both')
#     if i==0:
#         pcm=(metric.pr * 3600).plot.contourf(ax=ax[i],alpha=1,
#                         transform=rot,
#                         add_colorbar=True,
#                         cmap=cmap, norm=norm,
#                         cbar_kwargs={"shrink":0.7,
#                                     "orientation":"horizontal",
#                                     "label":f"{'fraction' if i in [len(ax)-3] else 'mm/h' if i in [len(ax)-2,len(ax)-1] else ''}"
#                                     }
#                         )
#     else:
#         pcm=(metric.pr * 3600).plot.contourf(ax=ax[i],alpha=1,
#                                             cmap=cmap, norm=norm, 
#                                             add_colorbar=True, 
#                                             cbar_kwargs={"shrink":0.7,
#                                                         "orientation":"horizontal",
#                                                         "label":f"{'fraction' if i in [len(ax)-3] else 'mm/h' if i in [len(ax)-2,len(ax)-1] else ''}"}
#                                             )

#     ax[i].coastlines()
#     gl = ax[i].gridlines(
#         draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',xlocs=[11],ylocs=[45,47]
#     )
#     ax[i].add_feature(cfeature.BORDERS, linestyle='--')
#     # ax[i].add_feature(cfeature.LAKES, alpha=0.5)
#     # ax[i].add_feature(cfeature.RIVERS, alpha=0.5)
#     # ax[i].add_feature(cfeature.STATES)
#     ax[i].set_title(f"{title[i]}")
#     # ax[i].set_ylabel(f"{'Winter' if i in [0,3] else 'Summer'}")
#     # shp_triveneto.boundary.plot(ax=ax[i], edgecolor="green")

#     # if i in [3,4,5]:
#     #     fig.colorbar(pcm,ax=ax[i], shrink=0.8, orientation='horizontal',
#     #                  label=f"{'fraction' if i in [3] else 'mm/h' if i in [4,5] else ''}")

# plt.savefig(f"figures/test_regrid_confronto.png")
# plt.close()



# mohc = f"{PATH_ERAINT}MOHC/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_1hr_201012010030-201012312330.nc"
# eth  = f"{PATH_ERAINT}ETH/CPM/uas/uas_ALP-3i_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200901010030-200912312330.nc"

# xr_mohc = xr.open_dataset(mohc).load()
# xr_eth  = xr.open_dataset(eth).sel(time="2009-08-31", method="nearest").isel(time= 16).load()

# # xr_mohc.longitude_bnds.min().item(),xr_mohc.longitude_bnds.max().item(),xr_mohc.latitude_bnds.min().item(),xr_mohc.latitude_bnds.max().item()
# xr_eth.lon_bnds.min().item(),xr_eth.lon_bnds.max().item(),xr_eth.lat_bnds.min().item(),xr_eth.lat_bnds.max().item()

# xr_mohc.rotated_latitude_longitude

# era52=xr.open_dataset("/mnt/data/lcesarini/test/air_temperature/era5-downscaled-over-italy-VHR-REA_IT_1989_2020_hourly.nc")



# fig,ax=plt.subplots(nrows=1,ncols=1,
#                     figsize=(8,8),constrained_layout=True, squeeze=True,
#                     subplot_kw={"projection":ccrs.PlateCarree()}, 

# )
# xr_eth.uas.plot(x='lon',y='lat')
# pcm=(era52.T_2M.max(dim='time')).plot.contourf(ax=ax,alpha=1,
#                 # transform=rot,
#                 transform=ccrs.RotatedPole(pole_longitude=-168.0, 
#                                            pole_latitude=47.0, 
#                                            central_rotated_longitude=0.0, 
#                                            globe=None),
#                 add_colorbar=True,
#                 # cmap=cmap, norm=norm,
#                 cbar_kwargs={"shrink":0.7,
#                             "orientation":"horizontal",
#                             "label":f"mm"
#                             }
#                 )
# ax.set_extent([-7,23,35,55])
# ax.coastlines()
# gl = ax.gridlines(
#     draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',#xlocs=[11],ylocs=[45,47]
# )
# plt.savefig(f"figures/test_regrid.png")
# plt.close()






