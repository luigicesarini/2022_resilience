#! /home/lcesarini/miniconda3/envs/my_xclim_env/bin/python
import os
os.environ['USE_PYGEOS'] = '0'
import sys
sys.path.append("/mnt/beegfs/lcesarini/2022_resilience")
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

os.chdir("/mnt/beegfs/lcesarini/2022_resilience")
from resilience.utils import *

"""

Time to run the bias correction for 8 models:
 ~ 90 minutes 

"""
"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-p","--period",
                    help="Which period to adjust the data",
                    required=True,default="VALIDATION",
                    choices=["TRAIN","VALIDATION"]  
                    )

parser.add_argument("-ref","--reference",
                    help="Which dataset to use for bias correction",
                    required=True,default="SPHERA",
                    choices=["SPHERA","STATIONS"]  
                    )

parser.add_argument("-s","--split",
                    help="How are the data split for fitting and evaluating the correction",
                    required=True,default="SEQUENTIAL",
                    choices=["RANDOM","SEQUENTIAL"]  
                    )
parser.add_argument("-a","--area",
                    help="Area over which the bias correction is applied",
                    default='triveneto',choices=["triveneto","northern_italy"]
                    )
parser.add_argument("-m","--model",
                    help="Which model to correct",
                    default='triveneto',choices=["CNRM","KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","ETH"]
                    )
parser.add_argument("-db","--debug",
                    help="Debugging some error",
                    action="store_true"
                    )
args = parser.parse_args()

lvl_f,lvl_i,lvl_q=get_levels()
cmap_f,cmap_i,cmap_q=get_palettes()

PATH_COMMON_DATA="/mnt/beegfs/lcesarini/DATA_FPS"
PATH_BIAS_CORRECTED = f"/mnt/beegfs/lcesarini/BIAS_CORRECTED/" 
# BM  = str(sys.argv[1])
# MDL = str(sys.argv[2])
REF = args.reference
ADJUST = args.period
SEASONS=['SON','DJF','MAM','JJA'] #,
SPLIT= args.split
AREA=args.area
list_mdl=[args.model]
list_mdl=["KNMI","ICTP","HCLIMcom","MOHC","CMCC","KIT","ETH"]
print(list_mdl)
DEBUG=args.debug
print(f"debug_:{DEBUG}")
if DEBUG:
    REF = "SPHERA"
    ADJUST = "VALIDATION"
    SEASONS=['JJA'] #
    SPLIT= "SEQUENTIAL"
    AREA="northern_italy"
"""
QM bias correction only on the wet hours. Otherwise the frequency of wet hours plays a crucial role
I split the fit in the first 5 years (2000-2004), and the test/adjustment on the following 5 years (2005-2009)
SPLIT BETWEEN TRAIN and ADJUST PERIOD
"""

#Create subroutine that based on SPLIT selects the years to use for bias correction
if SPLIT == "SEQUENTIAL":
    yrs_train = np.arange(2000,2005)
    yrs_valid = np.arange(2005,2010)
elif SPLIT == "RANDOM":
    yrs_train = np.sort(np.random.choice(np.arange(2000,2010),size=5,replace=False))
    yrs_valid = np.sort(np.setdiff1d(np.arange(2000,2010),yrs_train))

#LOAD DATA
if  REF == "SPHERA":
    list_ref_train=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{year}*") for year in yrs_train]


if (ADJUST=="TRAIN") & (REF == "SPHERA"):
    list_ref_adjust=list_ref_train
elif (ADJUST=="VALIDATION") & (REF == "SPHERA"):
    list_ref_adjust=[glob(f"{PATH_COMMON_DATA}/reanalysis/SPHERA/pr/*{year}*") for year in yrs_valid]

# mask_sa=xr.open_dataset("output/mask_study_area.nc")
if REF =="SPHERA":
    ref_train=xr.open_mfdataset([item for list in list_ref_train for item in list]).load()
    ref_adjust=xr.open_mfdataset([item for list in list_ref_adjust for item in list]).load()
    # sta_adjust=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_train])
    # max_sta=np.nanmax(sta_adjust.pr.values[:,:,:],axis=2)
elif REF == "STATIONS":
    ref_train=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_train]).load()
    if ADJUST=="TRAIN":
        ref_adjust=ref_train
    elif ADJUST=="VALIDATION":
        ref_adjust=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in yrs_valid]).load()
    all_sta=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{year}.nc" for year in np.arange(2000,2010)]).load()
    max_sta=np.nanmax(all_sta.pr.values[:,:,:],axis=2)

if AREA =='triveneto':
    x,y=np.where((max_sta > 0))
    x_,y_=np.where(np.isnan(max_sta))

    xy=np.concatenate([y.reshape(-1,1),x.reshape(-1,1)],axis=1)
    xy_=np.concatenate([y_.reshape(-1,1),x_.reshape(-1,1)],axis=1)
    assert xy.shape[0] == 172, f"Number of cells, different from number {xy.shape[0]}"
else:

    ref_train_tri=ref_train
    ref_adjust_tri=ref_adjust

    xy=create_list_coords(np.arange(ref_train_tri.longitude.values.shape[0]).reshape(-1,1),
                          np.arange(ref_train_tri.latitude.values.shape[0]).reshape(-1,1))
    
    
    xy_=np.where(np.isnan(np.nanmax(ref_train_tri.pr,axis=0)))
    
"""
DO THE BIAS CORRECTION FOR EACH CELL THAT CONTAINS A STATION, THUS:
- Comparison between Model and station
"""
print(f"Running:{mdl} {datetime.today().strftime('%d/%m/%Y %H:%M:%S')}")

list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
x=xr.open_mfdataset(list_mod_adjust[0]).load()

# mod_adjust.lat.values[:2] == ref_train.latitude.values[:2]

# print(x.lat[1]-x.lat[0])
if mdl == "ETH":
    from resilience.utils.fix_year_eth import fix_eth
    eth=fix_eth()
    # mod_train=eth.sel(time=slice("2000-01-01","2004-12-31")).load()
    mod_train=eth.sel(time=eth['time.year'].isin(yrs_train)).load()
    if ADJUST=="TRAIN":
        mod_adjust=mod_train
    elif ADJUST=="VALIDATION":
        # mod_adjust=eth.sel(time=slice("2005-01-01","2009-12-31")).load()
        mod_adjust=eth.sel(time=eth['time.year'].isin(yrs_valid)).load()
    
else:   
    list_mod_train=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_train]
    if ADJUST == "TRAIN":
        list_mod_adjust=list_mod_train
        mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
        mod_adjust=mod_train
    elif ADJUST == "VALIDATION":
        list_mod_adjust=[glob(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/{mdl}/CPM/pr/*{year}*") for year in yrs_valid]
        mod_train=xr.open_mfdataset([item for list in list_mod_train for item in list]).load()
        mod_adjust=xr.open_mfdataset([item for list in list_mod_adjust for item in list]).load()

# print(f"{mdl}")
# print(f"dates calibration:{mod_train.time.values[0]}-{mod_train.time.values[-1]}")
# print(f"dates adjustment:{mod_adjust.time.values[0]}-{mod_adjust.time.values[-1]}")


# for seas in tqdm(SEASONS,total=len(SEASONS)): 
for seas in tqdm(SEASONS,total=len(SEASONS)): 
    list_xr_qdm=[]
    list_xr_eqm=[]

    """
    CHECK A BIG CHUNK OF CODE TO DEBUG HEAVY PREC for DJF
    """

    for id_coord in xy:
        # print(id_coord)
        if REF == "SPHERA":
            if AREA=="triveneto":
                lon=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                lat=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                ref_train_sc=ref_train.sel(longitude=lon,latitude=lat,method='nearest').\
                    where(ref_train.sel(longitude=lon,latitude=lat,method='nearest').pr > 0.1).assign_attrs(units="mm/hr")
                ref_adjust_sc=ref_adjust.sel(longitude=lon,latitude=lat,method='nearest').\
                    where(ref_adjust.sel(longitude=lon,latitude=lat,method='nearest').pr > 0.1).assign_attrs(units="mm/hr")
            elif AREA=="northern_italy":
                lon,lat=id_coord[0],id_coord[1]
                
                ref_train_sc=ref_train.isel(longitude=lon,latitude=lat).\
                    where(ref_train.isel(longitude=lon,latitude=lat).pr > 0.1).assign_attrs(units="mm/hr")
                ref_adjust_sc=ref_adjust.isel(longitude=lon,latitude=lat).\
                    where(ref_adjust.isel(longitude=lon,latitude=lat).pr > 0.1).assign_attrs(units="mm/hr")
        elif REF == "STATIONS":
            lon=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
            lat=ref_train.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
            ref_train_sc=ref_train.sel(lon=lon,lat=lat,method='nearest').\
                where(ref_train.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
            ref_adjust_sc=ref_adjust.sel(lon=lon,lat=lat,method='nearest').\
                where(ref_adjust.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
        #Filter only WET hours for single cell
        if REF == "SPHERA":                
            mod_train_sc=mod_train.isel(lon=lon,lat=lat).\
                where(mod_train.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
            # sta_train_sc=sta_train.where(sta_train.pr > 0.2).assign_attrs(units="mm/hr").load()

            mod_adjust_sc=mod_adjust.isel(lon=lon,lat=lat).\
                where(mod_adjust.isel(lon=lon,lat=lat).pr > 0.1).assign_attrs(units="mm/hr")
        else:
            mod_train_sc=mod_train.sel(lon=lon,lat=lat,method='nearest').\
                where(mod_train.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")
            # sta_train_sc=sta_train.where(sta_train.pr > 0.2).assign_attrs(units="mm/hr").load()

            mod_adjust_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').\
                where(mod_adjust.sel(lon=lon,lat=lat,method='nearest').pr > 0.2).assign_attrs(units="mm/hr")

        #WRITE A CONDITION TO CHECK IF DATA ARE IN CHUNKS"""
        #ASSIGN UNITS
        # Qs=np.arange(0,1,0.0001) 
        Qs=np.arange(0.1,1,0.001)

        QDM = sdba.QuantileDeltaMapping.train(
        # QM = sdba.EmpiricalQuantileMapping.train(
            ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr"),
            mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr, 
            nquantiles=Qs, 
            group="time", 
            kind="+"
        )

        EQM = sdba.EmpiricalQuantileMapping.train(
            ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr"),
            mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr, 
            nquantiles=Qs, 
            group="time", 
            kind="+"
        )
        if REF=="SPHERA":
            mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest").drop_vars(['surface'])
            mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr, 
                                            extrapolation="constant", interp="nearest").drop_vars(['surface'])
            list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).drop_vars(['longitude','latitude']).\
                            expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
            list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).drop_vars(['longitude','latitude']).\
                            expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
        elif REF=="STATIONS":
            mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                            extrapolation="constant", interp="nearest")
            # mod_adjust_sc_eqm = EQM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
            #                                 extrapolation="constant", interp="nearest")
            mod_adjust_sc_qdm = QDM.adjust(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr,
                                            extrapolation="constant", interp="nearest")
            
            # xx=mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr
            # xr.where(~np.isnan(xx),xx,0)
            # x1=EQM.adjust(mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr,
            #                                 extrapolation="constant", interp="nearest")
            
            # x2=EQM.adjust(xr.where(~np.isnan(xx),xx,0).assign_attrs(units="mm/hr"),
            #                                 extrapolation="constant", interp="nearest")
            

            # x1.max() 
            # x2.max()
            ref_train_sc  = get_season(ref_train_sc,seas)
            ref_adjust_sc = get_season(ref_adjust_sc,seas)
            mod_train_sc  = get_season(mod_train_sc,seas)
            mod_adjust_sc = get_season(mod_adjust_sc,seas)



        if DEBUG:
            list_xr_qdm.append(xr.where(np.isnan(mod_adjust_sc_qdm),0,mod_adjust_sc_qdm).\
                            expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
            list_xr_eqm.append(xr.where(np.isnan(mod_adjust_sc_eqm),0,mod_adjust_sc_eqm).\
                            expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))
            
            if xy_[0].shape[0] > 0:
                for id_coord in xy_:
                    if REF == "SPHERA":
                        lon=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                        lat=sta_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                    elif REF == "STATIONS":
                        lon=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lon.item()
                        lat=ref_adjust.isel(lon=id_coord[0],lat=id_coord[1]).lat.item()
                
                    na_sta_sc=mod_adjust.sel(lon=lon,lat=lat,method='nearest').isel(time=mod_adjust['time.season'].isin(seas))
                    list_xr_qdm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])
                    list_xr_eqm.append(xr.where(np.isnan(na_sta_sc),0,np.nan).expand_dims(dim={"lat": 1,"lon":1})[["time","lat","lon","pr"]])

    ds_adj_qdm=xr.combine_by_coords(list_xr_qdm)
    ds_adj_eqm=xr.combine_by_coords(list_xr_eqm)
    
    if not os.path.exists(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/"):
        os.makedirs(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/")

    if not os.path.exists(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/"):
        os.makedirs(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/")

    # if ADJUST=="TRAIN":
    #     ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
    #     ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/{mdl}_CORR_{REF}_2000_2004_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}.nc")
    # elif ADJUST=="VALIDATION":
    ds_adj_qdm.to_netcdf(f"{PATH_BIAS_CORRECTED}/QDM/{mdl}/pr/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")
    ds_adj_eqm.to_netcdf(f"{PATH_BIAS_CORRECTED}/EQM/{mdl}/pr/{mdl}_CORR_{REF}_{seas}_Q{Qs.shape[0]}_{SPLIT}_{ADJUST}_{AREA}.nc")

# fig=plt.figure(figsize=(16,12))
# ax=plt.axes(projection=ccrs.PlateCarree())
# pcm=ds_adj.scen.rio.write_crs("epsg:4326").quantile(q=0.999,dim='time').plot.pcolormesh(levels=lvl_q,
#                             cmap=cmap_q,ax=ax,
#                             add_colorbar=False)
# shp_triveneto.boundary.plot(edgecolor='red',ax=ax,linewidths=1.5)
# ax.set_title(f"Bias Corrected Value of MOHC",fontsize=20)
# ax.set_extent([10.39,13.09980774,44.70745754,47.09988785])
# gl = ax.gridlines(
#         draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
#     )
# gl.xlocator = mpl.ticker.FixedLocator([10.5, 11.5, 12.5])
# gl.ylocator = mpl.ticker.FixedLocator([45, 46, 47])
# gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
# gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight':'bold'}
# cbar=fig.colorbar(pcm, ax=ax, 
#                     extend='both', 
#                     orientation='vertical',
#                     shrink=1,
#                     pad = 0.075)
# cbar.ax.tick_params(labelsize=30)
# cbar.ax.set_ylabel('[%]',fontsize=25,rotation=0)
# fig.suptitle("Heavy precipitation (p99.9)",fontsize=30)
# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.LAKES)
# ax.add_feature(cfeature.RIVERS)
# ax.coastlines()
# plt.savefig(f"figures/bias_corrected_mohc.png")
# plt.close()
            
    
    # Qsplot=np.arange(0.999,0.999999,.0001)
    # # Qsplot=np.arange(0.5,1,.01)
    # fig,axs=plt.subplots(1,2,figsize=(12,4))
    # axs[0].plot(np.nanquantile(ref_train_sc.pr, q=Qsplot[:]),'-^',label="ref train",)
    # axs[1].plot(np.nanquantile(ref_adjust_sc.pr, q=Qsplot[:]),'-^',label="ref adj")
    # axs[0].plot(np.nanquantile(mod_train_sc_adj, q=Qsplot[:]),'-*',label="mod train bc")
    # axs[0].plot(np.nanquantile(mod_train_sc.pr, q=Qsplot[:]),'-*',label="mod train")
    # axs[1].plot(np.nanquantile(mod_adjust_sc_adj,q=Qsplot[:]),'--d',label="mod adj bc")
    # axs[1].plot(np.nanquantile(mod_adjust_sc.pr,q=Qsplot[:]),'--d',label="mod adj")
    # axs[0].legend()
    # axs[1].legend()
    # axs[0].set_xlabel("Quantile")
    # axs[1].set_xlabel("Quantile")
    # axs[0].set_xticks(np.arange(0,Qsplot[:].shape[0],1),Qsplot[:].round(4))
    # axs[1].set_xticks(np.arange(0,Qsplot[:].shape[0],1),Qsplot[:].round(4))
    # axs[0].set_ylabel("Precipitation [mm/hr]")
    # axs[1].set_ylabel("Precipitation [mm/hr]")
    # plt.title(f"Quantile mapping for {lon:.3f} {lat:.3f}")
    # axs[0].set_xlim([Qsplot[:].shape[0]-(min(Qsplot[:].shape[0],20)),Qsplot[:].shape[0]])
    # axs[1].set_xlim([Qsplot[:].shape[0]-(min(Qsplot[:].shape[0],20)),Qsplot[:].shape[0]])
    # plt.savefig("histo_ref_mod.png")
    # plt.close()



    # """PLOT THE METRICS WITH THE CORRECTED VALUES OF THE MODEL"""
    # dict={}
    # dict.update({"metrics_mod_adjust":compute_metrics(get_season(mod_adjust_sc,season="JJA"),meters=True,quantile=0.999)})
    # dict.update({"metrics_mod_adjust_bc":compute_metrics(get_season(mod_adjust_sc_adj,season="JJA"),meters=True,quantile=0.999)})
    # dict.update({"metrics_ref_adjust":compute_metrics(get_season(ref_adjust_sc,season="JJA"),meters=True,quantile=0.999)})

    # dict.keys()
    # x=[dict['metrics_mod_adjust'][0].item(),dict['metrics_mod_adjust'][1].item(),dict['metrics_mod_adjust'][2].item(),dict['metrics_mod_adjust'][3].item()]
    # y=[dict['metrics_mod_adjust_bc'][0].item(),dict['metrics_mod_adjust_bc'][1].item(),dict['metrics_mod_adjust_bc'][2].item(),dict['metrics_mod_adjust_bc'][3].item()]
    # z=[dict['metrics_ref_adjust'][0].item(),dict['metrics_ref_adjust'][1].item(),dict['metrics_ref_adjust'][2].item(),dict['metrics_ref_adjust'][3].item()]

    # Qsplot=np.arange(0.999,0.999999,.0001)
    # # Qsplot=np.arange(0.5,1,.01)
    # plt.plot(["Frequency","Mean Intensity","Variance Intensity","Heavy Prec"],x,'^',label="Model")
    # plt.plot(["Frequency","Mean Intensity","Variance Intensity","Heavy Prec"],y,'d',label="Model BC")
    # plt.plot(["Frequency","Mean Intensity","Variance Intensity","Heavy Prec"],z,'*',label="Reference")
    # plt.legend()
    # plt.xlabel("Metric")
    # plt.xticks(np.arange(0,4),["Frequency","Mean Intensity","Variance Intensity","Heavy Prec"])
    # plt.ylabel("Precipitation [mm/hr]")
    # plt.title(f"Comparison of metrics for {lon:.3f} {lat:.3f}")
    # plt.savefig("histo_ref_mod.png")
    # plt.close()
            # ax=plt.axes()
            # ax.plot(np.sort(ref_train_sc.pr[~np.isnan(ref_train_sc.pr)]),
            #         np.arange(ref_train_sc.pr[~np.isnan(ref_train_sc.pr)].shape[0])/ref_train_sc.pr[~np.isnan(ref_train_sc.pr)].shape[0],
            #         '-',label='reference train')
            # # ax.plot(np.sort(ref_adjust_sc.pr[~np.isnan(ref_adjust_sc.pr)]),
            # #         np.arange(ref_adjust_sc.pr[~np.isnan(ref_adjust_sc.pr)].shape[0])/ref_adjust_sc.pr[~np.isnan(ref_adjust_sc.pr)].shape[0],
            # #         '-',label='reference adjust')
            # ax.plot(np.sort(mod_train_sc.pr[~np.isnan(mod_train_sc.pr)]),
            #         np.arange(mod_train_sc.pr[~np.isnan(mod_train_sc.pr)].shape[0])/mod_train_sc.pr[~np.isnan(mod_train_sc.pr)].shape[0],
            #         '-',label='Model train')
        
            # ax.plot(np.sort(mod_adjust_sc.pr[~np.isnan(mod_adjust_sc.pr)]),
            #         np.arange(mod_adjust_sc.pr[~np.isnan(mod_adjust_sc.pr)].shape[0])/mod_adjust_sc.pr[~np.isnan(mod_adjust_sc.pr)].shape[0],
            #         '-',label='Model adjustment')
            # ax.plot(np.sort(mod_adjust_sc_eqm[~np.isnan(mod_adjust_sc_eqm)]),
            #         np.arange(mod_adjust_sc_eqm[~np.isnan(mod_adjust_sc_eqm)].shape[0])/mod_adjust_sc_eqm[~np.isnan(mod_adjust_sc_eqm)].shape[0],
            #         '-',label='EQM adjusted')
            # # ax.plot(np.sort(mod_adjust_sc_qdm[~np.isnan(mod_adjust_sc_qdm)]),
            # #         np.arange(mod_adjust_sc_qdm[~np.isnan(mod_adjust_sc_qdm)].shape[0])/mod_adjust_sc_qdm[~np.isnan(mod_adjust_sc_qdm)].shape[0],
            # #         '-',label='QDM adjusted')
            # ax.set_title(f"{mdl} {seas} {lon} {lat}")
            # ax.set_ylim([0.85,1])
            # ax.legend()
            # plt.show()

            # ll=\
            #     [np.nanquantile(ref_train_sc.pr,q=[0.999]),
            #      np.nanquantile(ref_adjust_sc.pr,q=[0.999]),
            #      np.nanquantile(mod_train_sc.pr,q=[0.999]),
            #      np.nanquantile(mod_adjust_sc.pr,q=[0.999]),
            #      np.nanquantile(mod_adjust_sc_eqm,q=[0.999])]
            
            # q_ma=np.nanquantile(mod_adjust_sc.pr,0.999)
            # score_tr=stats.percentileofscore(a=mod_train_sc.pr,score=np.nanquantile(mod_adjust_sc.pr,0.999), nan_policy='omit')
            # q_mt=np.nanquantile(mod_train_sc.pr,score_tr/100)
            # q_ot=np.nanpercentile(ref_train_sc.pr,99.20212765957447)
            

            # dfdf=pd.DataFrame(ll)
            # dfdf.index=['ref_train','ref_adjsut','mod_train','mod_adjust','mod_adjust_eqm']
            # dfdf.columns=['99.9th']
            # # print(f"""
            # #     {mdl} - {seas}\n
            # #     {dfdf}
            # #       """
            # # )
            # bia_ori=(dfdf.iloc[3] - dfdf.iloc[1]) / dfdf.iloc[1]
            # bia_eqm=(dfdf.iloc[4] - dfdf.iloc[1]) / dfdf.iloc[1]

            # if np.abs(bia_ori.item()) < np.abs(bia_eqm.item()):
            #     print(f"Bias increases after correction")
            #     sb.run(f"echo {mdl} {seas} {lon} {lat} {q_ma:.2f} {q_mt:.2f} {q_ot:.2f} {np.nanquantile(mod_train_sc.pr,q=[0.999]).item():.2f} {np.nanquantile(ref_train_sc.pr,q=[0.999]).item():.2f} {np.nanquantile(ref_adjust_sc.pr,q=[0.999]).item():.2f} >> output/bias_increases.txt",shell=True)



            # ll=\
            #     [np.nanquantile(ref_train_sc.pr,q=[0.9,0.99,0.999]),
            #      np.nanquantile(ref_adjust_sc.pr,q=[0.9,0.99,0.999]),
            #      np.nanquantile(mod_train_sc.pr,q=[0.9,0.99,0.999]),
            #      np.nanquantile(mod_adjust_sc.pr,q=[0.9,0.99,0.999]),
            #      np.nanquantile(mod_adjust_sc_eqm,q=[0.9,0.99,0.999]),
            #      np.nanquantile(mod_adjust_sc_qdm,q=[0.9,0.99,0.999])]
            
            # dfdf=pd.DataFrame(ll)
            # dfdf.index=['ref_train','ref_adjust','mod_train','mod_adjust','mod_adjust_eqm','mod_adjust_qdm']
            # dfdf.columns=['90th','99th','99.9th']
            # print(f"""
            #     {mdl} - {seas}\n
            #     {dfdf}
            #       """
            # )

            # from bias_correction import XBiasCorrection

            # ds = xr.Dataset({'model_data':mod_train_sc.pr,
            #                  'obs_data':ref_train_sc.pr, 
            #                  'sce_data':mod_adjust_sc.pr})
            

            # bc = XBiasCorrection(ds['model_data'],
            #                     ds['obs_data'],
            #                     ds['sce_data'])
            
            # corrected = bc.correct(method='basic_quantile')
            
            # list_xr_eqm.append(xr.where(np.isnan(corrected),0,corrected).\
            #                 expand_dims(dim={"lat": 1,"lon":1}).to_dataset(name='pr'))

            # ds_adj_eqm=xr.combine_by_coords(list_xr_eqm).sel(time=slice("2005-01-02","2009-12-31"))

            # ds_adj_eqm = ds_adj_eqm.sel(time=ds_adj_eqm['time.season'].isin(seas))


            # ds_adj_eqm.pr.quantile(q=0.999,dim='time')

            # plot_panel_rotated(
            #                 (12,8),1,2,
            #                 [mod_adjust.pr.quantile(q=0.999,dim='time'),
            #                 ds_adj_eqm.pr.quantile(q=0.999,dim='time')],
            #                 name_fig=f"map_QDM_ensemble_{mdl}_{ADJUST}_{seas}_{AREA}",
            #                 list_titles=[f"99.9th percentile {mdl} {ADJUST} {seas} {AREA}",f"99.9th percentile {mdl} {ADJUST} {seas} {AREA}"],
            #                 levels=[lvl_q,lvl_q],cmap=[cmap_q,cmap_q],
            #                 name_metric=["mm/h","mm/h"],
            #                 SET_EXTENT=False,
            #                 SAVE=False
            #                 )


            # np.nanargmin(mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr.values)
            # mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr.isel(time=9504)
            # mod_adjust_sc_eqm.isel(time=9504)
            

            # ref_=ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr").values
            # ref_=ref_[~np.isnan(ref_)]

            # ref_adj=ref_adjust_sc.isel(time=ref_adjust_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr").values
            # ref_adj=ref_adj[~np.isnan(ref_adj)]

            # mod_train_=mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr.values
            # mod_train_=mod_train_[~np.isnan(mod_train_)]

            # mod_=mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr.values
            # mod_=mod_[~np.isnan(mod_)]

            # mod_adj=mod_adjust_sc_eqm.values
            # mod_adj=mod_adj[~np.isnan(mod_adj)]
            
            # mod_adj_qdm=mod_adjust_sc_qdm.values
            # mod_adj_qdm=mod_adj_qdm[~np.isnan(mod_adj_qdm)]

            # ref_.shape,mod_.shape,mod_adj.shape

            # plt.plot(np.sort(ref_),np.arange(ref_.shape[0])/ref_.shape[0],label='reference train')
            # plt.plot(np.sort(ref_adj),np.arange(ref_adj.shape[0])/ref_adj.shape[0],label='reference adjust')
            # plt.plot(np.sort(mod_train_),np.arange(mod_train_.shape[0])/mod_train_.shape[0],label='model train')
            # plt.plot(np.sort(mod_),np.arange(mod_.shape[0])/mod_.shape[0],label='model validation')
            # plt.plot(np.sort(mod_adj),np.arange(mod_adj.shape[0])/mod_adj.shape[0],label='model adj validationEQM')
            # plt.plot(np.sort(mod_adj_qdm),np.arange(mod_adj_qdm.shape[0])/mod_adj_qdm.shape[0],label='model adj validationQDM')
            # plt.legend()
            # plt.axes.Axes.set_xlim([3,5])
            # plt.show()

                        # sta_adjust_sc=sta_adjust.where(sta_adjust.pr > 0.2).assign_attrs(units="mm/hr").load()

        # plt.hist(mohc_rg.where((mohc_rg.pr > 0) & (mohc_rg.pr < 0.1)).pr.values.reshape(-1),bins=55,
        #          label="MOHC",ec='blue',fc='none',lw=2)
        # plt.hist(sphera.where((sphera.pr > 0) & (sphera.pr < 0.1)).pr.values.reshape(-1),bins=55,
        #          label="SPHERA",ec='red',fc='none',lw=2)
        # plt.savefig("histo_mohc.png")
        # plt.close()

        # sta_cal=sta_cal_all.sel(time=sta_cal_all['time.season'].isin(seas))
        # sta_adj=sta_adj_all.sel(time=sta_adj_all['time.season'].isin(seas))


        # x1=ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr")
        # x5=ref_adjust_sc.isel(time=ref_adjust_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr")
        # x2=mod_train_sc.isel(time=mod_train_sc['time.season'].isin(seas)).pr
        # x4=mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).pr
        # x3=get_triveneto(sph_cal,sta_cal_999).isel(lon=id_coord[0],lat=id_coord[1])
        # x6=get_triveneto(sph_adj,sta_cal_999).isel(lon=id_coord[0],lat=id_coord[1])
        # x3=np.where(x3.pr > 0.1, x3.pr,np.nan)
        # x6=np.where(x6.pr > 0.1, x6.pr,np.nan)
        # print("STA TRAIN",np.where(~np.isnan(x1),1,0).sum())
        # print("STA ADJUS",np.where(~np.isnan(x5),1,0).sum())
        # print("CNRM TRAIN",np.where(~np.isnan(x2),1,0).sum())
        # print("CNRM ADJUS",np.where(~np.isnan(x4),1,0).sum())
        # print("SPH TRAIN",np.where(~np.isnan(x3),1,0).sum())
        # print("SPH ADJUST",np.where(~np.isnan(x6),1,0).sum())

        # np.nanquantile(x4,q=0.999)
        # stats.mstats.mquantiles(x4[~np.isnan(x4)],prob=0.999,alphap=0,betap=1)
        

        # find the model value of adjustment in calibration period
        # score=stats.percentileofscore(x2[~np.isnan(x2)],np.nanquantile(x4,q=0.999)) / 100
        
        # # Compute value of reference (one for stations and one for sphera) at the same percentile of the model
        # # This is the adjusted valued of tjhe model in the adjusted period
        # sc_sta=np.nanquantile(x1[~np.isnan(x1)],min(1-1e-6,score))
        # sc_sph=np.nanquantile(x3[~np.isnan(x3)],min(1-1e-6,score))

        # # Check the correction
        # #Bias of original model in adjustment vs the 2 references
        # bias_ori_sta=(np.nanquantile(x4,q=0.999)-np.nanquantile(x5,q=0.999)) / np.nanquantile(x5,q=0.999) * 100
        # bias_ori_sph=(np.nanquantile(x4,q=0.999)-np.nanquantile(x6,q=0.999)) / np.nanquantile(x6,q=0.999) * 100
        # #Bias of "adjusted" model in adjustment vs the 2 references
        # bias_adj_sta=(np.nanquantile(sc_sta,q=0.999)-np.nanquantile(x5,q=0.999)) / np.nanquantile(x5,q=0.999) * 100
        # bias_adj_sph=(np.nanquantile(sc_sph,q=0.999)-np.nanquantile(x6,q=0.999)) / np.nanquantile(x6,q=0.999) * 100

        # # print(f"Station: {bias_ori_sta:.2f} =====> {bias_adj_sta:.2f}\nSPHERA: {bias_ori_sph:.2f} =====> {bias_adj_sph:.2f}")
        # if (np.abs(bias_ori_sta) < np.abs(bias_adj_sta)) & (int(score) == 1):
        #     print(f"Original bias smaller for stations at:{id_coord[0]},{id_coord[1]}\n{score}")
        #     plot()
        # elif (np.abs(bias_ori_sta) > np.abs(bias_adj_sta)) & (int(score) == 1):
        #     print(f"Original bias higher for stations at:{id_coord[0]},{id_coord[1]}\nbut score {score}")
        #     plot()


        # if np.abs(bias_ori_sph) < np.abs(bias_adj_sph):
        #     print(f"Original bias smaller for SPHERA at:{id_coord[0]},{id_coord[1]}\n{score}")

        # def plot():
        #     fig,ax=plt.subplots(1,1,figsize=(14,12))
        #     ax.plot(np.round(np.sort(x1[~np.isnan(x1)].values),2),np.arange(1,x1[~np.isnan(x1)].shape[0]+1)/(x1[~np.isnan(x1)].shape[0]+1),"-o",color='green',label="STATION_TRAIN")
        #     ax.plot(np.round(np.sort(x5[~np.isnan(x5)].values),2),np.arange(1,x5[~np.isnan(x5)].shape[0]+1)/(x5[~np.isnan(x5)].shape[0]+1),"-+",color='lightgreen',label="STATION_ADJUST")
        #     ax.plot(np.sort(x2[~np.isnan(x2)].values),np.arange(1,x2[~np.isnan(x2)].shape[0]+1)/(x2[~np.isnan(x2)].shape[0]+1),"-o",color='red',label='CNRM_TRAIN')
        #     ax.plot(np.sort(x4[~np.isnan(x4)].values),np.arange(1,x4[~np.isnan(x4)].shape[0]+1)/(x4[~np.isnan(x4)].shape[0]+1),"-+",color='indianred',label='CNRM_ADJUST')
        #     ax.plot(np.sort(x3[~np.isnan(x3)]),np.arange(1,x3[~np.isnan(x3)].shape[0]+1)/(x3[~np.isnan(x3)].shape[0]+1),"-o",color='blue',label='SPHERA_TRAIN')
        #     ax.plot(np.sort(x6[~np.isnan(x6)]),np.arange(1,x6[~np.isnan(x6)].shape[0]+1)/(x6[~np.isnan(x6)].shape[0]+1),"-+",color='dodgerblue',label='SPHERA_ADJUST')
        #     # plt.scatter(x=np.nanquantile(x4,q=0.999),y=0.999,marker='+',label="999_MOD_ADJ")
        #     # plt.scatter(x=np.nanquantile(x2,q=0.999),y=0.999,marker='+',label="999_MOD_TRAIN")
        #     # plt.scatter(x=np.nanquantile(x1,q=0.999),y=0.999,marker='*',label="999_STA_TRAIN")
        #     # plt.scatter(x=np.nanquantile(x3,q=0.999),y=0.999,marker='^',label="999_SPH_TRAIN")
        #     ax.set_xlim([5,np.nanmax(np.concatenate([x1,x2,x3,x4,x5,x6])) * 1.1])
        #     ax.set_ylim([0.95,1.05])
        #     fig.legend()
        #     plt.show()


        # EQM.ds.quantiles[-1]
        # EQM.ds.isel(quantiles=-1)

        # xxx=ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr")

        # np.quantile(xxx[~np.isnan(xxx)],0.999)
        # pd.DataFrame([np.arange(11040),np.sort(xxx)]).transpose()
        # np.nanquantile(ref_train_sc.isel(time=ref_train_sc['time.season'].isin(seas)).pr.assign_attrs(units="mm/hr"),q=0.9896640826873384)

        # stats.percentileofscore(xxx[~np.isnan(xxx)],mod_adjust_sc.isel(time=mod_adjust_sc['time.season'].isin(seas)).isel(time=np.arange(10460,10461)).pr.item())
        # np.nanquantile(xxx[~np.isnan(xxx)],q=0.9896640826873384)
        # mod_train_sc_adj = QM.adjust(mod_train_sc.pr, extrapolation="constant", interp="nearest").drop_vars(['surface'])
