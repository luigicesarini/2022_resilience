#!/home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from glob import glob
from tqdm import tqdm
from datetime import datetime, timedelta
# from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

from resilience.utils import *

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"

cmap_f,cmap_i,cmap_q=get_palettes()
lvl_f,lvl_i,lvl_q=get_levels()


if __name__=="__main__":
    # ds=xr.open_dataset("/mnt/data/commonData/OBSERVATIONS/ITALY/gripho-v1_1h_TSmin30pct_2001-2016.nc")
    # print(ds.lon[1]-ds.lon[0])


    # eth__rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/ETH/CPM/pr/ETH_ECMWF-ERAINT_200901010030_200912312330.nc"]).load()
    # eth__rg=xr.open_mfdataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200901010030_200912312330.nc").load()
    # eth__rg=xr.open_dataset("/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200801010030_200812312330.nc").load()


    # eth__rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/ETH/CPM/pr/ETH_ECMWF-ERAINT_{yr}01010030_{yr}12312330.nc" for yr in np.arange(2000,2010)]).load()
    mohc_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/MOHC/CPM/pr/MOHC_ECMWF-ERAINT_*.nc").load()
    ictp_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ICTP/CPM/pr/ICTP_ECMWF-ERAINT_*.nc").load()
    hcli_rg=xr.open_mfdataset(f"{PATH_COMMON_DATA}/HCLIMcom/CPM/pr/HCLIMcom_ECMWF-ERAINT_*.nc").load()
    cnrm_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/CNRM/CPM/pr/CNRM_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()
    knmi_rg=xr.open_mfdataset([f"{PATH_COMMON_DATA}/KNMI/CPM/pr/KNMI_ECMWF-ERAINT_{year}01010030-{year}12312330.nc" for year in np.arange(2000,2010)]).load()

    # cmcc_remap=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/cmcc/remap/reanalysis/pr/*.nc").load()
    gripho=xr.open_dataset("/mnt/data/lcesarini/gripho_3km.nc",chunks={'time':365})
    gri_per=gripho.isel(time=gripho['time.year'].isin(np.arange(2000,2010)))
    gri_per=gri_per.load()

    for SEAS in tqdm(['JJA'],total=3):

        # f,i,v,q=compute_metrics(get_season(gri_per,SEAS))
        

        # f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_f.nc")
        # i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_i.nc")
        # v.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_v.nc")
        # q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_q.nc")
        f=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_f.nc")
        i=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_i.nc")
        v=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_v.nc")
        q=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/GRIPHO_ORIGINAL_q.nc")

        plot_panel_rotated(
            figsize=(15,5.5),
            nrow=1,ncol=3,
            list_to_plot=[f.pr * 100,i.pr,q.isel(quantile=0).pr],
            name_fig=f"PANEL_GRIPHO_ORIGINAL_{SEAS}",
            list_titles=["Frequency","Intensity","Heavy Prec (99.9)"],
            levels=[np.arange(0,21,2),lvl_i,lvl_q],
            suptitle=f"GRIPHO original of for {SEAS}",
            name_metric=["[%]","[mm/hr]","[mm/hr]"],
            SET_EXTENT=False,
            cmap=[cmap_f,cmap_i,cmap_q],
            transform=ccrs.LambertAzimuthalEqualArea(central_latitude=52, central_longitude=10,
                                                    false_easting=4321000, false_northing=3210000),
        )            

    from scipy import stats
    mohc=stats.percentileofscore(mohc_rg.pr.values.reshape(-1,), 0.1,nan_policy='omit')
    ictp=stats.percentileofscore(ictp_rg.pr.values.reshape(-1,), 0.1,nan_policy='omit')
    hcli=stats.percentileofscore(hcli_rg.pr.values.reshape(-1,), 0.1,nan_policy='omit')

    cnrm=stats.percentileofscore(cnrm_rg.pr.values.reshape(-1,), 0.1,nan_policy='omit')
    stats.percentileofscore(cnrm_rg.pr.values.reshape(-1,), 3.212085e-29,nan_policy='omit')
    knmi=stats.percentileofscore(knmi_rg.pr.values.reshape(-1,)[knmi_rg.pr.values.reshape(-1,) >= 3.212085e-29], 0.1,nan_policy='omit')
    # cdo showtimestamp /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200901010030_200912312330.nc
    """
    Fix of time issues in data of 2009 and addition to 2000-2008
    """

    eth=xr.open_mfdataset([f'/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_{year}01010030_{year}12312330.nc' for year in range(2000,2009)])

    time_2009=np.arange(np.datetime64('2009-01-01T00:33:00.000000000'),
                        np.datetime64('2010-01-01T00:33:00.000000000'),
                        timedelta(hours=1))

    time_bnds_0=np.arange(np.datetime64('2009-01-01T00:00:00.000000000'),
                        np.datetime64('2010-01-01T00:00:00.000000000'),
                        timedelta(hours=1))

    time_bnds_1=np.arange(np.datetime64('2009-01-01T01:00:00.000000000'),
                        np.datetime64('2010-01-01T01:00:00.000000000'),
                        timedelta(hours=1))

    assert time_bnds_0.shape==time_bnds_1.shape,"Timess probleeemmm"

    time_bnds=np.concatenate([time_bnds_0.reshape(-1,1),time_bnds_1.reshape(-1,1)],axis=1)

    fn='/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200901010030_200912312330.nc'

    # xr.load_dataset(fn, engine='cfgrib')


    ds=nc.Dataset(fn)

    # eth_newt_eth=xr.DataArray(ds.variables['pr'][:],
    #                             coords={
    #                                 'time':time_2009,
    #                                 'lat':eth.lat,
    #                                 'lon':eth.lon,

    #                                 },
    #                             dims={
    #                                 "time":time_2009.shape[0],
    #                                 "lat":eth.lat.shape[0],
    #                                 "lon":eth.lon.shape[0]
    #                             }
    #                             )

    # eth_newt_eth=eth_newt_eth.expand_dims(dim={"bnds": 2})
    # eth_newt_eth.shape
    # ds_eth_new=eth_newt_eth.to_dataset(name='pr',promote_attrs=True)
    # ds_eth_new.assign(time_bnds=((ds_eth_new.time,ds_eth_new.bnds),time_bnds))


    ds = xr.Dataset(
        data_vars=dict(
            pr=(["time", "lat", "lon"], ds.variables['pr'][:]),
            time_bnds=(["time", "bnds"], time_bnds),
        ),
        coords=dict(
            lon=eth.lon,
            lat=eth.lat,
            time=time_2009,
        ),
        attrs=eth.attrs,
    )




    all_eth=xr.concat([eth,ds],dim='time')
    eth__rg=all_eth.load()

    print(eth__rg)
    #Remove 20210 from mohc and ictp
    mohc_rg=mohc_rg.isel(time=mohc_rg["time.year"].isin(np.arange(2000,2010)))
    ictp_rg=ictp_rg.isel(time=ictp_rg["time.year"].isin(np.arange(2000,2010)))

    name_models=['ETH','MOHC','ICTP','HCLIMcom','CNRM','KNMI']

    array_model=[eth__rg,mohc_rg,ictp_rg,hcli_rg,cnrm_rg,knmi_rg]

    assert np.all([x.time.shape == mohc_rg.time.shape for x in array_model]), "Time differs among modelssss" 

    # assert cmcc_remap.time.shape == mohc_rg.time.shape, "Very bad"

    WH=True


    for SEAS in ['SON','MAM','JJA','DJF']:
        dict_metrics={}

        for name,mdl in tqdm(zip(name_models,array_model), total=len(array_model)):
            
            if WH:
                dict_0={name:compute_metrics(get_season(mdl,season=SEAS),meters=True,quantile=0.999,wethours=WH)}
                
                dict_metrics.update(dict_0)
            else:
                dict_0={name:compute_metrics(get_season(mdl,season=SEAS),meters=True,quantile=0.999)}
                
                dict_metrics.update(dict_0)


        mean_f=(dict_metrics['ETH'][0]+dict_metrics['MOHC'][0]+dict_metrics['ICTP'][0]+
                dict_metrics['HCLIMcom'][0]+dict_metrics['CNRM'][0]+dict_metrics['KNMI'][0]) / 6
        mean_i=(dict_metrics['ETH'][1]+dict_metrics['MOHC'][1]+dict_metrics['ICTP'][1]+
                dict_metrics['HCLIMcom'][1]+dict_metrics['CNRM'][1]+dict_metrics['KNMI'][1]) / 6
        mean_q=(dict_metrics['ETH'][2]+dict_metrics['MOHC'][2]+dict_metrics['ICTP'][2]+
                dict_metrics['HCLIMcom'][2]+dict_metrics['CNRM'][2]+dict_metrics['KNMI'][2]) / 6



        for idx,metrics in enumerate(['f','i','q']):
            for mdl in name_models:
                    dict_metrics[mdl][idx].to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/{mdl}_{metrics}_{np.int8(WH)}.nc")

        for idx,(mdl,metrics) in enumerate(zip([mean_f,mean_i,mean_q],['f','i','q'])):
                mdl.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_{metrics}_{np.int8(WH)}.nc")


    for SEAS in tqdm(['SON','DJF','MAM','JJA'],total=4):
        dict_metrics={}
            
        cmcc_f,cmcc_i,cmcc_q=compute_metrics(get_season(cmcc_remap,season=SEAS),meters=True,quantile=0.999)
        
        cmcc_f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/CMCC_VHR_f.nc")
        cmcc_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/CMCC_VHR_i.nc")
        cmcc_q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/CMCC_VHR_q.nc")


    cmcc_sphera=xr.open_mfdataset("/mnt/data/lcesarini/SPHERA/tp/*.nc").load()
    for SEAS in tqdm(['JJA'],total=4): #'DJF','SON','MAM',
        dict_metrics={}
        if WH:
            cmcc_f,cmcc_i,cmcc_v,cmcc_q=compute_metrics(get_season(cmcc_sphera,season=SEAS),meters=True,quantile=[0.99,0.999],wethours=WH)
        else:
            cmcc_f,cmcc_i,cmcc_v,cmcc_q=compute_metrics(get_season(cmcc_sphera,season=SEAS),meters=True,quantile=0.999)

        # plot_panel_rotated(
        #     figsize=(15,5.5),
        #     nrow=1,ncol=3,
        #     list_to_plot=[cmcc_f * 100,cmcc_i,cmcc_q],    
        #     name_fig=f"PANEL_SPHERA_{SEAS}",
        #     list_titles=["Frequency","Intensity","Heavy Prec (99.9)"],
        #     levels=[lvl_f * 100,lvl_i,lvl_q],
        #     suptitle=f"SPHERA for {SEAS}",
        #     name_metric=["[%]","[mm/hr]","[mm/hr]"],
        #     SET_EXTENT=False,
        #     cmap=[cmap_f,cmap_i,cmap_q],
        #     SAVE=False
        # )
        
        cmcc_f.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/SPHERA_f{'_WH' if WH else ''}.nc")
        cmcc_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/SPHERA_i{'_WH' if WH else ''}.nc")
        cmcc_i.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/SPHERA_v{'_WH' if WH else ''}.nc")
        cmcc_q.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/SPHERA_q{'_WH' if WH else ''}.nc")

    st_al=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/*.nc")
    st_al=st_al.isel(time=st_al['time.year'].isin(np.arange(2000,2010))).load()
    # st_al.pr.values.reshape(-1)[st_al.pr.values.reshape(-1) > 0.2].shape           

    # rea_tri=(cnrm_rg.pr*mask.mask).isel(lon=cnrm_rg.lon.isin(st_al.lon),lat=cnrm_rg.lat.isin(st_al.lat))
    # rea_tri.values.reshape(-1)[rea_tri.values.reshape(-1) > 0.1].shape    
       
    # station=stats.percentileofscore(st_al.pr.values.reshape(-1,), 0.2,nan_policy='omit')

    # np.nanquantile(st_al.pr.values.reshape(-1,),q=0.915)
    for SEAS in tqdm(['SON','DJF','MAM','JJA'],total=4):
            
        st_fre,st_int,st_q99=compute_metrics_stat(get_season(st_al,SEAS))
        
        st_fre.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/STATIONS_f.nc")
        st_int.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/STATIONS_i.nc")
        st_q99.to_netcdf(f"/home/lcesarini/2022_resilience/output/{SEAS}/STATIONS_q.nc")



ds=get_season(st_al,SEAS)
(~np.isnan(st_al.pr.values.reshape(-1))).sum() / 87672
(~np.isnan(ds.pr.values.reshape(-1))).sum() / 22080


ax=plt.axes(projection=ccrs.PlateCarree())
ds.isel(time=18).pr.plot.pcolormesh(ax=ax,levels=9)
plt.savefig("figures/test.png")
plt.close()

ax=plt.axes(projection=ccrs.PlateCarree())
st_al.pr.isel(time='').plot.pcolormesh(ax=ax,levels=lvl_f,cmap=cmap_f)
ax.add_feature(cfeature.STATES)
plt.savefig("figures/test.png")
plt.close()


plt.imshow(np.flip(st_fre.values,axis=0) / 100)
plt.imshow((st_al.pr > 0.2).sum(dim='time'))
plt.colorbar()
plt.savefig("figures/test.png")
plt.close()

xx=(st_al.pr > 0.2).sum(dim='time')

np.sum(st_al.pr.isel(lon=34,lat=82).values > .2)

np.sum(cnrm_rg.sel(lat=46.96,lon=11.34,method='nearest').pr.values > .1)
np.sum(st_al.pr.isel(lon=82,lat=34).values > 5)
xx.isel(lon=28,lat=37)
np.unravel_index(np.argmax(xx.values,axis=None),shape=xx.values.shape)

st_jja=st_al.isel(time=st_al['time.season'].isin("JJA"))


def check_na(array):
     x_=array.reshape(-1)
     x=x_[~np.isnan(x_)]

     return x

for i in range(22080):
    # if check_na(st_al.isel(time=i).pr.values).shape[0] != 162:
    print(check_na(st_jja.isel(time=i).pr.values).shape)


check_na(ds.max(dim='time').pr.values).shape
(~np.isnan(st_jja.isel(time=1).pr.values)).sum()

st_jja=st_al.isel(time=st_al['time.season'].isin("JJA"))
st_son=st_al.isel(time=st_al['time.season'].isin("SON"))

st_jja.isel(time=1).pr.values.reshape(-1,1)[~np.isnan(st_jja.isel(time=1).pr.values.reshape(-1,1))].shape
st_son.isel(time=1).pr.values.reshape(-1,1)[~np.isnan(st_son.isel(time=1).pr.values.reshape(-1,1))].shape

st_fre_jja=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/JJA/STATIONS_f.nc")
st_fre_son=xr.load_dataset(f"/home/lcesarini/2022_resilience/output/SON/STATIONS_f.nc")

st_fre_jja.pr.values.reshape(-1,1)[st_fre_jja.pr.values.reshape(-1,1) > 0].shape
st_fre_son.pr.values.reshape(-1,1)[st_fre_son.pr.values.reshape(-1,1) > 0].shape

(~np.isnan(ds.max(dim='time').pr.values)).sum()
((ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2).sum(dim='time') > 0).sum()
freq = (ds["tp" if "tp" in list(ds.data_vars) else "pr"] > 0.2).sum(dim='time') / ds["tp" if "tp" in list(ds.data_vars) else "pr"].time.shape[0]
        
(freq > 0).sum()
ax=plt.axes(projection=ccrs.PlateCarree())
xr.where(st_fre_jja.pr == 0, np.nan,st_fre_jja.pr).plot.pcolormesh(ax=ax,levels=lvl_f,cmap="RdYlGn")
ax.add_feature(cfeature.STATES)
plt.savefig("figures/test.png")
plt.close()


"""
cdo atan2 /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/uas/200912_ten1_Tdeep_u_remapped.grb2 /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/vas/200912_ten1_Tdeep_v_remapped.grb2 /mnt/data/lcesarini/test_dir.nc
cdo -divc,3.14159265359 /mnt/data/lcesarini/test_dir.nc /mnt/data/lcesarini/o.nc
cdo mulc,180. /mnt/data/lcesarini/o.nc /mnt/data/lcesarini/test_dir.nc
cdo addc,180. /mnt/data/lcesarini/test_dir.nc /mnt/data/lcesarini/o.nc 
cdo setvar,winddir /mnt/data/lcesarini/o.nc /mnt/data/lcesarini/test_dir.nc

"""
mw=xr.load_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/mw_2009_remapped_SPHERA.nc").isel(time=np.arange(240))

dir=xr.load_dataset("/mnt/data/lcesarini/test_dir.grb2",engine='cfgrib')

d_1=dir.u10.values.reshape(-1,)
m_1=mw.mw.values.reshape(-1,)
from windrose import WindroseAxes 
ax = WindroseAxes.from_ax()
ax.bar(d_1, m_1, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
plt.savefig("figure/test_windrose.png")
plt.close()

seasons=["JJA"]

for i_s,SEAS in enumerate(seasons):
    for i_m,m in enumerate(["f","i","q"]):
        if m != 'q':
            ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc")
            ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{m}.nc") 

        else:
            ds_sta=xr.open_dataset(f"output/{SEAS}/STATIONS_{m}.nc").isel(quantile=0)
            ds_cpm=xr.open_dataset(f"output/{SEAS}/ENSEMBLE_{m}.nc")

        ds_sph=xr.open_dataset(f"output/{SEAS}/SPHERA_{m}.nc")

        ds_vhr=xr.open_dataset(f"output/{SEAS}/CMCC_VHR_{m}.nc")


        mask=xr.open_dataset("data/mask_stations_nan_common.nc")

        sea_mask=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")


        rea_sph=(ds_sph*sea_mask.sftlf).isel(lon=ds_sph.lon.isin(ds_sta.lon),lat=ds_sph.lat.isin(ds_sta.lat))
        
        rea_vhr=(ds_vhr*sea_mask.sftlf).isel(lon=ds_vhr.lon.isin(ds_sta.lon),lat=ds_vhr.lat.isin(ds_sta.lat))

        cpm_tri=(ds_cpm*sea_mask.sftlf ).isel(lon=ds_cpm.lon.isin(ds_sta.lon),lat=ds_cpm.lat.isin(ds_sta.lat))
        

        #BIAS CPM-STATION
        bias_cpm_sta=((cpm_tri - ds_sta) / ds_sta) * 100
        # BIAS SPHERA-CPM
        bias_cpm_sph=((cpm_tri - rea_sph) / rea_sph) * 100
        # BIAS CMCC_VHR-CPM
        bias_cpm_vhr=((cpm_tri - rea_vhr) / rea_vhr) * 100
        # # BIAS CPM-REANALYSIS PRODUCT
        # bias_rea_cpm=((cpm_tri - rea_tri) / rea_tri) * 100
        
        #MAPS THE BIAS
        col=mpl.cm.get_cmap("PuOr",12)


        cmap_q = (mpl.colors.ListedColormap([mpl.colors.rgb2hex(col(i)) for i in np.arange(1,12)])
                .with_extremes(over=mpl.colors.rgb2hex(col(12)), under=mpl.colors.rgb2hex(col(0))))
        cmap_q.set_bad("white")

        plot_panel_rotated(
            figsize=(13,3.5),
            nrow=1,ncol=3,
            list_to_plot=[bias_cpm_sta.pr,bias_cpm_sph.pr,bias_cpm_vhr.pr],
            name_fig=f"PANEL_BIAS_{SEAS}_{m}",
            list_titles=["CPM vs Station","CPM vs SPHERA","CPM vs CMCC_VHR"],
            levels=[np.arange(-50,51,10),np.arange(-50,51,10),np.arange(-50,51,10)],
            suptitle=f"Biases of {m} for {SEAS}",
            name_metric=["[%]","[%]","[%]"],
            SET_EXTENT=False,
            cmap=[cmap_q,cmap_q,cmap_q]
        )            


SEAS="JJA"

sph=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/SPHERA_q{'_WH' if WH else ''}.nc")
ens=xr.open_dataset(f"/home/lcesarini/2022_resilience/output/{SEAS}/ENSEMBLE_q{'_WH' if WH else ''}.nc")

qq=sph.pr.isel(quantile=0).quantile(q=np.arange(0,1,0.1)).values

plot_panel_rotated(
    figsize=(15,5.5),
    nrow=1,ncol=2,
    list_to_plot=[sph.pr.isel(quantile=0),ens.pr],    
    name_fig=f"Wethours {SEAS} sphera ensemble",
    list_titles=["Heavy Prec (99.9)","Heavy Prec (99.9)"],
    levels=[10,qq],
    suptitle=f"Wethours {SEAS} sphera ensemble",
    name_metric=["[mm/hr]","[mm/hr]"],
    SET_EXTENT=False,
    cmap=[cmap_q,cmap_q],
    SAVE=False
)


