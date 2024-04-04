#! /home/lcesarini/miniconda3/envs/detectron/bin/python
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature
from sklearn.metrics import r2_score

from utils.missing_years import *
from utils.retrieve_data import *
from utils.visualization import *
from utils.bias import *


os.chdir("/home/lcesarini/2022_resilience/")

proj = ccrs.PlateCarree()

rot = ccrs.RotatedPole(pole_longitude=-170.0, 
                    pole_latitude=43.0, 
                    central_rotated_longitude=0.0, 
                    globe=None)


if __name__=="__main__":

    name_station ="AA_0220"
    meta = pd.read_csv("meta_station_updated_col.csv")
    # print(meta[meta.name == name_station])

    prec_o,date = get_observed()

    ds = xr.open_mfdataset("/mnt/data/lcesarini/ERA5/resilience/tp/tp*")


    ds_station = get_mod_at_obs(model=ds * 1000,
                                name_station=name_station,
                                rotated_cartopy=rot, proj_cartopy=proj,
                                rotated=False)

    min_over_date,max_over_date = get_overlapping_dates(date,ds_station.time)
    
    index_overlap=np.where((date >= min_over_date) & (date <= max_over_date))[0]

    date_overlap=date[index_overlap]
    prec_o_over =prec_o[index_overlap]


    ds_sliced =  ds_station.sel(time = date_overlap, method='nearest').load()

    date_overlap[date_overlap > np.array('1985-01-01T02:00:00', dtype=np.datetime64)]
    ds_sliced.time[ds_sliced.time > np.array('1984-12-31T21:00:00', dtype=np.datetime64)]


    assert ds_sliced[list(ds_sliced.data_vars)[0]].values.shape[0] == prec_o_over.shape[0], "Shape of observation and filtered model are different"

    #CREATE THE CORRECTOR OBJECT

    corrector_hour = BiasCorrection(prec_o_over,ds_sliced[list(ds_sliced.data_vars)[0]])

    #BIAS ORARIO
    bias_medio_orario = np.nanmedian(corrector_hour.Bias())
    # print(bias_medio_orario)

    #BIAS MONLTHY MAX
    bms_station = ds_sliced.resample(time="1M").max()
    # bms_station.time[bms_station.time > np.array('1984-12-31T21:00:00', dtype=np.datetime64)]
    bms_obs     = pd.Series(prec_o_over.reshape(-1),index=date).resample("1M").max()
    # bms_obs[bms_obs.index > np.array('1984-12-31T21:00:00', dtype=np.datetime64)]
    corrector_bms  = BiasCorrection(bms_obs,bms_station[list(bms_station.data_vars)[0]])
    
    bias_bms = corrector_bms.Bias()
    bias_medio_bms = np.nanmedian(bias_bms)
    # print(bias_medio_bms)

    df_bias= pd.DataFrame(pd.Series(bias_bms.reshape(-1),index=bms_obs.index).resample("1M").max())
    df_bias['Date']=pd.to_datetime(df_bias.index).strftime("%m")
    by_month = df_bias.groupby('Date').mean()


    # plt.plot(by_month,'-or')
    # plt.hlines(y=0,
    #            xmin=by_month.index.min(),
    #            xmax=by_month.index.max(),
    #            colors=['black'])
    # # plt.plot(bms_obs.index,bms_station[list(bms_station.data_vars)[0]].values,'or', label='ERA5')
    # # plt.plot(bms_obs.index,bms_obs,'og',label='Obs.')
    # # plt.legend()

    # plt.title("Bias of the max monthly over the different months")
    # plt.savefig("Monthly max bias")
    # plt.close()
    #BIAS QUANTILI

    ecdf_obs,ecdf_mod=corrector_hour.EQM()

    corrector_hour.pltECDF(ecdf_obs,ecdf_mod)
    rainfall = 24.3999999999999
    print(f"""
            RP for {rainfall:.4f}mm rainfall:
            Observed: {1 / (1-ecdf_obs(rainfall)) / 24 / 365:.2f}
            Modelled: {1 / (1-ecdf_mod(rainfall)) / 24 / 365:.2f}
            """)

    print(f"Exceedance prob in the observations {ecdf_obs(rainfall):.5f}")
    print(f"Exceedance prob in the model {ecdf_obs(rainfall):.5f}")
    # threshold_model=np.percentile(ds_sliced.tp[ds_sliced.tp > 0], ecdf(0.2))
            # {np.nanmax(prec_o_over):.2f}
    
    # ds_sliced_thr=np.where(ds_sliced.tp <= threshold_model,0,ds_sliced.tp)
    # print(ecdf(ds_sliced.tp))
    
    # def ecdf(data):
    #     """ Compute ECDF """
    #     x = np.sort(data,axis=0)
    #     n = x.size
    #     y = np.arange(1, n+1) / n
    #     return(x,y)
    

    # corr = np.percentile(obs_data, p) -  np.percentile(mod_data, p)
    # prec_o_over[prec_o_over > 0.2].shape

    # np.argwhere(prec_o_over.reshape(-1) > 0.2)

    # prec_o_over.shape


    # np.percentile