#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience")
import pickle
import argparse
# import rioxarray
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
from shapely.geometry import mapping
from cartopy import feature as cfeature

from resilience.utils import *
from resilience.utils.fix_year_eth import fix_eth

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT"

"""
PARSER
"""
parser = argparse.ArgumentParser()

parser.add_argument("-thr","--threshold",
                    help="Quantile to use as threshold",
                    required=True,default=0.99  
                    )

args = parser.parse_args()

#missing ETH, HCLIMcom
# MDL="KNMI"
q_pr=float(args.threshold)
q_pr=0.90
SEAS="JJA"
WH=True
sftlf=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KNMI/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc")
dict_quantiles={}
for MDL in ["KNMI","CMCC","CNRM","KIT","MOHC","ETH","ICTP","HCLIMcom","SPHERA"][2:]: #"KNMI","CMCC","CNRM","KIT",
    list_thr=[]

    if MDL == "ICTP":
        ll_files=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/pr/*")[:-1]    
    else:
        ll_files=glob(f"{PATH_COMMON_DATA}/{MDL}/CPM/pr/*")


    if MDL == "ETH":
        model=fix_eth()
    elif MDL == "SPHERA":
        sphera = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/pr/*{yr}*") for yr in np.arange(2000,2010)]
        sphera_ds=xr.concat(sphera,dim="time")
        sphera_ds=sphera_ds.rename({'longitude':'lon','latitude':'lat'})
        model=sphera_ds.load()
        del sphera,sphera_ds
    elif MDL == "STATIONS":
        stations = [xr.open_mfdataset(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/stations/pr/*{yr}*") for yr in np.arange(2000,2010)]
        stati_ds = xr.concat(stations,dim="time")
        model=stati_ds.load()
        del stations,stati_ds
    else:
        model=xr.open_mfdataset(ll_files).load()


    print(f"Running {MDL}")

    if MDL == "STATION":
        THR = 0.2
    else:
        THR = 0.1

    # stat=stati_ds.pr.values.reshape(-1)

    # qq=np.quantile(stat[stat > 0.2],0.9)
    #11.4,3.6 99th and 90th percentile of station's wethours 
    def _count_threshold_periods(arr,thr):
        above_threshold_periods = []
        current_period = []
        for element in arr:
            if element > thr:
                current_period.append(element)
            else:
                if current_period:
                    above_threshold_periods.append(current_period)
                    current_period = []
        
        # Check if the last period extends beyond the end of the array
        if current_period:
            above_threshold_periods.append(current_period)

        n_event=[len(x) for x in above_threshold_periods]
        max_int=[max(x) for x in above_threshold_periods]
        mean_int=[np.mean(x) for x in above_threshold_periods]
        
        return n_event,max_int,mean_int

    len_per_above_threshold = []
    max_per_periods         = []
    mean_per_periods        = []

    model=model.sel(time=model["time.season"].isin(SEAS))


    for i in tqdm(range(model.lat.shape[0])):
        for j in range(model.lon.shape[0]):
            if WH:
                THR=np.nanquantile(model.pr.values[:,i,j][model.pr.values[:,i,j] > 0.1],q=q_pr)
            else:
                THR=np.nanquantile(model.pr.values[:,i,j],q=q_pr)
            list_thr.append(THR)
    
    dict_quantiles.update({MDL:list_thr})

    import seaborn as sns 
    sns.boxplot(data=pd.DataFrame.from_dict(dict_quantiles).melt(),
                y='value',x='variable'
                )
    plt.savefig("Boxplots90th.png")
    if MDL == "STATIONS":
        x,y,z=_count_threshold_periods(np.moveaxis(model.pr.values,2,0)[:,i,j],THR)
    else:
        x,y,z=_count_threshold_periods(model.pr.values[:,i,j],THR)

    len_per_above_threshold.append(x)
    max_per_periods.append(y)
    mean_per_periods.append(z)


    # Saving the object
    #CHANGE FORM DECIMAL TO INTEGER IN FILE NAME
    if WH:
        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_len_events_{q_pr}_{SEAS}_WH.pkl', 'wb') as file:
            pickle.dump(len_per_above_threshold, file)

        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_max_events_{q_pr}_{SEAS}_WH.pkl', 'wb') as file:
            pickle.dump(max_per_periods, file)

        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_mean_events_{q_pr}_{SEAS}_WH.pkl', 'wb') as file:
            pickle.dump(mean_per_periods, file)
    else:
        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_len_events_{q_pr}_{SEAS}.pkl', 'wb') as file:
            pickle.dump(len_per_above_threshold, file)

        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_max_events_{q_pr}_{SEAS}.pkl', 'wb') as file:
            pickle.dump(max_per_periods, file)

        with open(f'/mnt/data/lcesarini/EVENTS/pr/{MDL}_mean_events_{q_pr}_{SEAS}.pkl', 'wb') as file:
            pickle.dump(mean_per_periods, file)


    # one_cell = np.array([[2,5,7],[12,24,3],[3,7,87]])
    # def count_threshold_periods(arr, threshold):
    #     above_threshold_periods = []
    #     current_period = []
    #     for element in arr:
    #         if element > threshold:
    #             current_period.append(element)
    #         else:
    #             if current_period:
    #                 above_threshold_periods.append(current_period)
    #                 current_period = []
        
    #     # Check if the last period extends beyond the end of the array
    #     if current_period:
    #         above_threshold_periods.append(current_period)
        
    #     return above_threshold_periods

    # count_threshold_periods(one_cell[:,1,1].tolist(),threshold=0.0001)
    # x=np.apply_along_axis(count_threshold_periods,axis=0,arr=one_cell,threshold=10)



    # def calculate_mean(periods):
    #     return [sum(period) / len(period) for period in periods]





    # def custom_function(slice_2d):
    #     # Define your custom function here with multiple arguments
    #     # For example, let's calculate the sum of each column in the slice
    #     return np.sum(slice_2d, axis=0)

    # # Example 3D array with shape (4, 3, 5)
    # array_shape = (4, 3, 5)
    # random_3d_array = np.random.randint(0, 100, size=array_shape)

    # # Additional arguments to be passed to the custom function
    # additional_arg1 = 10
    # additional_arg2 = 5

    # # Apply the custom function along the 0-axis with additional arguments
    # result = np.apply_along_axis(custom_function, axis=0, arr=one_cell)
    # results2=np.sum(one_cell,axis=0)

    # print("Original 3D Array:")
    # print(random_3d_array)
    # print("\nResult after applying custom function along axis 0 with additional arguments:")
    # print(result)





    # # Example usage
    # arr = [2, 5, 8, 7, 6, 12, 14, 18, 3, 2, 9, 11, 10, 8, 7, 20, 21, 22, 19, 15]
    # threshold_value = 0.1 * 1,00

    # len(count_threshold_periods(one_cell[:,1,1],0.1))

    # def prrr(arr):
    #     return np.mean(arr) * 5

    # periods_above_threshold = []
    # for i in tqdm(range(158)):
    #     for j in range(272):
    #         periods_above_threshold.append(count_threshold_periods(one_cell[:,i,j], threshold_value))

    # len_per_above_threshold = [len(x) for x in periods_above_threshold]
    # max_per_periods         = [max(x) for x in periods_above_threshold]
    # mean_per_periods         = [np.mean(x) for x in periods_above_threshold]
    # mean_values = calculate_mean(periods_above_threshold)

    # print("Periods above threshold:", periods_above_threshold)
    # print("Mean values during each period:", mean_values)



    # xr.where(one_cell.values[:10],1,0)

    # xr.where(one_cell > 0.1, 1 ,0)

    # """
    # #MAKE IT FASTER FOR FUCK SAKE
    # """
    # start_event=[]
    # finish_event=[]
    # es=0
    # # ex=np.random.uniform(0,0.15,100)
    # for i in tqdm(range(one_cell.shape[0]),total=one_cell.shape[0]):
        
    #     if (es == 0) & (one_cell[i] > 0.1):
    #         start_event.append(i)
    #         es=1
    #     elif (es == 1) & (one_cell[i] < 0.1):
    #         finish_event.append(i)
    #         es=0

    # one_cell[start_event[0]:finish_event[0]].shape
    # one_cell[3] > 0.1


    # [fe-se for fe,se in zip(finish_event,start_event)]
    # [one_cell[se:fe].mean().item() for fe,se in zip(finish_event,start_event)]

    # intensity=[]
    # duration=[]
    # # def prec_events():
    # """
    # Events should start with 1 and end with -1
    # """

    # for lon in tqdm(np.arange(knmi_rg.lon.shape[0]),total=knmi_rg.lon.shape[0]):
    #     for lat in np.arange(knmi_rg.lat.shape[0]):
    #         one_cell=knmi_rg.isel(lon=lon,lat=lat)
            
    #         wet_hours=(one_cell.pr > 0.1).astype(np.int8)
    #         diff=np.diff(np.append(np.insert(wet_hours.values,0,0),0))        
            
    #         start_event=np.argwhere(diff == 1)
    #         end_event  =np.argwhere(diff == -1)
            
    #         intensity.append([one_cell.pr.isel(time=np.arange(init,end)).max().item() for init,end in zip(start_event,end_event)])
    #         duration.append(end_event-start_event)

    #         np.array(intensity[0]).shape
    #         duration[0].shape

    #     plt.hist2d(x=duration.reshape(-1),
    #                y=np.array(intensity).reshape(-1),
    #                bins=[np.array([0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]),
    #                      np.array([0.1,0.2,0.5,1,2,5,10,20,50,100,170])],
    #                cmap="RdBu",
    #                density=True)
    #     # sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="hist")
    #     plt.savefig("figures/test_2d.png")
    #     plt.close()


    # def magnitude(a):
    #     func = lambda x: (x**2)
    #     return xr.apply_ufunc(func, a)

    # np.random.seed(111)
    # arr_rand=np.random.choice(a=np.arange(30),size=(3,3))

    # array = xr.DataArray(arr_rand,dims=['x','y'],
    #                      coords={"x":[0.1, 0.2, 0.3],"y":[0.1, 0.2, 0.3]})

    # array
    # magnitude(array)
    # one_cell

    # def _count_threshold_periods(arr):
    #     above_threshold_periods = []
    #     current_period = []
    #     for element in arr:
    #         if element > 0.1:
    #             current_period.append(element)
    #         else:
    #             if current_period:
    #                 above_threshold_periods.append(current_period)
    #                 current_period = []
        
    #     # Check if the last period extends beyond the end of the array
    #     if current_period:
    #         above_threshold_periods.append(current_period)
        
    #     return above_threshold_periods

    # def count_threshold_periods(arr):
    #     return xr.apply_ufunc(_count_threshold_periods, arr)

    # _count_threshold_periods(array.isel(x=0,y=0))
    # count_threshold_periods(array)



    # arr_mdl=knmi_rg.drop_vars('time_bnds')


    # _count_threshold_periods(arr_mdl.isel(lon=0,lat=0))

    # magnitude(arr_mdl)

    # vfunc=np.vectorize(_count_threshold_periods)

    # vfunc(arr_mdl.pr.values)
    # list_xr=[]
    # len_per_above_threshold = []
    # max_per_periods         = []
    # mean_per_periods        = []

    # for i in tqdm(range(158)):
    #     for j in range(272):
    #         ll_above=_count_threshold_periods(arr_mdl.pr.values[:,i,j])

    #         len_per_above_threshold.append([len(x) for x in ll_above])
    #         max_per_periods.append([max(x) for x in ll_above])
    #         mean_per_periods.append([np.mean(x) for x in ll_above])

    # len(len_per_above_threshold)


    # print("Periods above threshold:", len(len_per_above_threshold))
    # print("Mean values during each period:", mean_values)

    # periods_above_threshold = []
    # for i in tqdm(range(158)):
    #     for j in range(272):
    #         periods_above_threshold.append(count_threshold_periods(one_cell[:,i,j], threshold_value))

    # len_per_above_threshold = [len(x) for x in periods_above_threshold]
    # max_per_periods         = [max(x) for x in periods_above_threshold]
    # mean_per_periods         = [np.mean(x) for x in periods_above_threshold]
    # mean_values = calculate_mean(periods_above_threshold)

    # print("Periods above threshold:", periods_above_threshold)
    # print("Mean values during each period:", mean_values)
