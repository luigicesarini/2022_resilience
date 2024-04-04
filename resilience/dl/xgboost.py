#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
sys.path.append("/mnt/data/lcesarini/2021_milk_2/2021_milk/")
import time
import datetime
import numpy as np
import pandas as pd 
import xarray as xr
# import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
#Keras import
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.backend import square, mean
# from tensorflow.python.keras.backend import variable

from resilience.utils import create_list_coords

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"

import xgboost as xgb
from sklearn.metrics import mean_squared_error


pr_mod=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/HCLIMcom/CPM/pr/HCLIMcom_ECMWF-ERAINT_200101010030-200112312330.nc").load()
pr_sta=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{yr}.nc" for yr in np.arange(2001,2002)]).load()

X_train,X_test = pr_mod.pr.isel(lat=61,lon=213).values.reshape(-1,1)[:6000,:],pr_mod.pr.isel(lat=61,lon=213).values.reshape(-1,1)[6000:,:]
y_train,y_test = pr_sta.pr.isel(lat=8,lon=71).values.reshape(-1,1)[:6000,:],pr_sta.pr.isel(lat=8,lon=71).values.reshape(-1,1)[6000:,:]
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
np.argmax(y_test)

plt.plot(np.arange(0,60))
plt.show()

plt.plot(np.arange(0,60),y_test[(1418-30):(1418+30)].ravel())
plt.plot(np.arange(0,60),y_pred[(1418-30):(1418+30)].ravel())
plt.legend(["y_test","y_pred"])
plt.savefig("/home/lcesarini/2022_resilience/figures/xgboost.png")