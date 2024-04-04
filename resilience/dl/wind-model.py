#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import torch
import torch.nn as nn


import numpy as np
import torch.optim as optim
import torch.utils.data as data
import os
import sys
sys.path.append("/home/lcesarini/2022_resilience/")
sys.path.append("/mnt/data/lcesarini/2021_milk_2/2021_milk/")
import time
import datetime
import numpy as np
import pandas as pd 
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

#Keras import
# #Imports from the resilience project
from resilience.utils import create_list_coords

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
mask=xr.open_dataset("data/mask_stations_nan_common.nc")

sphera=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/mw_*_remapped_SPHERA.nc")

modell=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/mw/mw_ETH_ECMWF-*.nc")

for mdl in ["CNRM","CMCC","KNMI","KIT","MOHC","ETH","HCLIMcom","ICTP"]:
    len(glob(f"/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{mdl}/CPM/mw/*.nc"))

# Coords Sant'apollinare 11.82572434	45.0340755
statio=pd.read_table("/home/lcesarini/2022_resilience/csv/stations/test_apolllinare.txt",sep=";",header=None)
statio=pd.read_table("/home/lcesarini/2022_resilience/csv/stations/231 - Sant'Apollinare (Rovigo) - Velocit… vento 10m media aritm. media (m-s).txt",sep=";",header=0)
statio['date'] = statio.loc[:,"DATA"] + " " +statio.loc[:,"ORA"]
statio['date']=pd.to_datetime(statio['date'],format="%d/%m/%Y %H:%M")
statio=statio[statio['date'] < pd.to_datetime("2010-01-01 01:00")]

list_dates=pd.date_range(start="2003-08-12 01:00",end="2010-01-01",freq="h")


#Filter arrays on dates in the period (i.e., list_dates)
sphera=sphera.sel(time=slice(pd.to_datetime("2003-08-12 01:00"), pd.to_datetime("2010-01-01 01:00")))
modell=modell.sel(time=slice(pd.to_datetime("2003-08-12 01:00"), pd.to_datetime("2010-01-01 01:00")))
statio=statio.loc[statio['date'].isin(list_dates),:]

#Remove the dates from all arrays that are missing forom at least on of the arrays
dm_sph=list_dates[~list_dates.isin(sphera.time.values)]
dm_mod=list_dates[~list_dates.isin(modell.time.values)]
dm_sta=list_dates[~list_dates.isin(statio['date'])]

dm_unique=np.unique(list(dm_sta) + list(dm_mod) + list(dm_sph))

sphera=sphera.sel(time=~sphera.time.isin(pd.to_datetime(dm_unique)))
modell=modell.sel(time=~modell.time.isin(pd.to_datetime(dm_unique)))
statio.loc[~statio['date'].isin(pd.to_datetime(dm_unique)),:]

arr_sta=np.array(statio.loc[~statio['date'].isin(pd.to_datetime(dm_unique)),"VALORE"])#[1:]

arr_sph=sphera.sel(lon=11.82572434,lat=45.0340755, method='nearest').mw.values
arr_mod=modell.sel(lon=11.82572434,lat=45.0340755, method='nearest').mw.values


def scale_data(array,scal_tech):
    """
    Function that scales an input according to a 
    specified scaling technique.

    @array: input to be scaled
    @scal_tech: scaling technique to be used among:
        - minmax1: scaling between [0,1]
        - minmax2: scaling between [-1,1]
        - standard: standard scaling
    """

    if not isinstance(array,np.ndarray):
        array = np.array(array)
    
    if len(array.shape) == 1:
        array = array.reshape(-1,1) 
    # Needed to scale back the MAE and MSE
    min_array,max_array = np.min(array), np.max(array)
    # normalize
    if scal_tech == 'minmax1':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_array = scaler.fit_transform(array)
    elif scal_tech == 'standard':
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(array)

    elif scal_tech == 'minmax2':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_array = scaler.fit_transform(array)

    return scaled_array,min_array,max_array

def scale_inverse(array,rmin,rmax,scal_tech):
    """
    Function that revert the scaling input 
    to its original values.

    @array: input to be scaled
    @rmin: minimum of the range of the original data
    @rmax: maximum of the range of the original data
    @scal_tech: scaling technique to be used among:
        - minmax1: scaling between [0,1]
        - minmax2: scaling between [-1,1]
        - standard: standard scaling
    Reference equation:
        \math{xo = (xs-tmin)*(rmax-rmin)/(tmax-tmin)+rmin}
    """

    if not isinstance(array,np.ndarray):
        array = np.array(array)
    
    if len(array.shape) == 1:
        array = array.reshape(-1,1) 
    if scal_tech == 'minmax1':
        tmin=0
        tmax=1
        scale_back=(array-tmin) * (rmax-rmin) / (tmax-tmin) + rmin
    elif scal_tech == 'minmax2':
        tmin=-1
        tmax=1
        scale_back=(array-tmin) * (rmax-rmin) / (tmax-tmin) + rmin


    return scale_back

def partitioning(total_inputs:np.ndarray,
                 total_output:np.ndarray,
                 window:int,
                 splitting:list = [0.6,0.2,0.2]):
    """
    Function that reshapes the inputs into an appropriate shape
    and returns the splitting of the dataset. Default: 60/20/20.

    @total_inputs:
    @total_outputs:
    @window: length of the single batch
    @splitting: proportioning of the splitting
    """

    tot_inp_batch,tot_out_batch = create_dataset(total_inputs,total_output,window)

    # Splitting
    val_perc   = splitting[1]
    test_perc  = splitting[2]
    n_batch    = tot_inp_batch.shape[0]

    index_val   = int(round(n_batch*val_perc))
    index_test  = int(round(n_batch*test_perc))
    #Get training batches as the difference between the total and the sum of validation and testing
    index_train = n_batch - index_val - index_test

    ##### CHECK that all the batches are used
    assert (index_train+index_val+index_test) - n_batch == 0, f"Mismatching shapes: TR:{index_train},VL:{index_val},TE:{index_test}"

    """
    SPLITTING:60/20/20
    """
    train_inputs, val_inputs, test_inputs = tot_inp_batch[0:index_train],tot_inp_batch[index_train:(index_val+index_train)],tot_inp_batch[(index_val+index_train):(index_val+index_train+index_test)]

    train_outputs, val_outputs, test_outputs = tot_out_batch[0:index_train],tot_out_batch[index_train:(index_val+index_train)],tot_out_batch[(index_val+index_train):(index_val+index_train+index_test)]


    return train_inputs, val_inputs, test_inputs, train_outputs, val_outputs, test_outputs

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

class SimpleTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.rnn(x)
        # out shape: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Predict the last time step
        # out shape: (batch_size, output_size)
        return out

model_name="LSTM"
window=12
# OPTIMIZER=Adam()
NB_EPOCH = 200
BATCH_SIZE = 64
VERBOSE = 0
NB_CLASSES = 1  # number of outputs = number of classes

"""LOAD THE DATA"""
total_inputs,total_output = arr_mod,arr_sph
"""SCALING THE DATA"""
total_inputs, min_inputs, max_inputs = scale_data(total_inputs,"minmax1")
total_output, min_output, max_output = scale_data(total_output,"minmax1")
"""SPLITTING THE DATA"""
(train_inputs, val_inputs, test_inputs, 
train_outputs, val_outputs, test_outputs) = partitioning(total_inputs,
                                                         total_output,
                                                         window)
# Create an instance of the model
input_size = 1  # Assuming univariate time series
hidden_size = 64
output_size = 1  # Predicting one future value
model = SimpleTimeSeriesModel(input_size, hidden_size, output_size)

 
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(torch.from_numpy(train_inputs[:4000,:,:]), torch.from_numpy(train_outputs[:4000,:])), shuffle=True, batch_size=BATCH_SIZE)
 
n_epochs = NB_EPOCH
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(train_inputs))
        train_rmse = np.sqrt(loss_fn(y_pred, torch.from_numpy(train_outputs)))
        y_pred = model(torch.from_numpy(val_inputs))
        test_rmse = np.sqrt(loss_fn(y_pred, torch.from_numpy(val_outputs)))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


# with torch.no_grad():
#     # shift train predictions for plotting
#     train_plot = np.ones_like(timeseries) * np.nan
#     y_pred = model(torch.from_numpy(train_inputs))
#     y_pred = y_pred[:, -1, :]
#     train_plot[lookback:train_size] = model(X_train)[:, -1, :]
#     # shift test predictions for plotting
#     test_plot = np.ones_like(timeseries) * np.nan
#     test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
plt.plot(scale_inverse(model(torch.from_numpy(train_inputs)).detach().numpy()[3500:4000],min_output,max_output,"minmax1"), c='b')
plt.plot(scale_inverse(train_outputs[3500:4000],min_output,max_output,"minmax1"), c='r')
plt.show()