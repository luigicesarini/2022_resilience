#! /home/lcesarini/miniconda3/envs/milk/bin/python
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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

#Keras import
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.backend import square, mean
from tensorflow.python.keras.backend import variable

#Imports from the milk package
from scripts.models.models import DNNC,LSTMBiSTACKnet,CNN1D
#Imports from the resilience project
# from resilience.utils import create_list_coords

# Check for available GPUs
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")

os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
mask=xr.open_dataset("data/mask_stations_nan_common.nc")

sphera=xr.open_mfdataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/mw/mw_*.nc")

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

# arr_sph.shape
# arr_mod.shape
# arr_sta.shape

# plt.plot(arr_sph[30:60],'-o',label="SPHERA")
# plt.plot(arr_mod[30:60],'-o',label="ECMWF")
# plt.plot(arr_sta[30:60],'-o',label="STATION")
# plt.legend()
# plt.show()
# plt.savefig("/home/lcesarini/2022_resilience/figures/eda_sphera_ecmwf.png")
# plt.close()

# plt.plot(np.quantile(arr_sph,q=np.arange(0.5,1,0.05)),'-o',label="SPHERA")
# plt.plot(np.quantile(arr_mod,q=np.arange(0.5,1,0.05)),'-o',label="ETH")
# plt.plot(np.quantile(arr_sta,q=np.arange(0.5,1,0.05)),'-o',label="STATION")
# plt.title(f"Wind Values of {np.arange(0.5,1,0.05)} quantiles")
# plt.legend()
# plt.show()

# plt.plot(np.quantile(arr_sph,q=np.arange(0.9,1,0.001)),'-o',label="SPHERA")
# plt.plot(np.quantile(arr_mod,q=np.arange(0.9,1,0.001)),'-o',label="ETH")
# plt.plot(np.quantile(arr_sta,q=np.arange(0.9,1,0.001)),'-o',label="STATION")
# plt.title(f"Wind Values of extreme quantiles quantiles")
# plt.legend()
# plt.show()


# #plot boxplot with seaborn
# import seaborn as sns
# sns.boxplot(data=[arr_sph,arr_mod,arr_sta],width=0.5)
# plt.xticks([0,1,2],["SPHERA","ECMWF","STATION"])
# plt.show()

"""


"""
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

def training_procedure(model,
                       train_inputs,train_outputs,  #training partitions
                       val_inputs, val_outputs,     #validation partitions    
                       BATCH_SIZE,NB_EPOCH,VERBOSE, #parameters training
                       reduce_lr,best_checkpoint,early_stopping    #callbacks
                       ):
    """
    Trains the model built
    """
    from time import time
    start = time()
    history = model.fit(
    train_inputs, train_outputs, 
    batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
    verbose=VERBOSE, # 0 for no logging tox stdout, 1 for progress bar logging, 2 for one log line per epoch
    validation_data=(val_inputs,val_outputs),
    #validation_split=VALIDATION_SPLIT,
    callbacks=[reduce_lr,best_checkpoint,early_stopping]
    )
    finish=time()

    training_time=finish-start

    return history,training_time
"""

"""
model_name="CNN"
window=24
OPTIMIZER=Adam()
NB_EPOCH = 500
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
"""SETTING TRAINING PROCEDURE"""

#INPUTS FOR BUILDING MODEL
timesteps, variables = train_inputs.shape[1],train_inputs.shape[2] # input image dimensions
#CALLBACKS
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01,patience=5, min_lr=0.0001)
best_checkpoint = ModelCheckpoint(f"model_saved/{model_name}-{OPTIMIZER.get_config()['name']}-{window}.h5", 
                                    monitor='val_loss', save_best_only=True, verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

K.clear_session()
"""
Building
Reflects on the activation function
"""
model = CNN1D.build_wo_attention(timesteps,variables,1)
"""Compile"""
model.compile(loss="mse", optimizer=OPTIMIZER)
"""Training"""
history = training_procedure(model,train_inputs,train_outputs,val_inputs, val_outputs,
                            BATCH_SIZE,NB_EPOCH,VERBOSE,reduce_lr,best_checkpoint,early_stopping)


def mape(actual,predicted):
    """
    Return the value of the mean absolute percentage error
    according to:

    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    @actual: values of the ground truth
    @predicted: values of the predicted variable

    """ 


    mape= np.sum(np.abs(( actual - predicted) / actual)) / actual.shape[0]

    return mape 

def evaluate_performances(model_name,model_name_df,window,
                          path_to_weights,
                          test_inputs,test_outputs,
                          max_output, min_output,
                          opti,
                          performance_to_csv=False,
                          performance_to_cmd=True,
                          return_model=False,
                          return_metrics=True                         
                          ):
    """
    Returns a pandas dataframe with the performances of the model chosen

    @model_name: name of the model to evaluate
    @model_name_df: name of the model to save/print 
    @country: the name of the country that the model was trained on. One of: France, Germany,and Italy
    @window: length of the single batch
    @test_inputs: inputs of the data unseen by the model 
    @test_outputs: target of the input never seen by the model
    @opti: optimizer usd to compile the model. One of Adam() or RMSProp()
    @EV: name of the variable used 
    @performance_to_csv: save the pandas dataframe to a CSV
    @performance_to_cmd: print the pandas dataframe in the terminal
    @return_model: return the TF model loaded to evaluate the performances
    @return_metrics: store the pandas dataframe into a variable

    !!TO DO:
    The CSV saved to a path that is passed as an argument
    """
    # model_loaded = tf.keras.models.load_model(f'model_saved/{model_name}-{country}-{EV}-{window}.h5',custom_objects={'CylindricalPadLSTMCNN': CylindricalPadLSTMCNN}) 
    # if model_name == "Roll-CNN2D":
    #     model_loaded = tf.keras.models.load_model(f'model_saved/{model_name}-{country}-{EV}-{opti}-{window}.h5',custom_objects={'CylindricalPad': CylindricalPad}) 
    # elif model_name == "Roll-WavenetCNN":
    #     model_loaded = tf.keras.models.load_model(f'model_saved/{model_name}-{country}-{EV}-{opti}-{window}.h5',custom_objects={'CylindricalPadCausal': CylindricalPadCausal}) 
    # else:
    #     model_loaded = tf.keras.models.load_model(f'model_saved/{model_name}-{country}-{EV}-{opti}-{window}.h5') 

    model_loaded = tf.keras.models.load_model(path_to_weights)
    #model_loaded = tf.keras.models.load_model(f"models_sensitivity/{model_name}-{country}-{EV}-{OPTIMIZER.get_config()['name']}-{window}-{shorten}/")
    
    dict_df = {
    'model':model_name_df,
    'date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
    'window':window,
    'r2':r2_score(test_outputs,model_loaded.predict(test_inputs)),
    'mse':mean_squared_error(test_outputs,model_loaded.predict(test_inputs)),
    'mae':mean_absolute_error(test_outputs,model_loaded.predict(test_inputs)),
    'mape':mape(test_outputs,model_loaded.predict(test_inputs)),
    }

    if performance_to_cmd:
        print(pd.DataFrame(dict_df, index = [0]))

    if not os.path.isfile(f'output/metrics_milk.csv') and performance_to_csv:
        pd.DataFrame(dict_df, index = [0]).to_csv(f'output/new_metrics_milk.csv',
                                                columns = ['model','date','ev','window',
                                                           'r2','mse','mae','mape'], 
                                                header = True, 
                                                index = False,
                                                index_label=False, 
                                                mode = 'a')
    else:
        pd.DataFrame(dict_df, index = [0]).to_csv(f'output/new_metrics_milk.csv',
                                                header = False, 
                                                index = False,
                                                index_label=False, 
                                                mode = 'a')
    
    if return_metrics and not return_model:
        return {'metrics':pd.DataFrame(dict_df, index = [0])}
    elif return_model and not return_metrics:
        return {'model':model_loaded}
    elif return_metrics and return_model:
        """
        Return metrics and then the model
        """
        return {'metrics':pd.DataFrame(dict_df, index = [0]),'model':model_loaded}    


model_loaded = evaluate_performances(model_name,"Wind one model",
                                     window,"/home/lcesarini/2022_resilience/model_saved/LSTM-Adam-24.h5",
                                     test_inputs,test_outputs,
                                     max_output, min_output,
                                     OPTIMIZER.get_config()["name"],
                                     performance_to_csv=False,
                                     return_model=True)


plt.plot(history[0].history['loss'], label='train')
plt.plot(history[0].history['val_loss'], label='validation')
plt.legend()
plt.show()

model_loaded = tf.keras.models.load_model(f"model_saved/{model_name}-{OPTIMIZER.get_config()['name']}-{window}.h5")
print(mean_squared_error(test_outputs,model_loaded.predict(test_inputs)))
test_inputs[0,:,:]


model_loaded['metrics']
r2_score(np.array([test_inputs[i,0,:] for i in range(test_inputs.shape[0])]),test_outputs)
r2_score(model_loaded['model'].predict(test_inputs),test_outputs)


plt.plot(scale_inverse(model_loaded['model'].predict(test_inputs)[100:150],min_output,max_output,'minmax1'),
         'ro--',label="prediction")
plt.plot(scale_inverse(test_outputs[100:150],min_output,max_output,"minmax1"),'bo--',label="truth")
plt.legend()
plt.show()

# import xgboost as xgb

# model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, max_depth=3)
# model.fit(total_inputs[:40000], total_output[:40000])

# y_pred = model.predict(total_inputs[40000:])

# # mse = mean_squared_error(total_output[40000:], y_pred)
# # rmse = np.sqrt(mse)
# np.argmax(y_pred)

# f"Error on RAW: {np.sqrt(mean_squared_error(total_inputs[40000:], total_output[40000:]))}"
# f"Error on BC: {np.sqrt(mean_squared_error(y_pred,total_output[40000:]))}"

# plt.plot(np.quantile(total_inputs[40000:],q=np.arange(0.9,1,0.001)).ravel(),"-o")
# plt.plot(np.quantile(total_output[40000:],q=np.arange(0.9,1,0.001)).ravel(),"-o")
# plt.plot(np.quantile(y_pred,q=np.arange(0.9,1,0.001)).ravel(),"-o")
# plt.legend(["Model","Sphera","Model corrected"])
# plt.show()

# plt.plot(np.arange(0,60),total_inputs[40000:][(3053-30):(3053+30)].ravel(),"-o")
# plt.plot(np.arange(0,60),total_output[40000:][(3053-30):(3053+30)].ravel(),"-o")
# plt.plot(np.arange(0,60),y_pred[(3053-30):(3053+30)].ravel(),"-o")
# plt.legend(["Model","Sphera","Model corrected"])
# plt.show()


# import torch
# import torch.nn as nn

# class SimpleTimeSeriesModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleTimeSeriesModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, input_size)
#         out, _ = self.rnn(x)
#         # out shape: (batch_size, sequence_length, hidden_size)
#         out = self.fc(out[:, -1, :])  # Predict the last time step
#         # out shape: (batch_size, output_size)
#         return out

# # Create an instance of the model
# input_size = 1  # Assuming univariate time series
# hidden_size = 16
# output_size = 1  # Predicting one future value
# model = SimpleTimeSeriesModel(input_size, hidden_size, output_size)


# import numpy as np
# import torch.optim as optim
# import torch.utils.data as data
 
# optimizer = optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()
# loader = data.DataLoader(data.TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_outputs)), shuffle=True, batch_size=8)
 
# n_epochs = 1000
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch)
#         loss = loss_fn(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Validation
#     if epoch % 100 != 0:
#         continue
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(torch.from_numpy(train_inputs))
#         train_rmse = np.sqrt(loss_fn(y_pred, torch.from_numpy(train_outputs)))
#         y_pred = model(torch.from_numpy(val_inputs))
#         test_rmse = np.sqrt(loss_fn(y_pred, torch.from_numpy(val_outputs)))
#     print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


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
# plt.plot(model(torch.from_numpy(train_inputs)).detach().numpy()[500:800], c='b')
# plt.plot(train_outputs[500:800], c='r')
# plt.show()

# train_inputs.shape
# # Create some example input data (batch_size=1, sequence_length=10, input_size=1)
# input_data = torch.from_numpy(train_inputs.reshape(-1,24,1))
# input_data = torch.randn(1,10,1)

# # Make predictions
# predictions = model(input_data)

# # Print the predictions
# print(predictions)
# plt.plot(predictions.detach().numpy()[:100],c='red')
# plt.plot(train_outputs[:100],c='green')
# plt.show()
