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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
#Keras import
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorflow.python.keras.backend import variable

from scripts.models.models import LSTMnet,LSTMSTACKnet,DNNC
from resilience.utils import create_list_coords
os.chdir("/home/lcesarini/2022_resilience/")

PATH_COMMON_DATA="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS"
mask=xr.open_dataset("data/mask_stations_nan_common.nc")
ele=xr.open_dataset("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")

def get_triveneto(ds,sta_val):
    return (ds * mask.mask).isel(lon=ds.lon.isin(sta_val.lon),
                                 lat=ds.lat.isin(sta_val.lat))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def scale_feature(df_train, df_validation, df_testing, prefix_feat):
    #Select columns based on feature
    df = df_train.loc[:,df_train.columns.str.contains(prefix_feat)]
    feature = np.array(df)
    mean_f  = feature.mean()
    std_f   = feature.std()
    scaled_training   = (df_train.loc[:,df_train.columns.str.contains(prefix_feat)] - mean_f)/std_f
    scaled_validation = (df_validation.loc[:,df_validation.columns.str.contains(prefix_feat)] - mean_f)/std_f
    scaled_testing    = (df_testing.loc[:,df_testing.columns.str.contains(prefix_feat)] - mean_f)/std_f

    return [scaled_training,scaled_validation,scaled_testing]


if __name__ == "__main__":
    print(tf.__version__)
    pr_mod=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/HCLIMcom/CPM/pr/HCLIMcom_ECMWF-ERAINT_200101010030-200112312330.nc").load()
    mw_mod=xr.open_mfdataset(f"{PATH_COMMON_DATA}/ECMWF-ERAINT/HCLIMcom/CPM/mw/mwIMcom_ECMWF-ERAINT_200101010000-200112312300.nc").load()

    pr_mod=xr.where(pr_mod.pr.values < 0.1,0,pr_mod.pr)
    # pr_sta=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{yr}.nc" for yr in np.arange(2000,2010)]).load()
    pr_sta=xr.open_mfdataset([f"{PATH_COMMON_DATA}/stations/pr/pr_st_{yr}.nc" for yr in np.arange(2001,2002)]).load()

    id_cells_sta=np.argwhere(~np.isnan(np.nanmax(pr_sta.pr,axis=2)))

    #Find the single cell
    X_=pr_sta.isel(lon=id_cells_sta[0,1],lat=id_cells_sta[0,0]).lon.values.item()
    Y_=pr_sta.isel(lon=id_cells_sta[0,1],lat=id_cells_sta[0,0]).lat.values.item()

    idx_lat_mod,idx_lon_mod=np.argwhere(pr_mod.lat.values==Y_).item(),np.argwhere(pr_mod.lon.values==X_).item()

    input=pr_mod.isel(lon=[idx_lon_mod-1,idx_lon_mod,idx_lon_mod+1],lat=[idx_lat_mod-1,idx_lat_mod,idx_lat_mod+1]).pr.values
    output=pr_sta.isel(lon=id_cells_sta[0,1],lat=id_cells_sta[0,0]).pr.values    

    WH_sta=np.argwhere(output > 0.2)
    WH_mod=np.sum(np.where(input > 0.1,1,0),axis=0)
    
    min=np.min([WH_sta.shape[0],WH_mod.min()])

    sorted_input=np.sort(input,axis=0)[-min:,:,:]
    sorted_output=np.sort(output,axis=0)[-min:]


    plt.plot(sorted_output,label='sta')
    plt.plot(sorted_input[:,1,1],label='model')
    plt.legend()
    plt.show()

    """
    SCALE, SPLIT and BATCH THE DATA
    Perform the correction only on the wet hours
    """
    # SPLIT
    size_train = int(sorted_input.shape[0] * 0.6)
    size_valid = int(sorted_input.shape[0] * 0.2)
    size_test  = sorted_input.shape[0] - size_train - size_valid

    size_train + size_valid + size_test == sorted_input.shape[0]


    pr_tr,pr_vl,pr_te=sorted_input[:size_train,:,:],sorted_input[size_train:(size_train+size_valid),:,:],sorted_input[(size_train+size_valid):,:,:]
    # mw_tr,mw_vl,mw_te=mw_mod.mw.values[:size_train,:,:],mw_mod.mw.values[size_train:(size_train+size_valid),:,:],mw_mod.mw.values[(size_train+size_valid):,:,:]
    pr_sta_tr,pr_sta_vl,pr_sta_te=sorted_output[:size_train],sorted_output[size_train:(size_train+size_valid)],sorted_output[(size_train+size_valid):]


    
    scaler=MinMaxScaler(feature_range=(0,1))
    #Precipitation
    pr_tr_scal=scaler.fit_transform(pr_tr.reshape(-1,1)).reshape(pr_tr.shape)
    pr_vl_scal=scaler.fit_transform(pr_vl.reshape(-1,1)).reshape(pr_vl.shape)
    pr_te_scal=scaler.fit_transform(pr_te.reshape(-1,1)).reshape(pr_te.shape)
    #Wind
    # mw_tr_scal=scaler.fit_transform(mw_tr.reshape(-1,1)).reshape(mw_tr.shape)
    # mw_vl_scal=scaler.fit_transform(mw_vl.reshape(-1,1)).reshape(mw_vl.shape)
    # mw_te_scal=scaler.fit_transform(mw_te.reshape(-1,1)).reshape(mw_te.shape)
    #Station
    pr_sta_tr_scal=scaler.fit_transform(pr_sta_tr.reshape(-1,1)).reshape(pr_sta_tr.shape)
    pr_sta_vl_scal=scaler.fit_transform(pr_sta_vl.reshape(-1,1)).reshape(pr_sta_vl.shape)
    pr_sta_te_scal=scaler.fit_transform(pr_sta_te.reshape(-1,1)).reshape(pr_sta_te.shape)


    # Batch the data
    length_batch=3
    # Precipitation
    # pr_mod_tri=get_triveneto(pr_mod.pr,pr_sta)
    pr_tr_scal=np.array([pr_tr_scal[i:(i+length_batch),:,:] for i in np.arange(pr_tr_scal.shape[0]-length_batch)])
    pr_vl_scal=np.array([pr_vl_scal[i:(i+length_batch),:,:] for i in np.arange(pr_vl_scal.shape[0]-length_batch)])
    pr_te_scal=np.array([pr_te_scal[i:(i+length_batch),:,:] for i in np.arange(pr_te_scal.shape[0]-length_batch)])

    # Wind
    # mw_mod_tri=get_triveneto(mw_mod.mw,pr_sta)
    # mw_tr_scal=np.array([mw_tr_scal[i:(i+length_batch),:,:] for i in np.arange(mw_tr_scal.shape[0]-length_batch)])
    # mw_vl_scal=np.array([mw_vl_scal[i:(i+length_batch),:,:] for i in np.arange(mw_vl_scal.shape[0]-length_batch)])
    # mw_te_scal=np.array([mw_te_scal[i:(i+length_batch),:,:] for i in np.arange(mw_te_scal.shape[0]-length_batch)])

    # mw_mod_batch=np.array([mw_mod_tri.values[i:(i+length_batch),:,:] for i in np.arange(pr_mod_tri.shape[0]-length_batch)])
    # Station prec
    # pr_sta_tr_scal=np.array([pr_sta_tr_scal[i:(i+length_batch),:,:] for i in np.arange(pr_sta_tr_scal.shape[0]-length_batch)])
    # pr_sta_vl_scal=np.array([pr_sta_vl_scal[i:(i+length_batch),:,:] for i in np.arange(pr_sta_vl_scal.shape[0]-length_batch)])
    # pr_sta_te_scal=np.array([pr_sta_te_scal[i:(i+length_batch),:,:] for i in np.arange(pr_sta_te_scal.shape[0]-length_batch)])
        
    
    
    # pr_tr_oc=pr_tr_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(pr_tr_scal.shape[0],length_batch,1)
    # pr_vl_oc=pr_vl_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(pr_vl_scal.shape[0],length_batch,1)
    # pr_te_oc=pr_te_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(pr_te_scal.shape[0],length_batch,1)

    # mw_tr_oc=mw_tr_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(mw_tr_scal.shape[0],length_batch,1)
    # mw_vl_oc=mw_vl_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(mw_vl_scal.shape[0],length_batch,1)
    # mw_te_oc=mw_te_scal[:,:,idx_lat_mod,idx_lon_mod].reshape(mw_te_scal.shape[0],length_batch,1)



    #precipitation channel
    K.clear_session()

    assert pr_sta_tr_scal.shape[0] == pr_tr_scal.shape[0] == mw_tr_scal.shape[0], "Shape of batches differ"

    inputs_prec  = keras.layers.Input(shape = (pr_tr_scal.shape[1:])) 
    conv1_1      = keras.layers.Conv2D(filters = 6, kernel_size = 3, activation = 'relu',padding = 'same')(inputs_prec)
    drop1_1      = keras.layers.Dropout(0.25)(conv1_1)
    conv1_2      = keras.layers.Conv2D(filters = 6,  kernel_size = 3, activation = 'relu', padding = 'same')(drop1_1)
    drop1_2      = keras.layers.Dropout(0.25)(conv1_2)
    flatten_1    = keras.layers.Flatten()(drop1_2)
    

    #wind channel
    inputs_mw    = keras.layers.Input(shape = (mw_tr_oc.shape[1:])) 
    conv2_1      = keras.layers.Conv1D(filters = 32, kernel_size = 3, activation = 'relu', data_format="channels_first", padding = 'same')(inputs_mw)
    drop2_1      = keras.layers.Dropout(0.25)(conv2_1)
    conv2_2      = keras.layers.Conv1D(filters = 64,  kernel_size = 3, activation = 'relu', data_format="channels_first", padding = 'same')(drop2_1)
    drop2_2      = keras.layers.Dropout(0.25)(conv2_2)
    flatten_2    = keras.layers.Flatten()(drop2_2)


    #elevation channel
    ele_tri=np.repeat(np.random.randint(0,4500,size=1),mw_mod_batch.shape[0]).reshape(mw_mod_batch.shape[0],1)
    # Static Inputs
    inputs_ele   = keras.layers.Input(shape = (ele_tri.shape[1],)) 
    # conv3_1      = keras.layers.Conv1D(filters = 16, kernel_size = 5, activation = 'relu', padding = 'same')(inputs_ele)
    # drop3_1      = keras.layers.Dropout(0.3)(conv3_1)
    # conv3_2      = keras.layers.Conv1D(filters = 64,  kernel_size = 5, activation = 'relu', padding = 'same')(drop3_1)
    # drop3_2      = keras.layers.Dropout(0.3)(conv3_2)
    flatten_3    = keras.layers.Flatten()(inputs_ele)
    
    # Merge the layers 
    concatenate  = keras.layers.concatenate([flatten_1]) #,flatten_2flatten3

    # Output of the model
    dense_1      = keras.layers.Dense(256, name = "Dense_1")(concatenate)
    drop_3       = keras.layers.Dropout(0.4, name = "Drop_3")(dense_1)
    dense_2      = keras.layers.Dense(32, name = "Dense_2")(drop_3)
    drop_4       = keras.layers.Dropout(0.2, name = "Drop_4")(dense_2)
    outputs = keras.layers.Dense(1, name = "Predictions",activation='softmax')(drop_4)

    model = keras.Model(inputs=[inputs_prec], outputs = outputs) #,inputs_mw,inputs_ele
    print(model.summary())
    # tf.keras.utils.plot_model(model,show_shapes = True,show_layer_names = True,to_file="/home/lcesarini/2022_resilience/MH_monster.png")

    model.compile(
        loss      = "mse",
        optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
        metrics   = ["mae"],
    )

    from tensorflow.keras.callbacks import EarlyStopping

    # Define an EarlyStopping callback
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



    history=model.fit(pr_tr_scal,
                      pr_sta_tr_scal[length_batch:,].reshape(-1,1), #ele_tri
                      epochs=100,
                      batch_size = 8, #int(0.1*x_tr.shape[0]),
                    #   callbacks = [early_stopping_callback],
                      verbose = 1,
                      shuffle = True,
                      validation_data = (pr_vl_scal,pr_sta_vl_scal[length_batch:,].reshape(-1,1))
            )


    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    if 'mae' in history.history:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.show()




    plt.plot(model.predict(pr_te_scal), label="pred")
    plt.plot(pr_sta_te_scal[length_batch:,].reshape(-1,1), label="true")
    plt.legend()
    plt.show()



    np.where(~np.isnan(pr_mod_batch))
    np.where((pr_sta.pr.values[10,41,:] > 5) & (pr_mod_tri[:,10,41] > 5))
    np.where()
    # another cell
    pr_mod_batch[:,:,10,41]
    mw_mod_batch[:,:,10,41]
    ele_tri=np.repeat(np.random.randint(0,4500,size=1),mw_mod_batch.shape[0]).reshape(mw_mod_batch.shape[0],1)

    pr_sta.pr.values[10,41,7:].reshape(-1,1)

    st=58805
    en=58815

    prediction=model.predict([pr_mod_batch[:,:,10,41],mw_mod_batch[:,:,10,41],ele_tri[:,:]])
    prediction[(st-7):(en-7),:]

    plt.plot(pr_sta.pr.values[10,41,st:en].reshape(-1,1), "-*",color = "green",label="truth")
    plt.plot(pr_mod_tri.values[st:en,10,41].reshape(-1,1), "-^",color = "red",label="model biased")
    plt.plot(prediction[(st-7):(en-7),:], "-+",color = "blue",label="model corrected")
    # plt.plot(history.history['val_loss'], color = "red",label="val")
    plt.title("Loss training history")
    plt.legend()
    plt.savefig("../FUUUUCK.png")
    plt.close()

    prediction2=model.predict([pr_one_cell,mw_one_cell])

    plt.plot(pr_sta.pr.values[8,71,7:].reshape(-1,1), "-*",color = "green",label="truth")
    plt.plot(pr_mod_tri.values[7:,8,71].reshape(-1,1), "-^",color = "red",label="model biased")
    plt.plot(prediction2, "-+",color = "blue",label="model corrected")
    # plt.plot(history.history['val_loss'], color = "red",label="val")
    plt.title("Loss training history")
    plt.legend()
    plt.show()
    plt.savefig("../FUUUUCK.png")
    plt.close()



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

input.reshape(8760,9)
# Generate sample data (replace with your own data)
# Input data (past precipitation values)
X_train = np.random.rand(100, 10, 1)  # 100 samples, sequence length 10, 1 feature
# Target data (future precipitation values)
y_train = np.random.rand(100, 1, 1)   # 100 samples, sequence length 5, 1 feature

# Normalize the data (important for LSTM)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

# Define the model
latent_dim = 64  # Size of the latent space

# Encoder
encoder_inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
encoder = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(y_train.shape[1], y_train.shape[2]))
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.Dense(1)  # Output layer (1 feature)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model (replace with your own data and parameters)
model.fit(
    [X_train, y_train[:, :-1, :]],  # Use past values for prediction
    y_train[:, 1:, :],  # Predict future values
    batch_size=32,
    epochs=50,
    validation_split=0.2,
)

# Make predictions
# Replace 'X_test' with your test data
X_test = np.random.rand(10, 10, 1)  # Test data (10 samples, sequence length 10, 1 feature)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_pred = model.predict([X_test, np.zeros((X_test.shape[0], y_train.shape[1], 1))])

# Inverse transform the predictions to the original scale
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

# Print or use 'y_pred' for precipitation forecasting
print(y_pred)

