import os
import time
import pickle
from glob import glob
import matplotlib 
import numpy as np
import scipy.io 
import pandas as pd


def unpack_mat(path_mat:str,path_output:str=f"/home/lcesarini/2022_resilience/stations",
               save_json:bool=True,save_csv:bool=False):
    """
    Unpack matlab file into something readable from other software's
    
    @path_mat: string obj oath to matlab's file.
    """
    # Load the file
    load_file = scipy.io.loadmat(path_mat)
    # Create dataframe from the content
    df_content = pd.DataFrame(load_file['S'][0])
 
    if not os.path.exists(path_output): os.makedirs(path_output)

    if save_json and not save_csv:
        dict_content = {
            'tp'    :[df_content['vals_mm'][0]],
            'coords':[df_content['xy_utm'][0]],
            'name'  :[df_content['name'][0]],
            'elv'   :[df_content['elev_m'][0]],
            'time'  :[df_content['time_utc'][0]]
        }
        if not os.path.exists(path_output+"/json"): os.makedirs(path_output+"/json")
        np.save(f"{path_output}/json/data_{df_content['name'][0].item()}.npy", dict_content)

    elif save_csv and not save_json:
        if not os.path.isfile(f'{path_output}/text/prec_{df_content["name"][0].item()}.csv'):
            cell_n=pd.DataFrame(df_content['vals_mm'][0])
            cell_n.rename(columns={0:df_content['name'][0].item()}, inplace=True)


            cell_n.to_csv(f'{path_output}/text/prec_{df_content["name"][0].item()}.csv',
                                            index = False,
                                            index_label=False, 
                                            mode = 'a')

        if not os.path.isfile(f'{path_output}/meta_stations.csv'):

            meta_df=pd.DataFrame([df_content['xy_utm'][0][0][0],df_content['xy_utm'][0][0][1],
                                  df_content['name'][0].item(),
                                  df_content['elev_m'][0].item()]).transpose()

            meta_df.rename(columns={0:"lon",1:"lat",2:"name",3:"elv"},inplace=True)
            
            meta_df.to_csv(f'{path_output}/meta_stations.csv',
                           columns = ['lon','lat','name','elv'], 
                           header = True, 
                           index = False,
                           index_label=False, 
                           mode = 'a'
                           )
        else:
            meta_df=pd.DataFrame([df_content['xy_utm'][0][0][0],df_content['xy_utm'][0][0][1],
                        df_content['name'][0].item(),
                        df_content['elev_m'][0].item()]).transpose()

            meta_df.rename(columns={0:"lon",1:"lat",2:"name",3:"elv"},inplace=True)


            pd.DataFrame(meta_df, index = [0]).to_csv(f'{path_output}/meta_stations.csv',
                                                    header = False, 
                                                    index = False,
                                                    index_label=False, 
                                                    mode = 'a')

    elif save_csv and save_json:
        if not os.path.isfile(f'{path_output}/prec_{df_content["name"][0].item()}.csv'):
            cell_n=pd.DataFrame(df_content['vals_mm'][0])
            cell_n.rename(columns={0:df_content['name'][0].item()}, inplace=True)


            cell_n.to_csv(f'{path_output}/text/prec_{df_content["name"][0].item()}.csv',
                                            index = False,
                                            index_label=False, 
                                            mode = 'a')
        if not os.path.isfile(f'{path_output}/meta_stations.csv'):

            meta_df=pd.DataFrame([df_content['xy_utm'][0][0][0],df_content['xy_utm'][0][0][1],
                                  df_content['name'][0].item(),
                                  df_content['elev_m'][0].item()]).transpose()

            meta_df.rename(columns={0:"lon",1:"lat",2:"name",3:"elv"},inplace=True)
            
            meta_df.to_csv(f'{path_output}/meta_stations.csv',
                           columns = ['lon','lat','name','elv'], 
                           header = True, 
                           index = False,
                           index_label=False, 
                           mode = 'a')


        else:
            meta_df=pd.DataFrame([df_content['xy_utm'][0][0][0],df_content['xy_utm'][0][0][1],
                        df_content['name'][0].item(),
                        df_content['elev_m'][0].item()]).transpose()

            meta_df.rename(columns={0:"lon",1:"lat",2:"name",3:"elv"},inplace=True)


            pd.DataFrame(meta_df, index = [0]).to_csv(f'{path_output}/meta_stations.csv',
                                                    header = False, 
                                                    index = False,
                                                    index_label=False, 
                                                    mode = 'a')
                                                    
        dict_content = {
            'tp'    :df_content['vals_mm'][0],
            'coords':df_content['xy_utm'][0],
            'name'  :df_content['name'][0],
            'time'   :df_content['time_utc'][0]
        }
        if os.path.exists(path_output+"/json"): os.makedirs(path_output+"/json")
        np.save(f"{path_output}/json/data_{df_content['name'][0].item()}.npy", dict_content)




PATH_DATA="/mnt/data/RESTRICTED/CARIPARO/cleaned_data/rain_gauge_data_1h_cleaned"

list_files=glob(PATH_DATA+"/*.mat")



# import matplotlib.pyplot as plt
# if(False):
#     plt.figure(figsize=(14,10))
#     plt.plot(df['vals_mm'][0],'o')
#     plt.savefig(f"2022_resilience/figures/test.png")

# print(df['vals_mm'][0].min())
# print(df['time_utc'][0])
# print(df['xy_utm'][0])
# print(df['elev_m'][0])
# print(df['name'][0])


# # data = f.get('data/variable1')



def test_time():
    iterations=10
    format = "text"
    start=time.time()
    for _ in np.arange(iterations):
        if format == "json":
            [np.load(file, allow_pickle=True) for file in glob(f"/home/lcesarini/2022_resilience/stations/{format}/*")]
        elif format == "text":
            [pd.read_csv(file) for file in glob(f"/home/lcesarini/2022_resilience/stations/{format}/*")]
    end=time.time()

    print(f"Takes {round(end-start,3)} seconds to read all files {format} for {iterations} iterations")

def unpack():
    import time
    start=time.time()
    for file in list_files:
        unpack_mat(path_mat=file,
                   path_output=f"/home/lcesarini/2022_resilience/stations",
                   save_json=True,
                   save_csv=False)
    end=time.time()

    print(f"Takes {round(end-start,3)} seconds to unpack")

def main():
    dict = np.load(f"/home/lcesarini/2022_resilience/stations/json/data_VE_0248.npy", allow_pickle=True)

    print( (dict.shape))


if __name__=="__main__":
    main()
