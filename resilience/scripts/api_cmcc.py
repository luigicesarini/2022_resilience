#!/home/lcesarini/miniconda3/envs/detectron/bin/python
"""
File name: api_cmcc.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 12 January 2023
Date last modified: 12 January 2023

#######################################################################################
PURPOSE:
This scripts uses the CMCC api to automate the download of data.
Iterates over the years.
For name of the variables and product type refer to the following webpage looking 
for the right dataset:
https://dds.cmcc.it/#/

IMPORTANT!! 
To run the code is necessary to install the DDS API key
Refer to:
https://dds.cmcc.it/#/docs
"""
import os
import ddsapi
import argparse
import numpy as np

c = ddsapi.Client()
#./api_cmcc.py -sy 1989 -fy 2021 -sn air_temperature -ln air_temperature -l single -pf -npf cmcc
parser = argparse.ArgumentParser()
parser.add_argument("-sy","--start_year",
                    help="Starting year of the Download", type = int,
                    required=True,default=1981)
parser.add_argument("-fy","--finish_year",
                    help="Finishing year of the Download", type = int,
                    required=True,default=2020)
"""
Possible values:
"air_temperature"="T2_M",
"precipitation_amount",
"grid_eastward_wind",
"grid_northward_wind"
"""                    
parser.add_argument("-sn", "--short_name", type = str,
                    help="Short name of the variable (it will also be the name of the folder and the prefix of the filename)",
                    required=True,default="tp"
                    )
parser.add_argument("-ln", "--long_name_var", type = str,
                    help="Name of the variable as specified by CDS (e.g 10m_u_component_of_wind) ",
                    required=False,default="tp"
                    )
parser.add_argument("-l","--levels", type = str,
                    help="single or pressure?",
                    required=False,default="single",
                    choices = ["single","pressure"])
parser.add_argument("-pf","--parent_folder", 
                    help="Save the data in a specific project folder?",
                    action="store_true",
                    required=True,
                    default=True)
parser.add_argument("-npf","--name_parent_folder", type = str,
                    help="Parent folder of the project that requires to save the variables",
                    required=False,default="resilience")
parser.add_argument("-type", type = str,
                    help="Which type of data to download",
                    required=False,default="reanalysis",
                    choices=["reanalysis","projection"])
parser.add_argument("-per","--period", type = str,
                    help="Which period needs to be downloaded",
                    required=False,default='',
                    choices=["historical","rcp45","rcp85"])
parser.add_argument("-nd","--name_dataset", type = str,
                    help="Which dataset to download?",
                    required=False,default="era5-downscaled-over-italy",
                    choices=["era5-downscaled-over-italy","climate-projections-8km-over-italy"])

parser.add_argument("-pt","--product_type", type = str,
                    help="Which dataset to download?",
                    required=False,default="VHR-REA_IT_1989_2020_hourly",
                    choices=["VHR-REA_IT_1989_2020_hourly","historical","rcp45","rcp85"])
args = parser.parse_args()


start_year  = args.start_year
finish_year = args.finish_year+1

short_name_var = args.short_name
long_name_var  = args.long_name_var
levels         = args.levels

year = np.arange(start_year,finish_year)


# Create directory if it doesn't exists
if args.parent_folder:
    if not(os.path.isdir(f'/mnt/data/RESTRICTED/CARIPARO/{args.name_parent_folder}/{args.type}/{args.period}/{short_name_var}')):
            os.makedirs(f'/mnt/data/RESTRICTED/CARIPARO/{args.name_parent_folder}/{args.type}/{args.period}/{short_name_var}')
else:
    if not(os.path.isdir(f'/mnt/data/lcesarini/{short_name_var}/')):
        os.mkdir(f'/mnt/data/lcesarini/{short_name_var}/')


list_month=["1","2","3","4","5","6","7","8","9","10","11","12"]

for i in np.arange(year.shape[0]):
    for mm in list_month:
        print(year[i])
        c.retrieve(
            args.name_dataset,
    {
        "product_type": args.product_type,
        "area": {
        #CHANGE BBOX OF DATA
            "north": 47.6,
            "south": 43.25,
            "east": 6.5,
            "west": 14
        },
        "time": {
            "hour": [
                "00","01","02","03","04","05",
                "06","07","08","09","10","11",
                "12","13","14","15","16","17",
                "18","19","20","21","22","23"
            ],
            "year": [
                f"{year[i]}"
            ],
            "month":f"{mm}" ,
            "day": [
                "1","2","3","4","5","6",
                "7","8","9","10","11","12",
                "13","14","15","16","17","18",
                "19","20","21","22","23","24",
                "25","26","27","28","29","30",
                "31"
            ]
        },
        "variable": [
            f"{short_name_var}"
        ],
        "format": "netcdf"
    },
    # f'/mnt/data/RESTRICTED/CARIPARO/cmcc/{args.name_parent_folder}/{short_name_var}/era5-downscaled-over-italy-VHR-REA_IT_{year[i]}_hourly.nc' if args.parent_folder else\
    #     f'/mnt/data/lcesarini/{short_name_var}/era5-downscaled-over-italy-VHR-REA_IT_{year[i]}_hourly.nc'

    f'/mnt/data/RESTRICTED/CARIPARO/{args.name_parent_folder}/{args.type}/{args.period}/{short_name_var}/{args.name_dataset}_{args.product_type}_{year[i]}{mm}_hourly.nc' if args.parent_folder else\
        f'/mnt/data/lcesarini/{short_name_var}/{args.name_dataset}_{args.product_type}_{year[i]}{mm}_hourly.nc'

    )

                
            

