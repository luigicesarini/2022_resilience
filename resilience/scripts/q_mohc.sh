#!/usr/bin/bash

for file in /mnt/data/lcesarini/EVENTS/pr/*WH*
do 
# Replace spaces with underscores
new_string=$(echo "$file" | sed 's/\0.9/90/')
echo ${new_string}
mv ${file} ${new_string}
done


# for yr in {2000..2009}
# do 
# pathIN="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/reanalysis/SPHERA/wind_dir/"
# 	for file in $pathIN/*$yr*
# 	do
# 		fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
# 		new_string=$(echo "$fileFIN" | sed 's/\.grb2/.nc/')
# 		# echo "$new_string"
	
# 		grib_to_netcdf -o ${pathIN}/${new_string}.nc ${pathIN}/${fileFIN}
# 	done
# done

# for j in {0..9}
# do 
# echo $j
# done
# cdo rotuvNorth




# cdo expr,'' output/JJA/ENSEMBLE_q.nc

# for year in {2000..2009}
# do
# 	cdo remapnn,lon=12.42374/lat=46.65285 /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_${year}01010030_${year}12312330.nc /mnt/data/lcesarini/eth_remap_${year}_VE_0091.nc
# done
# # cd /home/lcesarini/2022_resilience/s
# period=("ECMWF-ERAINT" "historical" "rcp85")
# sim=('CNRM' 'ETH' 'HCLIMcom' 'MOHC' 'KIT' 'ICTP' 'KNMI' 'CMCC') #'CMCC' 'JLU' 'FZJ' 
# variable=('pr' 'uas' 'vas' 'wsgsmax')

# baserotGRID="/mnt/data/gfosser/DATA_FPS/"
# pathTMP="/mnt/data/lcesarini/tmp/"
# commongrid="/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3"
# # select period and simulations
# i=0
# k=3
# echo Working on ${variable[k]}
# for j in 0 2 4 6
# do
# pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"
# # pathIN="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr"
# pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"

# #check if the directory exisys
# if [ ! -d ${pathOUT} ]
# then
# mkdir -p ${pathOUT}
# fi

# for file in $pathIN/*{2000,2001,2002,2003,2004,2005,2006,2007,2008,2009}*.nc
# do
# echo ${file}
# done
# done
# # time ./metrics_wind.py 20
# # time ./evaluation_metrics_wind.py 20
# # file="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/MOHC/CPM/uas/uas_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_day_20000101-20001231.nc"
# # echo $file
# # echo ${file##*day}

# # for model in ETH HCLIMcom CNRM MOHC KNMI ICTP
# # do
# #     list_files="ls /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/${model}/CPM/uas/"
# #     ${list_files}
# # done
# # cd /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/MOHC/CPM/pr/

# # list_files=$(ls pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_1hr_2004*)

# # echo Working...probably

# # echo $list_files
# # cdo cat $list_files -timmin $list_files -timmax $list_files /mnt/data/lcesarini/tmp/cat_2004_mohc.nc


