#!/usr/bin/bash

GRID_CMCC="/home/lcesarini/2022_resilience/data/grid_CMCC_VHR.txt"
PATH_FILE="/mnt/data/RESTRICTED/CARIPARO/cmcc/reanalysis/precipitation_amount/"
pathTMP="/mnt/data/lcesarini/tmp/"
PATH_OUTPUT="/mnt/data/RESTRICTED/CARIPARO/cmcc/remap/reanalysis/pr"
#for file in ${PATH_FILE}/era5-downscaled-over-italy_VHR-REA_IT_1989_2020_hourly_200*_hourly.nc
for file in ${PATH_FILE}/era5-downscaled-over-italy_VHR-REA_IT_1989_2020_hourly_200*_hourly.nc
do 
fileFIN=${file##*precipitation_amount//}
echo Working on ${fileFIN}
cdo setgrid,${GRID_CMCC} ${file} ${pathTMP}/ttt_${fileFIN}
cdo chname,TOT_PREC,pr ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
cdo remapycon,/home/lcesarini/2022_resilience/scripts/newcommongrid.txt ${pathTMP}/t_${fileFIN} ${PATH_OUTPUT}/${fileFIN}
done
#