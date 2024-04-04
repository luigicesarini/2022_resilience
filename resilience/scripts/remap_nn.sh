#!/usr/bin/bash



# cd /home/lcesarini/2022_resilience/s
period=("ECMWF-ERAINT" "Historical" "Rcp85")
sim=('KIT' 'ETH' 'HCLIMcom' 'CMCC') #'CMCC' 'JLU' 'FZJ' 'CNRM' 'ETH' 'HCLIMcom' 'MOHC''ICTP' 'KNMI' 'KIT' 'MOHC'
variable=('pr' 'uas' 'vas')

baserotGRID="/mnt/data/gfosser/DATA_FPS/"
pathTMP="/mnt/data/lcesarini/tmp/"
commongrid="/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3"
# select period and simulations
i=1
k=0
for j in 0 1 2 3
do
echo Model ${sim[j]} on ${period[i]}
pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"
# pathIN="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr"
pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"

#check if the directory exisys
if [ ! -d ${pathOUT} ]
then
mkdir -p ${pathOUT}
fi

for file in $pathIN/*.nc
do

echo $file
# fileFIN=${sim[j]}_${period[i]}_${file##*1hr_}
done
done
# REMAP="con"
# PATH_FILE="/mnt/data/lcesarini/GRIPHO/"
# pathTMP="/mnt/data/lcesarini/tmp/"
# PATH_OUTPUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ICTP"

# for file in ${PATH_FILE}/*.nc
# do 
# fileFIN="${pathTMP}remap_${REMAP}_${file##*GRIPHO//}"
# echo Working on ${fileFIN}

# if [ ${REMAP} == "con" ]
# then
#     ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t.nc
#     cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt ${pathTMP}/t.nc ${fileFIN}
# fi
# if [ ${REMAP} == "bil" ]
# then
#     cdo remapbil,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt ${file} ${fileFIN}
# fi
# done

# # for file in ${PATH_FILE}/*.nc
# # do 
# # fileFIN="ETH_ECMWF-ERAINT${file##*1hr}"
# # echo Working on ${fileFIN}
# # # ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
# # # cdo remapcon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${PATH_OUTPUT}/${fileFIN}
# # cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_602x572_latlon_bounds_ICTP.txt /mnt/data/gfosser/DATA_FPS/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc /mnt/data/lcesarini/tmp/ttt_oro_ictp.nc
# # cdo remapycon,newcommongrid.txt /mnt/data/lcesarini/tmp/ttt_oro_ictp.nc /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ICTP/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc
# # done



# ncks -C -O -x -v lon,lat /mnt/data/commonData/OBSERVATIONS/ITALY/gripho-v1_1h_TSmin30pct_2001-2016_cut3km.nc /mnt/data/lcesarini/tmp/entire_gripho.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/entire_gripho.nc /mnt/data/lcesarini/entire_gripho_remapped.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/t.nc /mnt/data/lcesarini/rem.nc





