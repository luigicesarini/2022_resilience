#!/usr/bin/bash



paths=(
    "/mnt/data/gfosser/DATA_FPS/VarFixed/ICTP/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc"\
    "/mnt/data/gfosser/DATA_FPS/VarFixed/KIT/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_CLMcom-KIT-CCLM5-0-14_fpsconv-x2yn2-v1_fx.nc"\
    "/mnt/data/gfosser/DATA_FPS/VarFixed/KNMI/CPM/sftlf_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc"
)


names=("ICTP" "KIT" "KNMI")

for i in {0..2}
do
    ./remap_sftlf.sh ${paths[i]} ${names[i]}
done

# list_mdl=("ETH" "CNRM" "KNMI" "ICTP" "HCLIMcom" "MOHC" "CMCC" "KIT") 
# for i in {0..7}
# do
#     for VAR in "mw" "uas" "vas"
#     do
#         if [ -d /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/${list_mdl[i]}/CPM/${VAR} ]
#         then
#             echo ${list_mdl[i]} has ${VAR}
#         else
#             echo Error: ${list_mdl[i]} does not have ${VAR}
#         fi
#     done
# done
   
# for file in /mnt/data/lcesarini/SPHERA/tp/*200903**m.grb2*
# do 
#     grib_to_netcdf -o ${file%%.*}.nc $file  
# done

# cd /home/lcesarini/2022_resilience/resilience/bootstrap/
# for SEAS in SON DJF MAM JJA
# do
# for M in q
#     do
#         # ./eval_boot.py -s ${SEAS} 
#         /home/lcesarini/2022_resilience/resilience/scripts/spatial_analysis.py -s ${SEAS} -m ${M}
#     done
# done

# cd /home/lcesarini/2022_resilience/resilience/scripts/
# for seas in DJF SON MAM JJA 
# do 
#     for split in SEQUENTIAL RANDOM
#     do

#         ./plot_maps_output.R q ${seas} ${split} no no
#     done
# done

# for BM in EQM QDM #MBC SDM
# do 
#     for mdl in KNMI ETH CNRM ICTP HCLIMcom #MOHC
#     do
#         mv ${BM}/${mdl}/pr/train_${mdl}_CORR_STATIONS_2005_2009.nc ${BM}/${mdl}/pr/${mdl}_CORR_STATIONS_2000_2004.nc
#         # mkdir -p /mnt/data/lcesarini/BIAS_CORRECTED/${BM}/${mdl}/pr/
#         # mkdir -p /mnt/data/lcesarini/BIAS_CORRECTED/${BM}/${mdl}/uas/
#         # mkdir -p /mnt/data/lcesarini/BIAS_CORRECTED/${BM}/${mdl}/vas/
#     done
# done


#  ./api_cmcc.py -sy 2002 -fy 2010 -sn grid_eastward_wind -ln grid_eastward_wind -pf -npf cmcc -type reanalysis -nd era5-downscaled-over-italy -pt VHR-REA_IT_1989_2020_hourly

# for file in /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/HCLIMcom/CPM/uas/*{2007,2008,2009}*.nc
# do
#     echo $file
# done


# for year in {2000..2009}
# do
# 	cdo sellonlatbox,10.38,13.1,44.7,47.1 /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_${year}01010030_${year}12312330.nc /mnt/data/lcesarini/eth_remap_${year}.nc
# done
# cd /mnt/data/lcesarini
# cdo cat $(ls *eth_remap*) eth_merged_remap.nc

# cdo timpctl,99.9  eth_merged_remap.nc -timmin  eth_merged_remap.nc -timmax  eth_merged_remap.nc q99_eth_merged_remap.nc





