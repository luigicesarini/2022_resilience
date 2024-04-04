#!/usr/bin/bash

# cd /home/lcesarini/2022_resilience/s
period=("ECMWF-ERAINT" "Historical" "Rcp85")
sim=('CNRM' 'ICTP' 'MOHC' 'KNMI') #'CMCC' 'JLU' 'FZJ' 'CNRM' 'ETH' 'HCLIMcom' 'MOHC''ICTP' 'KNMI' 'KIT' 'MOHC'
variable=('pr' 'uas' 'vas')

baserotGRID="/mnt/data/gfosser/DATA_FPS/"
pathTMP="/mnt/data/lcesarini/tmp/"
commongrid="/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3"
# select period and simulations
i=2
k=0
for j in 0 1 2 3
do
echo Model ${sim[j]} on ${period[i]}
pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"
# pathIN="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr"
pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"

#check if the directory exists
if [ ! -d ${pathOUT} ]
then
mkdir -p ${pathOUT}
fi

for file in $pathIN/*.nc
do
      fileFIN=${sim[j]}_${period[i]}_${file##*1hr_} # ## removes everything in strig file before 1hr_
#     # DIFFERENT METHODS FOR EACH PAIR OF PERIOD AND SIMULATIONS
#     # 1) HCLIMcom
    if [ ${sim[j]} == 'HCLIMcom' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with HCLIMcom
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3 ${file} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}

    fi
#     # 2) MOHC
    if [ ${sim[j]} == 'MOHC' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with MOHC
        cdo mulc,3600 ${file} ${pathTMP}/ttt_${fileFIN}
        cdo chname,precipitation_flux,pr ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'ETH' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with ETH
        ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
        # ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200001010030_200012312330.nc /mnt/data/lcesarini/tmp/t_ETH_2000.nc
    fi

    if [ ${sim[j]} == 'CNRM' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with CNRM
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_587x487_latlon_bounds_CNRM_FIPS ${file} ${pathTMP}/tttt_${fileFIN}
        #To remove below zero values
        cdo gec,0 ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN}
        cdo ifthenelse ${pathTMP}/mask_${fileFIN} ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'ICTP' -a ${period[i]} == 'Rcp85' ]
    then 
    echo Working with ICTP
    cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_602x572_latlon_bounds_ICTP.txt ${file} ${pathTMP}/ttt_${fileFIN}
    cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
    cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'KIT' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with KIT
        cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/KIT/CPM/pr/pr_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
        cdo mulc,3600 ${file} ${pathTMP}/tttt_${fileFIN}
        cdo setgrid,${pathTMP}/tx.nc ${pathTMP}/tttt_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        ncks -C -O -x -v lon,lat ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'KNMI' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with KNMI
        echo File name: ${fileFIN}
        ncks -C -O -x -v date,hms ${file} ${pathTMP}/ttttt_${fileFIN}
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/gridCPM_KNMI_mod ${pathTMP}/ttttt_${fileFIN} ${pathTMP}/tttt_${fileFIN}
        #To remove below zero values
        cdo gec,0 ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN}
        cdo ifthenelse ${pathTMP}/mask_${fileFIN} ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi      

    if [ ${sim[j]} == 'CMCC' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with CMCC
        echo File name: ${fileFIN}
        cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/CMCC/CPM/pr/pr_ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CMCC-CCLM5-0-9_x2yn2v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
        ncks -C -O -x -v lon,lat ${file} ${pathTMP}/ttttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttttt_${fileFIN} ${pathTMP}/tttt_${fileFIN}
        cdo setgrid,${pathTMP}/tx.nc ${pathTMP}/tttt_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        ncks -C -O -x -v lon,lat ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi
         

    echo Remapping ${pathTMP}/t_${fileFIN}
    cdo remapycon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${pathOUT}/${fileFIN}
    # cdo remapycon,/home/lcesarini/2022_resilience/scripts/nemwcommongrid.txt /mnt/data/lcesarini/tmp/t_ETH_2000.nc /mnt/data/RESTRICTED/CARIPARO/common/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200001010030_200012312330.nc

rm -f /mnt/data/lcesarini/tmp/*.nc

done

done

# for year in {2000..2009}
# do 

# baseIN="pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_${year}01010030-${year}12312330.nc"
# baseOUT="pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_1hr_${year}01010030-${year}12312330_regrid.nc"


# done

# rm -f /mnt/data/lcesarini/tmp/*.nc


# cdo remapycon,/home/lcesarini/2022_resilience/data/empty_common_grid.nc /mnt/data/lcesarini/tmp/test_cmcc.nc /mnt/data/lcesarini/tmp/test_regridded_cmcc.nc

# ncks -C -O -x -v lon,lat /mnt/data/lcesarini/tmp/sphera_netcdf.nc /mnt/data/lcesarini/tmp/sphera_corner.nc

# cdo remapycon,/home/lcesarini/2022_resilience/data/empty_common_grid.nc  /mnt/data/lcesarini/tmp/sphera_corner.nc /mnt/data/lcesarini/tmp/test_regridded_sphera.nc

