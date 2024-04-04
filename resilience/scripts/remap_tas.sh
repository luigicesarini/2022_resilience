#!/usr/bin/bash
Help()
{
   # Display Help
   echo "Scripts that creates repository folder structure at a given location provided as parameter"
   echo
   echo "Syntax: scriptTemplate [-h|m]"
   echo "options:"
   echo "h     Print this Help."
   echo "m     Name of the model"
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

#ETH, KIT, CMCC, CNRM, HCLIMcom, KNMI, IDL-WRF381BH, IBG3-WRF381B, ICTP

NAME_MODEL=$1
echo ${NAME_MODEL}

# cd /home/lcesarini/2022_resilience/
period=("ECMWF-ERAINT") # "Historical" "Rcp85"
sim=( 'HCLIMcom' 'KNMI' 'IDL-WRF381BH' 'IBG3-WRF381B' 'ICTP' 'ETH' 'KIT' 'CMCC' 'CNRM') 
variable=('tas') # 'pr' 'uas' 'vas'

PATH_DATA="/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS"
baserotGRID="/mnt/data/gfosser/DATA_FPS/"
pathTMP="/mnt/data/lcesarini/tmp/"
# commongrid="/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3"
# select period and simulations
k=0
for i in 0 
do
    for j in 8
    do
        echo Model ${sim[j]} on ${period[i]} for ${variable[k]}
        if [ ${sim[j]} == 'FZJ' ]
        then
            pathIN="${PATH_DATA}/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"
            pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"
        elif [ ! ${sim[j]} == 'FZJ' ]
        then
            pathIN="${PATH_DATA}/${period[i]}/${sim[j]}/CPM/${variable[k]}"
            # pathIN="/mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/CPM/pr"
            pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"
        fi
        #check if the directory exisys
        if [ ! -d ${pathOUT} ]
        then
        mkdir -p ${pathOUT}
        fi

        for file in $pathIN/*.nc
        do
            fileFIN=${sim[j]}_${period[i]}_${file##*1hr_} # ## removes everything in strig file before 1hr_
        #     # DIFFERENT METHODS FOR EACH PAIR OF PERIOD AND SIMULATIONS
        #     # 1) HCLIMcom
            if [ ${sim[j]} == 'FZJ' ]
            then
                echo Working with FZJ
                ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
            fi
            if [ ${sim[j]} == 'HCLIMcom' ]
            then
                echo Working with HCLIMcom
                cdo setgrid,${PATH_DATA}/VarFixed/HCLIMcom/CPM/grid_573x485_latlon_bounds_ALP-3_HClim ${file} ${pathTMP}/t_${fileFIN}

            fi
        #     # 2) MOHC
            if [ ${sim[j]} == 'MOHC' ]
            then
                echo Working with MOHC
            fi

            if [ ${sim[j]} == 'ETH' ]
            then
                echo Working with ETH ${file}
                ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
            fi

            if [ ${sim[j]} == 'CNRM' ]
            then
                echo Working with CNRM
                cdo setgrid,${PATH_DATA}/VarFixed/CNRM/CPM/grid_587x487_latlon_bounds_CNRM_FIPS ${file} ${pathTMP}/t_${fileFIN}
            fi

            if [ ${sim[j]} == 'ICTP' ]
            then 
            echo Working with ICTP
            cdo setgrid,${PATH_DATA}/VarFixed/ICTP/CPM/grid_602x572_latlon_bounds_ICTP.txt ${file} ${pathTMP}/t_${fileFIN}
            fi

            if [ ${sim[j]} == 'KIT' ]
            then
                echo Working with KIT
                cdo seltimestep,1 ${PATH_DATA}/Historical/KIT/CPM/uas/uas_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_199601010000-199612312300.nc ${pathTMP}/tx.nc
                cdo setgrid,${pathTMP}/tx.nc ${file} ${pathTMP}/tt_${fileFIN}
                ncks -C -O -x -v lon,lat ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
            fi

            if [ ${sim[j]} == 'KNMI' ]
            then
                echo Working with KNMI
                echo File name: ${fileFIN}
                ncks -C -O -x -v date,hms ${file} ${pathTMP}/tt_${fileFIN}
                cdo setgrid,${PATH_DATA}/VarFixed/KNMI/CPM/gridCPM_KNMI_mod ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
            fi      

            if [ ${sim[j]} == 'CMCC' ]
            then
                echo Working with CMCC
                echo File name: ${fileFIN}
                cdo seltimestep,1 ${PATH_DATA}/Historical/CMCC/CPM/uas/uas_ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CMCC-CCLM5-0-9_x2yn2v1_1hr_199601010000-199612312300.nc ${pathTMP}/tx.nc
                ncks -C -O -x -v lon,lat ${file} ${pathTMP}/ttt_${fileFIN}
                cdo setgrid,${pathTMP}/tx.nc ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
                ncks -C -O -x -v lon,lat ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
            fi
            
            if [ ${sim[tp]} == 'JLU' ]
            then
                echo Working with JLU
                cp ${file} tmp/t_${fileFIN}
            fi

                
            #!/bin/bash

            file_path=${pathOUT}/${fileFIN}

            if [ -e "$file_path" ]; then
                echo "File exists. Proceeding with the operation."
                continue
                # Your code here
            else
                echo "File does not exist. Exiting."
                echo Remapping ${pathTMP}/t_${fileFIN}
                cdo remapycon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${pathOUT}/${fileFIN}
                # Optionally, you may want to exit the script or handle this case in some way.
            fi

            # cdo remapycon,/home/lcesarini/2022_resilience/scripts/nemwcommongrid.txt /mnt/data/lcesarini/tmp/t_ETH_2000.nc /mnt/data/RESTRICTED/CARIPARO/common/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200001010030_200012312330.nc

        rm -f /mnt/data/lcesarini/tmp/*.nc

        done

    done
done
