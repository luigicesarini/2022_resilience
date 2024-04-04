#!/usr/bin/bash
Help()
{
   # Display Help
   echo "Scripts that creates repository folder structure at a given location provided as parameter"
   echo
   echo "Syntax: scriptTemplate [-h|p]"
   echo "options:"
   echo "h     Print this Help."
   echo "p     PATH of the original file"
   echo "m     Name of the model"

}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

pathIN=$1
mdl=$2


fileFIN=$(basename -- "$pathIN") 
pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/${mdl}/CPM/"
pathTMP="/mnt/data/lcesarini/tmp"


# if [ ${mdl} == 'CMCC' ]
# then
#     echo Working with CMCC
#     echo File name: ${fileFIN}
#     cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/CMCC/CPM/uas/uas_ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CMCC-CCLM5-0-9_x2yn2v1_1hr_199601010000-199612312300.nc ${pathTMP}/tx.nc
#     cdo setgrid,${pathTMP}/tx.nc ${pathIN} ${pathTMP}/tt_${fileFIN}
#     ncks -C -O -x -v lon,lat ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
# fi

if [ ${mdl} == 'CNRM' ]
then
    echo Working with CNRM
    echo File name: ${fileFIN}
    cdo setgrid,/mnt/data/lcesarini/grid_587x487_latlon_bounds_CNRM_FIPS ${pathIN} ${pathTMP}/t_${fileFIN}
fi

if [ ${mdl} == 'HCLIMcom'  ]
then
    echo Working with HCLIMcom
    echo File name: ${fileFIN}
    cdo setgrid,/mnt/data/lcesarini/grid_573x485_latlon_bounds_ALP-3 ${pathIN} ${pathTMP}/t_${fileFIN}

fi

if [ ${mdl} == 'ICTP'  ]
then 
    echo Working with ICTP
    echo File name: ${fileFIN}
    cdo setgrid,/mnt/data/lcesarini/grid_602x572_latlon_bounds_ICTP.txt ${pathIN} ${pathTMP}/t_${fileFIN}
fi

if [ ${mdl} == 'KIT'  ]
then
    echo Working with KIT
    echo File name: ${fileFIN}

    cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/KIT/CPM/uas/uas_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_199601010000-199612312300.nc ${pathTMP}/tx.nc
    cdo setgrid,${pathTMP}/tx.nc ${pathIN} ${pathTMP}/tt_${fileFIN}
    ncks -C -O -x -v lon,lat ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
fi

if [ ${mdl} == 'KNMI'  ]
then
    echo Working with KNMI
    echo File name: ${fileFIN}
    ncks -C -O -x -v date,hms ${pathIN} ${pathTMP}/tt_${fileFIN}
    cdo setgrid,/mnt/data/lcesarini/gridCPM_KNMI_mod ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
fi  


echo Remapping ${pathTMP}/t_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${pathOUT}/${fileFIN}


rm -f /mnt/data/lcesarini/tmp/*.nc




