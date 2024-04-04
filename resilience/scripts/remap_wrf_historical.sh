#!/usr/bin/bash

Help()
{
   # Display Help
   echo "Scripts that creates repository folder structure at a given location provided as parameter"
   echo
   echo "Syntax: scriptTemplate [-h|m|v]"
   echo "options:"
   echo "h     Print this Help."
   echo "m     Name of the model"
   echo "v     Name of the environmental variable"
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

NAME_MODEL=$1
VAR=$2
echo ${NAME_MODEL}
echo ${VAR}
##### Historical/FZJ-IDL-WRF381DA ######
if [ ${NAME_MODEL} == "FZJ-IDL-WRF381DA" ]
then
    pathIN="/mnt/data/gfosser/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr"
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr"
    pathTMP="/mnt/data/lcesarini/tmp"

    GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"

    #check if the directory exisys
    if [ ! -d ${pathOUT} ]
    then
    mkdir -p ${pathOUT}
    fi

    # ECMWF-ERAINT/IDL-WRF381BH
    for file in $pathIN/*.nc
    do
        fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
        echo ${fileFIN}

        cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN} 
        ncks -C -O -x -v lon,lat  ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN}
        cdo mulc,3600 ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
        cdo remapycon,${GRID} ${pathTMP}/s4_${fileFIN} ${pathOUT}/${fileFIN}
    done
fi



##### Historical/FZJ-IBG3-WRF381CA ######
if [ ${NAME_MODEL} == "FZJ-IBG3-WRF381CA" ]
then
    pathIN="/mnt/data/gfosser/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr"
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr"
    pathTMP="/mnt/data/lcesarini/tmp"

    GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"

    #check if the directory exisys
    if [ ! -d ${pathOUT} ]
    then
    mkdir -p ${pathOUT}
    fi

    # ECMWF-ERAINT/IDL-WRF381BH
    for file in $pathIN/*.nc
    do
        fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
        echo ${fileFIN}

        cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN} 
        ncks -C -O -x -v lon,lat  ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN}
        cdo mulc,3600 ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
        cdo remapycon,${GRID} ${pathTMP}/s4_${fileFIN} ${pathOUT}/${fileFIN}
    done
fi

##### ECMWF-ERAINT/IDL-WRF381BH ######
if [ ${NAME_MODEL} == "ECMWF-ERAINT/IDL-WRF381BH" ]
then
    pathIN="/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/"${VAR}
    echo ${pathIN}
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/"${VAR}
    pathTMP="/mnt/data/lcesarini/tmp"

    GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"

    #check if the directory exists
    if [ ! -d ${pathOUT} ]
    then
    mkdir -p ${pathOUT}
    fi

    if [ ${VAR} == "pr" ]
    then
        # ECMWF-ERAINT/IDL-WRF381BH
        for file in $pathIN/*.nc
        do
            fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
            echo ${fileFIN}

            ncatted -h -a _FillValue,pr,o,f,1e+20 ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN}
            cdo setgrid,/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN} 
            ncks -C -O -x -v lon,lat  ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
            cdo mulc,3600 ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
            cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s4_${fileFIN} ${pathTMP}/s5_${fileFIN}
            cdo remapycon,${GRID} ${pathTMP}/s5_${fileFIN} ${pathOUT}/${fileFIN}
        done
    elif [ ${VAR} == "tas" ]
    then
        # ECMWF-ERAINT/IDL-WRF381BH
        for file in $pathIN/*.nc
        do
            fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
            echo ${fileFIN}

            file_path=${pathOUT}/${fileFIN}

            if [ -e "$file_path" ]; then
                echo "File exists. Proceeding with the operation."
                continue
                # Your code here
            else
                echo "File does not exist. Exiting."
                echo Remapping ${pathOUT}/${fileFIN}
                cdo setgrid,/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN} 
                ncks -C -O -x -v lon,lat  ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN}
                cdo remapycon,${GRID} ${pathTMP}/s2_${fileFIN} ${pathOUT}/${fileFIN}
            fi
        done
    fi
fi

##### ECMWF-ERAINT/IBG3-WRF381BB ######
if [ ${NAME_MODEL} == "ECMWF-ERAINT/IBG3-WRF381BB" ]
then
    pathIN="/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/ECMWF-ERAINT/IBG3-WRF381BB/CPM/"${VAR}
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/IBG3-WRF381BB/CPM/"${VAR}
    pathTMP="/mnt/data/lcesarini/tmp"

    GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"

    #check if the directory exists
    if [ ! -d ${pathOUT} ]
    then
    mkdir -p ${pathOUT}
    fi

    if [ ${VAR} == "pr" ]
    then
        # ECMWF-ERAINT/IDL-WRF381BH
        for file in $pathIN/*.nc
        do
            fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
            echo ${fileFIN}

            # ncatted -h -a _FillValue,pr,o,f,1e+20 ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN}
            cdo setgrid,/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathIN}/${fileFIN} ${pathTMP}/s2_${fileFIN} 
            ncks -C -O -x -v lon,lat  ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
            cdo mulc,3600 ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
            cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s4_${fileFIN} ${pathTMP}/s5_${fileFIN}
            cdo remapycon,${GRID} ${pathTMP}/s5_${fileFIN} ${pathOUT}/${fileFIN}
        done
    elif [ ${VAR} == "tas" ]
    then
        # ECMWF-ERAINT/IDL-WRF381BH
        for file in $pathIN/*.nc
        do

            fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
            echo ${fileFIN}

            file_path=${pathOUT}/${fileFIN}

            if [ -e "$file_path" ]; then
                echo "File exists. Proceeding with the operation."
                continue
                # Your code here
            else
                echo "File does not exist. Exiting."
                echo Remapping ${pathOUT}/${fileFIN}
                cdo setgrid,/mnt/data/RESTRICTED/GROUP02/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN} 
                ncks -C -O -x -v lon,lat  ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN}
                cdo remapycon,${GRID} ${pathTMP}/s2_${fileFIN} ${pathOUT}/${fileFIN}
            fi
            
        done
    fi
fi

##### Rcp85/IDL-WRF381CA ######
if [ ${NAME_MODEL} == "Rcp85/IDL-WRF381CA" ]
then
    pathIN="/mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr"
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr"
    pathTMP="/mnt/data/lcesarini/tmp"

    GRID="/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt"

    #check if the directory exists
    if [ ! -d ${pathOUT} ]
    then
    mkdir -p ${pathOUT}
    fi

    # ECMWF-ERAINT/IDL-WRF381BH
    for file in /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209401010030-209412312330.nc #$pathIN/*1994*
    do
        fileFIN=$(basename -- "$file") # ## removes everything in strig file before 1hr_
        echo ${fileFIN}

        ncatted -h -a _FillValue,pr,o,f,1e+20 ${pathIN}/${fileFIN} ${pathTMP}/s1_${fileFIN}
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN} 
        ncks -C -O -x -v lon,lat  ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
        cdo mulc,3600 ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s4_${fileFIN} ${pathTMP}/s5_${fileFIN}
        cdo remapycon,${GRID} ${pathTMP}/s5_${fileFIN} ${pathOUT}/${fileFIN}
    done
fi

# Here:

# -h preserves the original file and creates a new one (output.nc).
# -a specifies the attribute operation:
# units is the name of the attribute you want to modify.
# var_name is the variable to which the attribute belongs.
# o,c indicates that the attribute type is "o" (opaque) and "c" (character).
# "New units" is the new value for the units attribute.
# input.nc is the input file.

rm -rf /mnt/data/lcesarini/tmp/*.nc
