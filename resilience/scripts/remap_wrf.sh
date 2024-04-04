#!/usr/bin/bash

pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"
pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"

# ECMWF-ERAINT/IDL-WRF381BH
cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_200001010030-200012312330.nc /mnt/data/lcesarini/tmp/s1.nc
ncks -C -O -x -v lon,lat  /mnt/data/lcesarini/tmp/s1.nc /mnt/data/lcesarini/tmp/s2.nc
cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/s2.nc /mnt/data/lcesarini/tmp/s3.nc 


cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/wrf_ecmwf.nc /mnt/data/lcesarini/tmp/wrf_ecmwf_remap.nc


# cp  /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_208901010030-208912312330.nc /mnt/data/lcesarini/tmp/ttttttt_FJZ.nc
# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209801010030-209812312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
# cdo mulc,3600 tmp/ttt_${fileFIN} tmp/tt_${fileFIN}
# cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' tmp/tt_${fileFIN} tmp/t_${fileFIN}
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_208901010030-208912312330.nc /mnt/data/lcesarini/tmp/tttt_FJZ.nc

#PROBLEM WITH _FillValue
cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IDL /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_IDL-WRF381BH_fpsconv-x1n2-v1_1hr_199901010030-199912312330.nc /mnt/data/lcesarini/tmp/s4.nc
ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_IDL-WRF381BH_fpsconv-x1n2-v1_1hr_199901010030-199912312330.nc /mnt/data/lcesarini/tmp/ttssasat_FJZ.nc

#PROBLEM WITH WORKS
cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_200001010030-200012312330.nc /mnt/data/lcesarini/tmp/s1.nc
ncks -C -O -x -v lon,lat  /mnt/data/lcesarini/tmp/s1.nc /mnt/data/lcesarini/tmp/s2.nc
cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/s2.nc /mnt/data/lcesarini/tmp/s3.nc 

#WORKS
ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12_FZJ-IBG3-WRF381CA_v1_1hr_199801010030-199812312330.nc /mnt/data/lcesarini/tmp/s1.nc
cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12_FZJ-IBG3-WRF381CA_v1_1hr_199801010030-199812312330.nc /mnt/data/lcesarini/tmp/s2.nc

# # FZJ-IDL-WRF381DA WORKS
# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/FZJ-IDL-WRF381DA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_209301010030-209312312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
# cdo mulc,3600 /mnt/data/lcesarini/tmp/ttt_FJZ.nc /mnt/data/lcesarini/tmp/tt_FJZ.nc
# cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' /mnt/data/lcesarini/tmp/tt_FJZ.nc /mnt/data/lcesarini/tmp/t_FJZ.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/t_FJZ.nc /mnt/data/lcesarini/tmp//pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_209301010030-209312312330.nc

ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209101010030-209112312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
cp /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209101010030-209112312330.nc /mnt/data/lcesarini/tmp/sss.nc

ncatted -O -a _FillValue,pr,d,, /mnt/data/lcesarini/tmp/sss.nc

cp /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_IDL-WRF381BH_fpsconv-x1n2-v1_1hr_199901010030-199912312330.nc /mnt/data/lcesarini/tmp/s1.nc

ncatted    -a _FillValue,,o,d,-9.99999979021476795361e+33 /mnt/data/lcesarini/tmp/s1.nc -o /mnt/data/lcesarini/tmp/copi2.nc
ncatted    -a missing_value,,o,d,-9.99999979021476795361e+33 /mnt/data/lcesarini/tmp/copi2.nc -o /mnt/data/lcesarini/tmp/copi3.nc

cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IDL /mnt/data/lcesarini/tmp/copi3.nc /mnt/data/lcesarini/tmp/copi4.nc
ncatted    -a _FillValue,,o,d,-9.99999979021476795361e+33 /mnt/data/lcesarini/tmp/s1.nc -o /mnt/data/lcesarini/tmp/copi2.nc
ncatted    -a missing_value,,o,d,-9.99999979021476795361e+33 /mnt/data/lcesarini/tmp/copi2.nc -o /mnt/data/lcesarini/tmp/copi3.nc
ncks -C -O -x -v lon,lat /mnt/data/lcesarini/tmp/copi4.nc /mnt/data/lcesarini/tmp/copi5.nc
cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/copi5.nc /mnt/data/lcesarini/tmp/copi6.nc 

# cd /home/lcesarini/2022_resilience/
period=("ECMWF-ERAINT" "historical" "Rcp85")
sim=( 'FZJ' ) #'CMCC' 'JLU' 'FZJ' 'CNRM' 'ETH' 'HCLIMcom' 'MOHC''ICTP' 'KNMI' 'KIT' 'MOHC'
variable=('tas' 'pr' 'uas' 'vas')

baserotGRID="/mnt/data/gfosser/DATA_FPS/"
pathTMP="/mnt/data/lcesarini/tmp/"
commongrid="/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3"
# select period and simulations
i=2
k=0
for j in 0 
do
echo Model ${sim[j]} on ${period[i]}
if [ ${sim[j]} == 'FZJ' ]
then
    pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"
    pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/${period[i]}/FZJ-IDL-WRF381DA/CPM/${variable[k]}"
elif [ ! ${sim[j]} == 'FZJ' ]
then
    pathIN="/mnt/data/gfosser/DATA_FPS/${period[i]}/${sim[j]}/CPM/${variable[k]}"
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
    if [ ${sim[j]} == 'FZJ' -a ${period[i]} == 'Rcp85' ]
    then
        echo Working with FZJ
        ncks -C -O -x -v lon,lat ${file} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi
    if [ ${sim[j]} == 'HCLIMcom' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then
        echo Working with HCLIMcom
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3 ${file} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}

    fi
#     # 2) MOHC
    if [ ${sim[j]} == 'MOHC' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then
        echo Working with MOHC
        cdo mulc,3600 ${file} ${pathTMP}/ttt_${fileFIN}
        cdo chname,precipitation_flux,pr ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'ETH' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then
        echo Working with ETH
        ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
        # ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_COSMO-pompa_5.0_2019.1_1hr_200001010030_200012312330.nc /mnt/data/lcesarini/tmp/t_ETH_2000.nc
    fi

    if [ ${sim[j]} == 'CNRM' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then
        echo Working with CNRM
        cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_587x487_latlon_bounds_CNRM_FIPS ${file} ${pathTMP}/tttt_${fileFIN}
        #To remove below zero values
        cdo gec,0 ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN}
        cdo ifthenelse ${pathTMP}/mask_${fileFIN} ${pathTMP}/tttt_${fileFIN} ${pathTMP}/mask_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'ICTP' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then 
    echo Working with ICTP
    cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_602x572_latlon_bounds_ICTP.txt ${file} ${pathTMP}/ttt_${fileFIN}
    cdo mulc,3600 ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
    cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'KIT' -a ${period[i]} == 'ECMWF-ERAINT' ]
    then
        echo Working with KIT
        cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/KIT/CPM/pr/pr_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
        cdo mulc,3600 ${file} ${pathTMP}/tttt_${fileFIN}
        cdo setgrid,${pathTMP}/tx.nc ${pathTMP}/tttt_${fileFIN} ${pathTMP}/ttt_${fileFIN}
        ncks -C -O -x -v lon,lat ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
        cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/tt_${fileFIN} ${pathTMP}/t_${fileFIN}
    fi

    if [ ${sim[j]} == 'KNMI' -a ${period[i]} == 'ECMWF-ERAINT' ]
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

    if [ ${sim[j]} == 'CMCC' -a ${period[i]} == 'ECMWF-ERAINT' ]
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
    
    if [ ${sim[tp]} == 'JLU' ]
    then
        echo Working with JLU
        cp ${file} tmp/t_${fileFIN}
    fi

         
    file_path=${pathOUT}/${fileFIN}

    if [ -e "$file_path" ]; then
        echo "File exists. Proceeding with the operation."
        continue
        # Your code here
    else
        echo "File does not exist. Exiting."
        echo Remapping ${pathTMP}/t_${fileFIN}
        cdo remapycon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${pathOUT}/${fileFIN}
        # cdo remapycon,/home/lcesarini/2022_resilience/scripts/nemwcommongrid.txt /mnt/data/lcesarini/tmp/t_ETH_2000.nc /mnt/data/RESTRICTED/CARIPARO/common/ECMWF-ERAINT/ETH/CPM/pr/ETH_ECMWF-ERAINT_200001010030_200012312330.nc
    fi

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

# cdo seltimestep,1111 /mnt/data/lcesarini/gripho_clipped.nc /mnt/data/lcesarini/tx2.nc
# cdo runmax,10000 /mnt/data/lcesarini/gripho_3km.nc  /mnt/data/lcesarini/tx.nc
# cdo setgrid,/home/lcesarini/2022_resilience/resilience/utils/grid_gripho.txt /mnt/data/lcesarini/tx12.nc /mnt/data/lcesarini/tx13.nc
# # cdo sellonlatbox,6.5,13.9,43.25,47.5 /mnt/data/lcesarini/gripho_3km.nc /mnt/data/lcesarini/gripho_clipped.nc
# ncks -C -O -x -v lon,lat /mnt/data/lcesarini/tx.nc /mnt/data/lcesarini/tx2.nc
# cdo setmisstoc,1 /mnt/data/lcesarini/tx2.nc /mnt/data/lcesarini/tx3.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tx13.nc /mnt/data/lcesarini/tx14.nc
# cdo remapcon,/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CMCC/CPM/pr/CMCC_ECMWF-ERAINT_200101010030-200112312330.nc /mnt/data/lcesarini/tx2.nc /mnt/data/lcesarini/tx3.nc

# # IDL-WRF381CA DOES NOT WORk
# cp  /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_208901010030-208912312330.nc /mnt/data/lcesarini/tmp/ttttttt_FJZ.nc
# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209801010030-209812312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
# cdo mulc,3600 tmp/ttt_${fileFIN} tmp/tt_${fileFIN}
# cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' tmp/tt_${fileFIN} tmp/t_${fileFIN}
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_208901010030-208912312330.nc /mnt/data/lcesarini/tmp/tttt_FJZ.nc


# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_IDL-WRF381BH_fpsconv-x1n2-v1_1hr_199901010030-199912312330.nc /mnt/data/lcesarini/tmp/ttssasat_FJZ.nc

# cdo setgrid,/mnt/data/gfosser/DATA_FPS/VarFixed/FZJ-IBG3-IDL/CPM/gridCPM_FZJ-IBG_mod /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/IDL-WRF381BH/CPM/pr/pr_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_IDL-WRF381BH_fpsconv-x1n2-v1_1hr_199901010030-199912312330.nc /mnt/data/lcesarini/tmp/pp.nc
# ncks -C -O -x -v lon,lat  ${pathTMP}/s1_${fileFIN} ${pathTMP}/s2_${fileFIN}
# cdo mulc,3600 ${pathTMP}/s2_${fileFIN} ${pathTMP}/s3_${fileFIN}
# cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' ${pathTMP}/s3_${fileFIN} ${pathTMP}/s4_${fileFIN}
# cdo remapycon,${GRID} ${pathTMP}/s4_${fileFIN} ${pathOUT}/${fileFIN}


# ncks -C -O -x -v lon,lat  /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IDL-WRF381DA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_200001010030-200012312330.nc /mnt/data/lcesarini/tmp/ttssasat_FJZ.nc

# #WORKS
# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Historical/FZJ-IBG3-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_historical_r12_FZJ-IBG3-WRF381CA_v1_1hr_199801010030-199812312330.nc /mnt/data/lcesarini/tmp/ttssasat_FJZ.nc
# cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/ttssasat_FJZ.nc /mnt/data/lcesarini/tmp/ttssasat_FJZzz.nc

# # # FZJ-IDL-WRF381DA WORKS
# # ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/FZJ-IDL-WRF381DA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_209301010030-209312312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
# # cdo mulc,3600 /mnt/data/lcesarini/tmp/ttt_FJZ.nc /mnt/data/lcesarini/tmp/tt_FJZ.nc
# # cdo -setattribute,pr@units='mm hr-1',pr@standard_name='precipitation' /mnt/data/lcesarini/tmp/tt_FJZ.nc /mnt/data/lcesarini/tmp/t_FJZ.nc
# # cdo remapycon,/home/lcesarini/2022_resilience/resilience/scripts/newcommongrid.txt /mnt/data/lcesarini/tmp/t_FJZ.nc /mnt/data/lcesarini/tmp//pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381DA_fpsconv-x1n2-v1_1hr_209301010030-209312312330.nc

# ncks -C -O -x -v lon,lat /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209101010030-209112312330.nc /mnt/data/lcesarini/tmp/ttt_FJZ.nc
# cp /mnt/data/gfosser/DATA_FPS/Rcp85/IDL-WRF381CA/CPM/pr/pr_ALP-3_SMHI-EC-EARTH_rcp85_r12i1p1_FZJ-IDL-WRF381CA_fpsconv-x1n2-v1_1hr_209101010030-209112312330.nc /mnt/data/lcesarini/tmp/sss.nc

# ncatted -O -a _FillValue,pr,d,, /mnt/data/lcesarini/tmp/sss.nc



# cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_405x261_latlon_bounds_MED11 /mnt/data/gfosser/DATA_FPS/ECMWF-ERAINT/CNRM/RCM/tas/tas_MED-11_ECMWF-ERAINT_evaluation_CNRM-ALADIN62_v1_1hr_200701010030-200712312330.nc /mnt/data/lcesarini/tmp/sss.nc
# cdo remapbil /mnt/data/lcesarini/tmp/ssss.nc /mnt/data/lcesarini/tmp/ssss2.nc

