#!/usr/bin/bash
# cd /home/lcesarini/2022_resilience/
period=("ECMWF-ERAINT" "historical" "Rcp85")
sim=( 'CNRM' 'MOHC' 'ICTP' 'KNMI' ) #'CMCC' 'JLU' 'FZJ' 'CNRM' 'ETH' 'HCLIMcom' 'MOHC''ICTP' 'KNMI' 'KIT' 'MOHC'
variable=('pr' 'uas' 'vas')

for file in ${PATH_FILE}/*.nc
do 
fileFIN="ETH_ECMWF-ERAINT${file##*1hr}"
echo Working on ${fileFIN}
# ncks -C -O -x -v lon,lat ${file} ${pathTMP}/t_${fileFIN}
# cdo remapcon,newcommongrid.txt ${pathTMP}/t_${fileFIN} ${PATH_OUTPUT}/${fileFIN}
cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_602x572_latlon_bounds_ICTP.txt /mnt/data/gfosser/DATA_FPS/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc /mnt/data/lcesarini/tmp/ttt_oro_ictp.nc
cdo remapycon,newcommongrid.txt /mnt/data/lcesarini/tmp/ttt_oro_ictp.nc /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/ICTP/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_ICTP-RegCM4-7_fpsconv-x2yn2-v1_fx.nc
done


pathTMP="/mnt/data/lcesarini/tmp/"
pathIN="/mnt/data/gfosser/DATA_FPS/"
pathOUT="/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/"
file="orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-v1_fx.nc"
fileFIN="orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-v1_fx.nc"
echo Working with CMCC
echo File name: ${fileFIN}
cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/CMCC/CPM/pr/pr_ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CMCC-CCLM5-0-9_x2yn2v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
ncks -C -O -x -v lon,lat ${pathIN}/${file} ${pathTMP}/ttttt_${fileFIN}
# cdo mulc,3600 ${pathTMP}/ttttt_${fileFIN} ${pathTMP}/tttt_${fileFIN}
cdo setgrid,${pathTMP}/tx.nc ${pathTMP}/ttttt_${fileFIN} ${pathTMP}/ttt_${fileFIN}
ncks -C -O -x -v lon,lat ${pathTMP}/ttt_${fileFIN} ${pathTMP}/tt_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/tt_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_CLMcom-CMCC-CCLM5-0-9_fpsconv-x2yn2-v1_fx.nc


echo Working with HCLIMcom
echo File name: ${fileFIN}

file="orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_fx.nc"
fileFIN="orog_ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1_fx.nc"

cdo setgrid,/mnt/data/gfosser/DATA_FPS/grid_573x485_latlon_bounds_ALP-3 ${pathIN}/${file} ${pathTMP}/ttt_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/ttt_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/HCLIMcom/${fileFIN}

echo Working with KIT
echo File name: ${fileFIN}

file="orog_ALP-3_MPI-M-MPI-ESM-LR_historical_r0i0p0_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_fx.nc"
fileFIN="orog_ALP-3_MPI-M-MPI-ESM-LR_historical_r0i0p0_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_fx.nc"

cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/KIT/CPM/pr/pr_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
cdo setgrid,${pathTMP}/tx.nc ${pathIN}/${file} ${pathTMP}/sss_${fileFIN}
ncks -C -O -x -v lon,lat ${pathTMP}/sss_${fileFIN} ${pathTMP}/tt_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/tt_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KIT/${fileFIN}

file="sftlf_ALP-3_MPI-M-MPI-ESM-LR_historical_r0i0p0_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_fx.nc"
fileFIN="sftlf_ALP-3_MPI-M-MPI-ESM-LR_historical_r0i0p0_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_fx.nc"

cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/KIT/CPM/pr/pr_ALP-3_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-KIT-CCLM5-0-15_fpsconv-x2yn2-v1_1hr_200001010030-200012312330.nc ${pathTMP}/tx.nc
cdo setgrid,${pathTMP}/tx.nc ${pathIN}/${file} ${pathTMP}/sss_${fileFIN}
ncks -C -O -x -v lon,lat ${pathTMP}/sss_${fileFIN} ${pathTMP}/tt_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/tt_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KIT/${fileFIN}






echo Working with KNMI
echo File name: ${fileFIN}

file="orog_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc"
fileFIN="orog_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-HCLIM38h1-AROME_fpsconv-x2yn2-v1_fx.nc"

ncks -C -O -x -v date,hms ${pathIN}/${file} ${pathTMP}/ttttt_${fileFIN}
cdo setgrid,/mnt/data/gfosser/DATA_FPS/gridCPM_KNMI_mod ${pathTMP}/ttttt_${fileFIN} ${pathTMP}/tttt_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/tttt_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/KNMI/${fileFIN}



echo Working with MOHC
echo File name: ${fileFIN}

file="orog_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_fx.nc"
fileFIN="orog_ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_fx.nc"

cdo seltimestep,1 /mnt/data/gfosser/DATA_FPS/Historical/MOHC/CPM/pr/pr_ALP-3_HadGEM3-GC3.1-N512_historical_r1i1p1_HadREM3-RA-UM10.1_fpsconv-x0n1-v1_1hr_200708010030-200708302330.nc ${pathTMP}/tx.nc
cdo setgrid,${pathTMP}/tx.nc ${pathIN}/${file} ${pathTMP}/sss_${fileFIN}
cdo remapycon,newcommongrid.txt ${pathTMP}/sss_${fileFIN} /mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/MOHC/${fileFIN}
