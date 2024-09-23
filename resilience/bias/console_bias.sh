#!/usr/bin/bash

# # Set the pattern to replace
# old_pattern="909"
# new_pattern="99"

# # Iterate over files and rename them
# for file in /mnt/data/lcesarini/EVENTS/pr/*909*; 
# do
#     # Use parameter expansion to replace the old pattern with the new one
#     new_name="${file/$old_pattern/$new_pattern}"

#     # Rename the file
#     mv "$file" "$new_name"
# done
# cd /home/lcesarini/2022_resilience/resilience/bias/

# time ./bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m HCLIMcom
# time ./bias_corr_hist.py -p VALIDATION -mt Historical -ref SPHERA -a northern_italy -seas JJA --split SEQUENTIAL
# ./bc_metrics.py -ap VALIDATION -nq 1000 -rp STATIONS -s SEQUENTIAL 

# ./get_csv_from_nc.py -ap VALIDATION -m q -s SEQUENTIAL -nq 1000

MODEL=('KIT' 'KNMI' 'CNRM' 'ICTP' 'CMCC' 'HCLIMcom' 'MOHC' 'ETH')
for model in "${MODEL[@]}"; do
    mv /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/EQM/$model/pr/*Historical*.nc /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/EQM/$model/pr/Historical/
    mv /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/EQM/$model/pr/*ECMWF-ERAINT*.nc /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/EQM/$model/pr/ECMWF-ERAINT/
    mv /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/QDM/$model/pr/*Historical*.nc /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/QDM/$model/pr/Historical/
    mv /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/QDM/$model/pr/*ECMWF-ERAINT*.nc /mnt/beegfs/lcesarini/BIAS_CORRECTED/Rcp85/QDM/$model/pr/ECMWF-ERAINT/
done
