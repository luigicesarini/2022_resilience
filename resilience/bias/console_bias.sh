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
cd /home/lcesarini/2022_resilience/resilience/bias/

time ./bias_corr.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy

./bc_metrics.py -ap VALIDATION -nq 1000 -rp STATIONS -s SEQUENTIAL 

./get_csv_from_nc.py -ap VALIDATION -m q -s SEQUENTIAL -nq 1000
