#!/bin/bash
#SBATCH --job-name=biascorr
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 12
#SBATCH --cpus-per-task=10
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn06
#SBATCH --account=PR_climate

echo "start of job"
date +'%Y-%m-%d %H:%M:%S'
echo "eliminating results slurm older than 10 days"

DATE=$(date -d "1 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/2022_resilience/resilience/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m KIT 

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m KIT &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m CNRM &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m HCLIMcom &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas MAM -m MOHC &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m CNRM &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m HCLIMcom &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas JJA -m MOHC &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m CNRM &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m HCLIMcom &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas SON -m MOHC &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m KIT 
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m KNMI 
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m CNRM &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m HCLIMcom &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -a northern_italy --slice far -seas DJF -m MOHC 

date
echo "end of job"


