#!/bin/bash
#SBATCH --job-name=biascorr2
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 16
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=cn02
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
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


# Problem
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m KIT &

# Problem
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m KNMI &

# Done
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m ETH &

# to do
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m MOHC &

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m MOHC &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m MOHC &
#Problem calendar
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m ICTP &

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m ICTP &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m ICTP &
#Done
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m CMCC 

#Problem monotonically increasing or decreasing,
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m CNRM 

#Problem monotonically increasing or decreasing,
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas MAM -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas JJA -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas SON -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_future.py -p 2000_2010 -ref SPHERA -mt Historical -a northern_italy --slice near -seas DJF -m HCLIMcom 

wait

date
echo "end of job"


