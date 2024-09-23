#!/bin/bash
#SBATCH --job-name=bcHist
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=cn08
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --account=PR_climate

echo "start of job"
date +'%Y-%m-%d %H:%M:%S'
echo "eliminating results slurm older than 10 days"

DATE=$(date -d "10 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/2022_resilience/resilience/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env


# # KIT Done!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m KIT &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m KIT &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m KIT &

# # KNMI DOne!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m KNMI &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m KNMI &

# # HCLIMcom Done!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m HCLIMcom &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m HCLIMcom &

# # CNRM DOne!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m CNRM &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m CNRM &

# # CMCC: Done!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m CMCC &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m CMCC &

# # ICTP: DONE!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m ICTP &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m ICTP &

# # MOHC
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m MOHC &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m MOHC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m MOHC &

# # ETH: Done!
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p VALIDATION -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m ETH &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas MAM -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas JJA -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas SON -m ETH &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_hist.py -p TRAIN -ref SPHERA -mt Historical -a northern_italy --split SEQUENTIAL -seas DJF -m ETH &



wait

date
echo "end of job"


