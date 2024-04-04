#!/bin/bash
#SBATCH --job-name=biascorr
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn04
#SBATCH --account=PR_climate

echo "start of job"
date +'%Y-%m-%d %H:%M:%S'
echo "eliminating results slurm older than 10 days"

DATE=$(date -d "1 days ago" +'%Y-%m-%d %H:%M:%S')

# find /mnt/beegfs/lcesarini/2022_resilience/resilience/resultsSLURM/* -type f ! -newermt $DATE -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env


# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm ETH  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm KNMI  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm KIT  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm CNRM  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm CMCC  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm ICTP  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm HCLIMcom  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm MOHC  -p -c &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm SPHERA  -p -c 

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./visualization/make_diurnal.py -s JJA SON -m q -ev pr -q 0.5 0.6 0.7 0.8 0.9 -nm MOHC  -p -c 
date
echo "end of job"


