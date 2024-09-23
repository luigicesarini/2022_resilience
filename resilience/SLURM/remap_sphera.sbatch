#!/bin/bash
#SBATCH --job-name=REMAP_SPHERA
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --cpus-per-task=25
#SBATCH --nodelist=cn04
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --account=PR_climate

echo "start of job"
date +'%Y-%m-%d %H:%M:%S'
echo "eliminating results slurm older than 10 days"

DATE=$(date -d "100 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/2022_resilience/resilience/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env


# 
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/conversion_and_remapping_sphera.py --year 1995  --long_name tp --short_name pr &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/conversion_and_remapping_sphera.py --year 1996  --long_name tp --short_name pr &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/conversion_and_remapping_sphera.py --year 1997  --long_name tp --short_name pr &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/conversion_and_remapping_sphera.py --year 1998  --long_name tp --short_name pr &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/conversion_and_remapping_sphera.py --year 1999  --long_name tp --short_name pr &


wait

date
echo "end of job"


