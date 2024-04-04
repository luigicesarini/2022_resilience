#!/bin/bash
#SBATCH --job-name=plotpaper
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn06
#SBATCH --account=PR_climate

echo "start of job"
date

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env
srun python ./visualization/make_maps.py

date
echo "end of job"
