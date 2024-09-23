#!/bin/bash
#SBATCH --job-name=decumulate_prec
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 60
#SBATCH --cpus-per-task=2
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

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 01 -ev pr 
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1995 -m 12 -ev pr 

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1996 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1997 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1998 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera2.py  -y 1999 -m 12 -ev pr 

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2005 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2006 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2007 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 01 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2008 -m 12 -ev pr &

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 01 -ev pr & 
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 02 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 03 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 04 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 05 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 06 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 07 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 08 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 09 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 10 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 11 -ev pr &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./scripts/new_sphera.py  -y 2009 -m 12 -ev pr 

date
echo "end of job"


