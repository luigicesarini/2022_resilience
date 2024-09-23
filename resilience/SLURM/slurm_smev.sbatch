#!/bin/bash
#SBATCH --job-name=smev
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 28
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn04
#SBATCH --account=PR_climate

echo "start of job"
date +'%Y-%m-%d %H:%M:%S'
echo "eliminating results slurm older than 10 days"

DATE=$(date -d "1 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/2022_resilience/resilience/resultsSLURM/* -type f ! -newermt $DATE -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env

python smev/smev_luigi.py -y 1992 &
python smev/smev_luigi.py -y 1993 &
python smev/smev_luigi.py -y 1994 &
python smev/smev_luigi.py -y 1995 &
python smev/smev_luigi.py -y 1996 &
python smev/smev_luigi.py -y 1997 &
python smev/smev_luigi.py -y 1998 &
python smev/smev_luigi.py -y 1999 &
python smev/smev_luigi.py -y 2000 &
python smev/smev_luigi.py -y 2001 &
python smev/smev_luigi.py -y 2002 &
python smev/smev_luigi.py -y 2003 &
python smev/smev_luigi.py -y 2004 &
python smev/smev_luigi.py -y 2005 &
python smev/smev_luigi.py -y 2006 &
python smev/smev_luigi.py -y 2007 &
python smev/smev_luigi.py -y 2008 &
python smev/smev_luigi.py -y 2009 &
python smev/smev_luigi.py -y 2010 &
# python smev/smev_luigi.py -y 2011 &
python smev/smev_luigi.py -y 2012 &
python smev/smev_luigi.py -y 2013 &
python smev/smev_luigi.py -y 2014 &
python smev/smev_luigi.py -y 2015 &
python smev/smev_luigi.py -y 2016 &
python smev/smev_luigi.py -y 2017 &
python smev/smev_luigi.py -y 2018 &
python smev/smev_luigi.py -y 2019 &
python smev/smev_luigi.py -y 2020

date
echo "end of job"


