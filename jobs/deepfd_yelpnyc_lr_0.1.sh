#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=24
#SBATCH --constraint='skylake'
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      # Destination email address
#SBATCH --mem=40G

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate deepfd-env

cd ../deepfd

python -m src.main --cuda -1 --dataSet yelpnyc --cls_method none --lr 0.1
