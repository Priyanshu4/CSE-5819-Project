#!/bin/bash

#SBATCH --partition=general
#SBATCH -N 1
#SBATCH --ntasks=24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niteesh.saravanan@uconn.edu      # Destination email address
#SBATCH --mem=20G

module purge

source /home/nns20006/miniconda3/etc/profile.d/conda.sh

conda activate lightgcn

cd ../hclust

python hclustmp.py --data=largeblobs
