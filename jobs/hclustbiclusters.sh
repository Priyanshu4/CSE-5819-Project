#!/bin/bash

#SBATCH --partition=general-gpu

#SBATCH -N 1

#SBATCH -n 24

module purge

source /home/nns20006/miniconda3/etc/profile.d/conda.sh

conda activate lightgcn

cd ../hclust

python hclusttest.py biclusters