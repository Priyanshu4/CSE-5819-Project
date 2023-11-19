#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=12
#SBATCH --constraint='skylake'
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      # Destination email address
#SBATCH --mem=20G

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection-env

cd ../lightgcn_embedder/src

python __main__.py --dataset yelpnyc --epochs 500
