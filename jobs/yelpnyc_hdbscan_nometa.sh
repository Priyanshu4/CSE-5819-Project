#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=80G
#SBATCH --job-name=yelpnyc_8d_hdbscan

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-group-detection

cd ..

python -m src.main --name yelpnyc_8d_hdbscan --dataset yelpnyc --embedding results/yelpnyc_8d.pkl  --clustering hdbscan --no_metadata
