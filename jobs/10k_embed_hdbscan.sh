#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=hdbscan 

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-group-detection

cd ..

python -m src.main --name hdbscan_embed_test --dataset synthetic_easy_1000 --loss simi --optimizer adam --epochs 50 --dim 2 --clustering hdbscan
