#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=hclust_vis

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-group-detection

cd ..

python -m src.main --name mid_10k_hclust --dataset synthetic_mid_10k --loss simi --optimizer adam --epochs 100 --dim 16 --clustering hclust