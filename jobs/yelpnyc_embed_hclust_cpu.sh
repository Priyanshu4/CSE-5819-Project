#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=yelpnyc_8d_hclust

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection

cd ..

python -m src.main --name yelpnyc_8d_hclust --dataset yelpnyc --loss simi --optimizer adam --epochs 100 --dim 8 --clustering hclust
