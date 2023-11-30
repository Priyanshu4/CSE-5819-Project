#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=hclust

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection

cd ..

python -m src.main --name hclust_test1 --dataset synthetic_easy_1000 --loss simi --optimizer adam --epochs 1 --dim 4 --clustering hclust
