#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=yelpnyc_embed_1000ep

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-group-detection

cd ..

python -m src.main --name yelpnyc_embed_1000ep_8d --dataset yelpnyc --loss simi --optimizer adam --epochs 1000 --dim 8 --clustering none --a_fold 10
