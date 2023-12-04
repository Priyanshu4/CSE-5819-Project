#!/bin/bash

#SBATCH --partition=general-gpu
#SBATCH --constraint='v100'
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=lgcn_adam_100ep_16d 

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-group-detection

cd ..

python -m src.main --name adam_100ep_8d --dataset yelpnyc --loss simi --optimizer adam --epochs 100 --dim 16 --clustering dbscan --a_fold 10
