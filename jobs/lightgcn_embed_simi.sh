#!/bin/bash

#SBATCH --partition=general-gpu
#SBATCH --constraint='v100'
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=16G
#SBATCH --job-name=lightgcn_simi_adam_100epochs_8d 

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection-env

cd ../lightgcn_embedder/src

python __main__.py --dataset yelpnyc --loss simi --fast_simi --optimizer adam --epochs 100 --dim 8 --name simi_adam_100epochs_8d 
