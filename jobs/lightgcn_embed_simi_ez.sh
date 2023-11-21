#!/bin/bash

#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      
#SBATCH --mem=20G

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection-env

cd ../lightgcn_embedder/src

python __main__.py --loss simi --optimizer sgd --dataset synthetic_easy_1000 --epochs 20 --dim 2 --name ez_simi_sgd_20epochs_2d 
