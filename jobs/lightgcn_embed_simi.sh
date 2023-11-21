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

python __main__.py --dataset yelpnyc --loss simi --optimizer sgd --epochs 50 --dim 2 --name simi_sgd_50epochs_2d 
