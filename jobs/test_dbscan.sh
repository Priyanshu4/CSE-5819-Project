#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=24
#SBATCH --constraint='skylake'
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal@uconn.edu      # Destination email address
#SBATCH --mem=40G

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate deepfd-env

cd ../testing

python test_dbscan.py --embeddings ../data/yelpnyc/embedded/deepfd/embs_ep10.pkl --labels ../data/yelpnyc/yelpnyc_labels.pkl
