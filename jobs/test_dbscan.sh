#!/bin/bash

#SBATCH --partition=general
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshu.agrawal+hpc@uconn.edu      # Destination email address
#SBATCH --mem=80G

module purge

source /home/pra20003/miniconda3/etc/profile.d/conda.sh

conda activate fake-review-detection

cd ../testing

python test_dbscan.py --embeddings ../data/yelpnyc/embedded/deepfd/embs_ep10.pkl --labels ../data/yelpnyc/yelpnyc_labels.pkl --algorithm DBSCAN
