#!/bin/bash

#SBATCH --partition=general-gpu

#SBATCH -N 1

#SBATCH -n 24

module purge

module load slurm python/3.6.7 

source /home/nns20006/miniconda3/etc/profile.d/conda.sh

conda activate lightgcn

cd lightgcn/code

python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64