#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

python3 train_hypnet.py $1 --checkpoint_name $SLURM_JOB_ID --lr 1e-3 --steps 750000 --hidden_size 256 --pca_dim 16 --subsample up