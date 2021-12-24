#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109


python3 train_gan.py $1 --steps 1000 --set_size 6 10 --checkpoint_name $SLURM_JOB_ID