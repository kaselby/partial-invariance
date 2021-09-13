#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=t4v1,t4v2,p100,rtx6000
#SBATCH -c 1
#SBATCH --mem=25GB

python3 train.py $1