#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109


python3 test_rho.py --run_name $1 --set_size $2 --d $3