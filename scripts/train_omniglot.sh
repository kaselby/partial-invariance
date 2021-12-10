#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

python3 train_omniglot.py $1 --checkpoint_name $SLURM_JOB_ID --dataset $2 --pretrain_steps $3 --poisson --model $4 --weight_sharing cross --lr $5