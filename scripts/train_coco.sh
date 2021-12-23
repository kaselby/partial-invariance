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


python3 train_coco.py $1 --batch_size $2 --latent_size $3 --hidden_size $4 --lr $5 --num_blocks $6 --steps $7 --model $8 --set_size 6 10 --checkpoint_name $SLURM_JOB_ID