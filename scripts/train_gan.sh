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



python3 train_gan.py $1 --batch_size 16 --steps 16000 --set_size 10 30 --img_encoder cnn --p_dl 0 --checkpoint_name $SLURM_JOB_ID --model $2 --merge $3