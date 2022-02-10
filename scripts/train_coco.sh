#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --partition=t4v1,t4v2,p100,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109


argstring=""
if [ ${15} -eq 1 ]
then
    argstring="${argstring} --ln"
fi
if [ ${16} -eq 1 ]
then
    argstring="${argstring} --anneal_set_size"
fi
if [ -n "${19}" ]
then
    argstring="${argstring} --init_from ${19}"
fi


python3 train_coco.py $1 --batch_size $2 --latent_size $3 --hidden_size $4 --lr $5 --num_blocks $6 --steps $7 --model $8 --set_size $9 ${10} --merge ${11} --dataset ${12} --warmup_steps ${13} --decoder_layers ${14} --lambda0 ${17} --residual ${18} --checkpoint_name $SLURM_JOB_ID $argstring