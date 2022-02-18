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

argstring=""
if [ ${17} -eq 1 ]
then
    argstring="${argstring} --ln"
fi

python3 train_omniglot.py $1 --checkpoint_name $SLURM_JOB_ID --dataset $2 --pretrain_steps $3 --lr $4 --model $5 --weight_sharing $6 --merge $7 --warmup_steps $8 --latent_size $9 --hidden_size ${10} --num_blocks ${11} --batch_size ${12} --steps ${13} --set_size ${14} ${15} --dropout ${16} $argstring