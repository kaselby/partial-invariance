#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=t4v1,t4v2,p100,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

argstring=""
if [ ${15} -eq 1 ]
then
    argstring="${argstring} --ln"
fi

python3 train_gan.py $1 --img_encoder cnn --p_dl 0 --checkpoint_name $SLURM_JOB_ID --model $2 --merge $3 --data $4 --n $5 --latent_size $6 --hidden_size $7 --batch_size $8 --lr $9 --steps ${10} --set_size ${11} ${12} --warmup_steps ${13} --weight_sharing ${14} --num_blocks ${16} --decoder_layers ${17} $argstring