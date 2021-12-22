#!/bin/bash

n_runs=3

batch_size=8
num_blocks=2 
latent_size=512
hidden_size=2048
lr="1e-5"
basedir="final-runs"
run_name=$1

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train.sh "${run_name}/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks "csab"
    sbatch scripts/train.sh "${run_name}_naive/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks "naive"
    sbatch scripts/train.sh "${run_name}_pine/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks "pine" 
done
