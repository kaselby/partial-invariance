#!/bin/bash

n_runs=3

batch_size=4
num_blocks=1
latent_size=512
hidden_size=1024
lr="1e-3"
basedir="final-runs"
run_name=$1
steps=20000
set_size1=3
set_size2=10
merge="concat"
dataset="coco"
warmup_steps=5000

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_coco.sh "${run_name}_csab/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "csab" $set_size1 $set_size2 $merge $dataset $warmup_steps
    sbatch scripts/train_coco.sh "${run_name}_naive/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "naive" $set_size1 $set_size2 $merge $dataset $warmup_steps
    sbatch scripts/train_coco.sh "${run_name}_pine/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "pine" $set_size1 $set_size2 $merge $dataset $warmup_steps
    sbatch scripts/train_coco.sh "${run_name}_rn/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "rn" $set_size1 $set_size2 $merge $dataset $warmup_steps
    sbatch scripts/train_coco.sh "${run_name}_cross-only/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "cross-only" $set_size1 $set_size2 $merge $dataset $warmup_steps
    sbatch scripts/train_coco.sh "${run_name}_sum-merge/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "csab" $set_size1 $set_size2 "sum" $dataset $warmup_steps
done
