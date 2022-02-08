#!/bin/bash

n_runs=3

run_name=$1

data="md"
n=8
merge="concat"

latent_size=512
hidden_size=1024
bs=16
lr="1e-5"
steps=20000
set_size1=10
set_size2=30
warmup_steps=-1
weight_sharing="none"
ln=1
num_blocks=1
decoder_layers=1
basedir="final-runs2"

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_gan.sh "${run_name}_csab/$i" "csab" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $warmup_steps $weight_sharing $ln $num_blocks $decoder_layers $basedir
    sbatch scripts/train_gan.sh "${run_name}_pine/$i" "pine" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $warmup_steps $weight_sharing $ln $num_blocks $decoder_layers $basedir
    sbatch scripts/train_gan.sh "${run_name}_naive/$i" "naive" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $warmup_steps $weight_sharing $ln $num_blocks $decoder_layers $basedir
    sbatch scripts/train_gan.sh "${run_name}_rn/$i" "rn" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $warmup_steps $weight_sharing $ln $num_blocks $decoder_layers $basedir
    sbatch scripts/train_gan.sh "${run_name}_cross-only/$i" "cross-only" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $warmup_steps $weight_sharing $ln $num_blocks $decoder_layers $basedir
    #sbatch scripts/train_gan.sh "${run_name}_sum-merge/$i" "csab" "sum" $data $n
done
