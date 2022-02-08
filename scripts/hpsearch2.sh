#!/bin/bash

n_runs=5

run_name=$1

data="synth"
n=8
merge="concat"

latent_size=8
hidden_size=16
bs=64
lr="1e-3"
steps=30000
set_size1=10
set_size2=30
weight_sharing="none"
ln=1
decoder_layers=1
basedir="final-runs2/hptest"

nblocks=(1 2 4 6)
warmup=(-1 5000 10000)

for nb in "${nblocks[@]}"
do
    for wsteps in "${warmup[@]}"
    do
        for (( i = 0 ; i < $n_runs ; i++ ))
        do
            sbatch scripts/train_gan.sh "${run_name}_nb${nb}_w${wsteps}/$i" "csab" $merge $data $n $latent_size $hidden_size $bs $lr $steps $set_size1 $set_size2 $wsteps $weight_sharing $ln $nb $decoder_layers $basedir
        done
    done
done