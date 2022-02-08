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

nblocks=(1 2 4 6)
warmup=(-1 5000 10000)

for nb in "${nblocks[@]}"
do
    for wsteps in "${warmup[@]}"
    do
        for (( i = 0 ; i < $n_runs ; i++ ))
        do
            sbatch scripts/train.sh "${run_name}_${nb}_w${wsteps}/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $nb $ss1 $ss2 $merge $wsteps $steps $dl
        done
    done
done