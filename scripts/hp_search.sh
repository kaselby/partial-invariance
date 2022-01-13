#!/bin/bash

n_runs=3

target='mi'
data='corr'
dim=2
latent_size=32
hidden_size=64
clip=-1
basedir="final-runs/hptest"
run_name=$1

lrs=["1e-3" "1e-4"]
nblocks=[1 2]

for lr in "${lrs[@]}"
do
    for nb in "${nblocks[@]}"
    do
        for (( i = 0 ; i < $n_runs ; i++ ))
        do
            sbatch scripts/train.sh "${run_name}_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir
        done
    done
done


