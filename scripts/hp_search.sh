#!/bin/bash

n_runs=3

target='mi'
data='corr'
dim=2
latent_size=16
hidden_size=32
clip=-1
basedir="final-runs2/hptest"
run_name=$1
ss1=100
ss2=150
merge="concat"
steps=80000

warmup=(-1 5000 20000)
lrs=("1e-2" "1e-3" "1e-4")
nblocks=(1 2 3)

for lr in "${lrs[@]}"
do
    for nb in "${nblocks[@]}"
    do
        for wsteps in "${warmup[@]}"
        do
            for (( i = 0 ; i < $n_runs ; i++ ))
            do
                sbatch scripts/train.sh "${run_name}_${lr}_${nb}/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $nb $ss1 $ss2 $merge $warmup_steps $steps
            done
        done
    done
done


