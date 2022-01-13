#!/bin/bash

n_runs=3

target='mi'
data='corr'
dim=2
latent_size=32
hidden_size=64
lr="1e-3"
clip=-1
basedir="final-runs"
run_name=$1

for (( i = 0 ; i < $n_runs ; i++ ))
do
    #sbatch scripts/train.sh "${run_name}_csab/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr $clip "csab" $basedir
    sbatch scripts/train.sh "${run_name}_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir
    #sbatch scripts/train.sh "${run_name}_pine/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size*2 )) $lr $clip "pine" $basedir
done
