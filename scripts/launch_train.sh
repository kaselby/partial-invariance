#!/bin/bash

n_runs=3

target='kl'
data='gmm'
dim=2
latent_size=16
hidden_size=32
lr="1e-3"

run_name=$1

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train.sh "${run_name}/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr "csab"
    sbatch scripts/train.sh "${run_name}_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr "csab" 
    sbatch scripts/train.sh "${run_name}_pine/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size*2 )) $lr "pine"
done
