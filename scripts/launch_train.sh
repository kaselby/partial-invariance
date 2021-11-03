#!/bin/bash

n_runs=3

target='kl'
dim=2
latent_size=16
hidden_size=32

run_name=$1

for i in {1..$n_runs}
do
    sbatch scripts/train.sh "${run_name}/${i}" $target 'gmm' -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size ))
    sbatch scripts/train.sh "${run_name}_equi/${i}" $target 'gmm' -1 1 $dim $latent_size $hidden_size
    sbatch scripts/train.sh "${run_name}_pine/${i}" $target 'gmm' -1 1 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size*2 ))
done
