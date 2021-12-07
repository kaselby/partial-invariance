#!/bin/bash

run_name=$1
dataset=$2
pretrain=$3

n_runs=3

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_omniglot.sh "${run_name}_csab/${i}" $dataset $pretrain "csab"
    sbatch scripts/train_omniglot.sh "${run_name}_naive/${i}" $dataset $pretrain "naive"
    sbatch scripts/train_omniglot.sh "${run_name}_pine/${i}" $dataset $pretrain "pine"
done
