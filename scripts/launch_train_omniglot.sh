#!/bin/bash

run_name=$1
dataset=$2
pretrain=$3
lr=$4

n_runs=3

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_omniglot.sh "${run_name}_csab_cross/${i}" $dataset $pretrain $lr "csab" "cross"
#    sbatch scripts/train_omniglot.sh "${run_name}_naive/${i}" $dataset $pretrain $lr "naive" "cross"
#    sbatch scripts/train_omniglot.sh "${run_name}_pine/${i}" $dataset $pretrain $lr "pine" "cross"

    sbatch scripts/train_omniglot.sh "${run_name}_csab_none/${i}" $dataset $pretrain $lr "csab" "none"
    sbatch scripts/train_omniglot.sh "${run_name}_csab_sym/${i}" $dataset $pretrain $lr "csab" "sym"
done
