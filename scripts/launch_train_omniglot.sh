#!/bin/bash

run_name=$1
dataset=$2
pretrain=$3
lr=$4

weight_sharing="cross"
merge="concat"
warmup_steps=5000

n_runs=3

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_omniglot.sh "${run_name}_csab/${i}" $dataset $pretrain $lr "csab" $weight_sharing $merge $warmup_steps
    sbatch scripts/train_omniglot.sh "${run_name}_naive/${i}" $dataset $pretrain $lr "naive" $weight_sharing $merge $warmup_steps
    sbatch scripts/train_omniglot.sh "${run_name}_pine/${i}" $dataset $pretrain $lr "pine" $weight_sharing $merge $warmup_steps
    sbatch scripts/train_omniglot.sh "${run_name}_rn/${i}" $dataset $pretrain $lr "rn" $weight_sharing $merge $warmup_steps
    sbatch scripts/train_omniglot.sh "${run_name}_cross-only/${i}" $dataset $pretrain $lr "cross-only" $weight_sharing $merge $warmup_steps
    sbatch scripts/train_omniglot.sh "${run_name}_sum-merge/${i}" $dataset $pretrain $lr "csab" $weight_sharing "sum" $warmup_steps

    #sbatch scripts/train_omniglot.sh "${run_name}_csab_none/${i}" $dataset $pretrain $lr "csab" "none"
    #sbatch scripts/train_omniglot.sh "${run_name}_csab_sym/${i}" $dataset $pretrain $lr "csab" "sym"
done
