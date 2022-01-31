#!/bin/bash

n_runs=3

run_name=$1

data="md"
merge="concat"

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_gan.sh "${run_name}_csab/$i" "csab" $merge
    sbatch scripts/train_gan.sh "${run_name}_pine/$i" "pine" $merge
    sbatch scripts/train_gan.sh "${run_name}_naive/$i" "naive" $merge
    sbatch scripts/train_gan.sh "${run_name}_rn/$i" "rn" $merge
    sbatch scripts/train_gan.sh "${run_name}_cross-only/$i" "cross-only" $merge
    sbatch scripts/train_gan.sh "${run_name}_sum-merge/$i" "csab" "sum"
done
