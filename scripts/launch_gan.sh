#!/bin/bash

n_runs=3

run_name=$1

data="md"
n=8
merge="concat"

ls=512
hs=1024

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_gan.sh "${run_name}_csab/$i" "csab" $merge $data $n $ls $hs
    sbatch scripts/train_gan.sh "${run_name}_pine/$i" "pine" $merge $data $n $ls $hs
    sbatch scripts/train_gan.sh "${run_name}_naive/$i" "naive" $merge $data $n $ls $hs
    sbatch scripts/train_gan.sh "${run_name}_rn/$i" "rn" $merge $data $n $ls $hs
    sbatch scripts/train_gan.sh "${run_name}_cross-only/$i" "cross-only" $merge $data $n $ls $hs
    #sbatch scripts/train_gan.sh "${run_name}_sum-merge/$i" "csab" "sum" $data $n
done
