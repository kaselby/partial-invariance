#!/bin/bash

n_runs=3

batch_size=6
num_blocks=1
latent_size=512
hidden_size=1024
lr="1e-5"
basedir="final-runs"
run_name=$1
steps=20000
set_size1=3
set_size2=10
merge="concat"
dataset="coco"

models=("csab" "pine" "rn" "cross-only" "naive")

for (( i = 0 ; i < $n_runs ; i++ ))
do
    for model in "${models[@]}"
    do
        if  [ ! -f "${basedir}/${dataset}/${run_name}_${model}/${i}/model.pt" ]
        then
            sbatch scripts/train_coco.sh "${run_name}_${model}/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps $model $set_size1 $set_size2 $merge $dataset
        fi
    done
done