#!/bin/bash


n_runs=3

basedir="final-runs2"

batch_size=64
num_blocks=4
latent_size=128
hidden_size=256
lr="3e-4"
run_name=$1
steps=20000
set_size1=10
set_size2=30
dataset="mnist"
weight_sharing="none"
pretrain=1000
dropout=0

warmup_steps=-1
decoder_layers=0
ln=1
lambda0=0.5
residual="base"
merge="concat"
anneal_ss=-1

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_omniglot.sh "${run_name}_csab/${i}" $dataset $pretrain $lr "csab" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    sbatch scripts/train_omniglot.sh "${run_name}_naive/${i}" $dataset $pretrain $lr "naive" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    sbatch scripts/train_omniglot.sh "${run_name}_pine/${i}" $dataset $pretrain $lr "pine" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    sbatch scripts/train_omniglot.sh "${run_name}_rn/${i}" $dataset $pretrain $lr "rn" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    sbatch scripts/train_omniglot.sh "${run_name}_cross-only/${i}" $dataset $pretrain $lr "cross-only" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    #sbatch scripts/train_omniglot.sh "${run_name}_sum-merge/${i}" $dataset $pretrain $lr "csab" $weight_sharing "sum" $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    #sbatch scripts/train_omniglot.sh "${run_name}_naive-rn/${i}" $dataset $pretrain $lr "naive-rn" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    #sbatch scripts/train_omniglot.sh "${run_name}_naive-rff/${i}" $dataset $pretrain $lr "naive-rff" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss

    sbatch scripts/train_omniglot.sh "${run_name}_union/${i}" $dataset $pretrain $lr "union-enc" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss
    sbatch scripts/train_omniglot.sh "${run_name}_union-enc/${i}" $dataset $pretrain $lr "union-enc" $weight_sharing $merge $warmup_steps $latent_size $hidden_size $num_blocks $batch_size $steps $set_size1 $set_size2 $decoder_layers $dropout $ln $anneal_ss

    #sbatch scripts/train_omniglot.sh "${run_name}_csab_none/${i}" $dataset $pretrain $lr "csab" "none"
    #sbatch scripts/train_omniglot.sh "${run_name}_csab_sym/${i}" $dataset $pretrain $lr "csab" "sym"
done
