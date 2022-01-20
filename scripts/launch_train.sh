#!/bin/bash

n_runs=3

target='mi'
data='corr'
dim=2
latent_size=32
hidden_size=64
lr="1e-4"
clip=-1
basedir="final-runs"
run_name=$1
num_blocks=2
ss1=100
ss2=150

#for (( i = 0 ; i < $n_runs ; i++ ))
#do
    #sbatch scripts/train.sh "${run_name}_csab/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr $clip "csab" $basedir 0 $num_blocks $ss1 $ss2
#    sbatch scripts/train.sh "${run_name}_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $num_blocks $ss1 $ss2
    #sbatch scripts/train.sh "${run_name}_pine/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size*2 )) $lr $clip "pine" $basedir 0 $num_blocks $ss1 $ss2
#done

sizes=(20 50 100 200 500)
for ss in "${sizes[@]}"
do
    for (( i = 0 ; i < $n_runs ; i++ ))
    do
        sbatch scripts/train.sh "${run_name}_ss${ss}/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $num_blocks $ss $(($ss+$ss/4))
    done
done
