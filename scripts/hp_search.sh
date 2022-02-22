#!/bin/bash

n_runs=3

target='kl'
data='gmm'
dim=2
latent_size=16
hidden_size=32
clip=-1
basedir="final-runs2/hptest"
run_name=$1
ss1=100
ss2=150
merge="concat"
steps=100000
residual="base"
scale_out="none"
warmup=-1

vardims=(0 1)
lrs=("1e-3" "1e-4")
nblocks=(4)
dls=(1)
latents=(16 32)

for lr in "${lrs[@]}"
do
    for nb in "${nblocks[@]}"
    do
        for vardim in "${vardims[@]}"
        do
            for dl in "${dls[@]}"
            do
                for ls in "${latents[@]}"
                do
                    for (( i = 0 ; i < $n_runs ; i++ ))
                    do
                        sbatch scripts/train.sh "${run_name}_kl_ls${ls}_lr${lr}_vardim${vardim}/${i}" $target $data -1 1 $dim $ls $(( 2*latent_size )) $lr $clip "csab" $basedir 0 $nb $ss1 $ss2 $merge $warmup $steps $dl $residual $scale_out $vardim
                    done
                done
            done
        done
    done
done


