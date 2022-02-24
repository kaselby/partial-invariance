#!/bin/bash

n_runs=3

target='mi'
data='corr'
dim=2
latent_size=32
hidden_size=64
lr="1e-4"
clip=-1
basedir="final-runs2"
run_name=$1
num_blocks=4
ss1=100
ss2=150
merge="concat"
warmup_steps=-1
steps=100000
dl=1
residual='base'
scale_out='none'
vardim=1
batch_size=32

for (( i = 0 ; i < $n_runs ; i++ ))
do
    #sbatch scripts/train.sh "${run_name}_csab/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr $clip "csab" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    sbatch scripts/train.sh "${run_name}_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_naive/${i}" $target $data -1 0 $dim $latent_size $hidden_size $lr $clip "naive" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_rn/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size/2 )) $lr $clip "rn" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_pine/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size*2 )) $lr $clip "pine" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_sum-merge/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $num_blocks $ss1 $ss2 "sum" $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_cross-only/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr $clip "cross-only" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_naive-rn/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size/2 )) $lr $clip "naive-rn" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_naive-rff/${i}" $target $data -1 0 $dim $(( dim*latent_size/2 )) $(( dim*hidden_size/2 )) $lr $clip "naive-rff" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size

    #sbatch scripts/train.sh "${run_name}_naive_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "naive" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_rn/${i}" $target $data -1 0 $dim $(( dim*latent_size )) $(( dim*hidden_size )) $lr $clip "rn" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_rn_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "rn" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    #sbatch scripts/train.sh "${run_name}_cross-only_equi/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "cross-only" $basedir 0 $num_blocks $ss1 $ss2 $merge $warmup_steps $steps $dl $residual $scale_out $vardim $batch_size
    
done

#sizes=(20 50 100 200 500)
#for ss in "${sizes[@]}"
#do
#    for (( i = 0 ; i < $n_runs ; i++ ))
#    do
#        sbatch scripts/train.sh "${run_name}_ss${ss}/${i}" $target $data -1 1 $dim $latent_size $hidden_size $lr $clip "csab" $basedir 0 $num_blocks $ss $(($ss+$ss/4))
#    done
#done
