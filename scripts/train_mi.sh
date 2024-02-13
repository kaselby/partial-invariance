#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB


basedir="runs"

run_name=$1

checkpoint_dir="/checkpoint/$USER/$run_name"



task='stat/DV-MI'

###     data parameters
dataset='corr'
n=8
max_rho=0.9

###     general training parameters
bs=32
lr="1e-5"
eval_every=500
save_every=2000
train_steps=100000
val_steps=200
test_steps=500

use_amp=0

## set size parameters
ss1=250
ss2=350
ss_schedule=-1

## weight decay is important for the unsupervised version i think, but not for the supervised
weight_decay=0.01
grad_clip=-1


###     general model parameters
ls=16
hs=32
num_heads=4
weight_sharing='none'
dropout=0
ln=0
decoder_layers=1
normalize='none'

## these parameters control the dimension equivariance. they should usually be set to the same value - 1 to use it or 0 to not use it
equi=1
vardim=1


###     supervised parameters
model='multi-set-transformer'   
num_blocks=4

###     unsupervised parameters
dv_model='encdec'
enc_blocks=4
dec_blocks=1
eps="1e-6"

## these flags control some finicky/complicated aspects of the unsupervised model. i believe these are the right values for them
sample_marg=1
estimate_size=-1
split_inputs=1
decoder_self_attn=0


#i forget what these do, i dont think they matter. i wouldnt change them though
scale='none'
criterion=''



argstring="$run_name --basedir $basedir --checkpoint_dir $checkpoint_dir \
    --model $model --dataset $dataset --task $task --batch_size $bs --lr $lr --set_size $ss1 $ss2 \
    --eval_every $eval_every --save_every $save_every --train_steps $train_steps --val_steps $val_steps \
    --test_steps $test_steps --num_blocks $num_blocks --num_heads $num_heads --latent_size $ls \
    --hidden_size $hs --dropout $dropout --decoder_layers $decoder_layers --weight_sharing $weight_sharing \
    --n $n --normalize $normalize --enc_blocks $enc_blocks --dec_blocks $dec_blocks \
    --max_rho $max_rho --clip $grad_clip --weight_decay $weight_decay --estimate_size $estimate_size \
    --dv_model $dv_model --scale $scale --eps $eps"

if [ $equi -eq 1 ]
then
    argstring="$argstring --equi"
fi
if [ $vardim -eq 1 ]
then
    argstring="$argstring --vardim"
fi
if [ $split_inputs -eq 1 ]
then
    argstring="$argstring --split_inputs"
fi
if [ $decoder_self_attn -eq 1 ]
then
    argstring="$argstring --decoder_self_attn"
fi
if [ $ln -eq 1 ]
then
    argstring="$argstring --layer_norm"
fi
if [ ! -z $criterion ]
then
    argstring="$argstring --criterion $criterion"
fi
if [ $sample_marg -eq 1 ]
then
    argstring="$argstring --sample_marg"
fi


if [ $use_amp -eq 1 ]
then
    argstring="$argstring --use_amp"
fi

python3 main.py $argstring
