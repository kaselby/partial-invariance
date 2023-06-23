#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109

checkpoint_id=$SLURM_JOB_ID
checkpoint_dir="/checkpoint/$USER/$checkpoint_id"
basedir="2023-runs-fixed"

run_name=$1

model='multi-set-transformer'
dataset='gmm'
task='stat/DV'

bs=128
lr="1e-5"
ss1=200
ss2=300
ss_schedule=-1
eval_every=500
save_every=2000
train_steps=100000
val_steps=200
test_steps=500
use_amp=0

grad_clip=50

num_blocks=4
num_heads=4
ls=16
hs=32
dropout=0
decoder_layers=1
weight_sharing='none'

pretrain_steps=0
pretrain_lr="3e-4"

poisson=0
val_split=0.1

text_model='bert'
img_model='resnet'

episode_classes=100
episode_datasets=5
episode_length=500
p_dl=0.3
md_path="/ssd003/projects/meta-dataset"

n=2

normalize='whiten'
equi=1
vardim=1

split_inputs=1
decoder_self_attn=0
enc_blocks=4
dec_blocks=1
ln=1
max_rho=0.99

criterion=''

argstring="$run_name --basedir $basedir --checkpoint_dir $checkpoint_dir \
    --model $model --dataset $dataset --task $task --batch_size $bs --lr $lr --set_size $ss1 $ss2 \
    --eval_every $eval_every --save_every $save_every --train_steps $train_steps --val_steps $val_steps \
    --test_steps $test_steps --num_blocks $num_blocks --num_heads $num_heads --latent_size $ls \
    --hidden_size $hs --dropout $dropout --decoder_layers $decoder_layers --weight_sharing $weight_sharing \
    --pretrain_steps $pretrain_steps --pretrain_lr $pretrain_lr --val_split $val_split \
    --text_model $text_model --img_model $img_model --episode_classes $episode_classes \
    --episode_datasets $episode_datasets --episode_length $episode_length --p_dl $p_dl \
    --md_path $md_path --n $n --normalize $normalize --enc_blocks $enc_blocks --dec_blocks $dec_blocks \
    --max_rho $max_rho --clip $grad_clip"

if [ $equi -eq 1 ]
then
    argstring="$argstring --equi"
fi
if [ $vardim -eq 1 ]
then
    argstring="$argstring --vardim"
fi
if [ $poisson -eq 1 ]
then
    argstring="$argstring --poisson"
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
if [ -n $criterion ]
then
    argstring="$argstring --criterion $criterion"
fi

if [ $use_amp -eq 1 ]
then
    argstring="$argstring --use_amp"
fi

python3 main.py $argstring
