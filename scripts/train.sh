#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

run_name=$1
target=$2
data=$3
num_inds=$4
equi=$5
is=$6
lts=$7
hs=$8
lr=$9
clip=${10}
model=${11}
basedir=${12}
nn=${13}
nb=${14}
ss1=${15}
ss2=${16}
merge=${17}


if [ $target == "w1" ]
then
    argstring="--normalize scale-linear --blur 0.001 --scaling 0.98"
elif [ $target == "w2" ]
then
    argstring="--normalize scale-linear"
elif [ $target == "w1_exact" ]
then
    argstring="--normalize scale-linear"
elif [ $target == "kl" ]
then
    argstring="--normalize whiten"
elif [ $target == "mi" ]
then
    argstring="--normalize none"
fi


if [ $equi -eq 1 ]
then
    argstring="${argstring} --equi"
fi
if [ $nn -eq 1 ]
then
    argstring="${argstring} --nn"
fi

python3 train.py $run_name --target $target --data $data --model $model --num_inds $num_inds --dim $is --latent_size $lts --hidden_size $hs --lr $lr --clip $clip --basedir $basedir --num_blocks $nb --set_size $ss1 $ss2 --merge $merge --checkpoint_name $SLURM_JOB_ID $argstring 