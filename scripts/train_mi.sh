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

## note on some conventions:
# parameters set to -1 mostly mean 'dont use this'
# parameters with values of 1 or 0 are mostly boolean flags (except for a couple like decoder_layers)

# stat/MI for supervised, stat/DV-MI for unsupervised
task='stat/DV-MI'

###     data parameters
# 'corr' for synthetic gaussian data, 'adult' for adult dataset
dataset='corr'
# dimensionality of the data
n=8
# correlations go from -max_rho to +max_rho for 'corr' dataset
max_rho=0.9

###     general training parameters
bs=32
# lr 1e-4 for supervised or 1e-5 for unsupervised
lr="1e-5"
# total number of training steps as well as how often to evaluate/save and how many steps to perform for evaluation
train_steps=100000
eval_every=500
save_every=2000
val_steps=200
test_steps=500

# whether or not to use amp acceleration. i forget if this actually sped things up at all, it may not have.
use_amp=0

## set size parameters
# set sizes go from ss1 to ss2
ss1=250
ss2=350
# whether to use a training schedule that starts with smaller set sizes and expands the set sizes during training.
# i dont believe this was needed for the MI stuff, but im going to leave it in just in case. I think the set size values used for the
# schedule are hardcoded somewhere.
ss_schedule=-1


# weight decay might have been used for unsupervised at one point, but i believe i replaced it with scale='logcov'
weight_decay=0
# i dont think gradient clipping was important, but not 100% sure
grad_clip=-1


###     general model parameters
# note that for the equivariant model, the actual latent size/hidden size is this value * the dimension of the data (n)
# thus if you want to turn off equivariance you should probably increase these iirc
ls=16
hs=32
num_heads=4
decoder_layers=1
# not sure exactly what the best values for these flags are or if they matter. i dont think they have a significant effect
dropout=0
ln=0
# 'none' or 'whiten'. i think for MI it doesnt matter.
normalize='none'

# i think this may actually be hardcoded based on the task/model right now so i dont remember if this flag matters.
# it might be used for certain model settings? the supervised model might use it.
# it refers to whether weight sharing is used in the MSAB blocks - 'none', 'cross' for XY=YX, and 'sym' for XY=YX and XX=YY
# if the order of the model inputs should be irrelevant and the two sets are wholly symmetric than using 'sym' can be useful.
# this might be the case for MI but it depends whether its formulated via the KL divergence or not because the KL divergence
# isnt symmetric.
weight_sharing='none'

# these parameters control the dimension equivariance. 
# they should usually be set to the same value - 1 to use it or 0 to not use it
equi=1
vardim=1

###     supervised parameters
model='multi-set-transformer'   
num_blocks=4

###     unsupervised parameters
dv_model='encdec'
enc_blocks=4
dec_blocks=1
# this is important for a finicky aspect of the unsupervised model. i believe 1e-5 or 1e-6 was a fine value.
eps="1e-6"

## these flags control some finicky/complicated aspects of the unsupervised model. i believe these are the right values for them
scale='logcov'
# this should be 1 if youre sampling directly from the marginals like for the gaussians and 0 for empirical samples like for the adult dataset
sample_marg=1
# i dont think this ended up doing what i wanted so its disabled
estimate_size=-1
split_inputs=1
decoder_self_attn=0


#i think this is maybe for deciding whether the sueprvised model uses mse or l1 error but i cant remember if its being used or not
#it may just be hardcoded
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
