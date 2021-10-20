#!/bin/bash

DATASETS=("HypNet_test", "LenciBenotto", "EVALution", "BLESS", "BIBLESS", "Kotlerman2010", "Levy2014", "Turney2014", "Weeds")

for dataset in "${DATASETS[@]}"
do
    sbatch scripts/eval_hypnet.sh $1 $dataset
done