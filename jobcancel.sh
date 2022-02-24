#!/bin/bash

for (( i = $1 ; i < $2+1 ; i++ ))
do
    scancel $i
done