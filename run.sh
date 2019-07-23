#!/usr/bin/env bash

dataset_path=$1

if [ -z "$var" ]
then
      echo "dataset variable not provided, assuming /root/dataset"
      dataset_path=/root/dataset
else
      echo provided dataset location: $dataset_path
fi

python3 train_CyclicLR.py --model=model_A --image_mode=color --image_size=48 --dataset_path=/root/dataset