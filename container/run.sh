#!/usr/bin/env bash

dataset_path=$1

echo assuming dataset in $dataset_path

docker run --runtime nvidia -it -v $dataset_path:/root/dataset antispoof:latest