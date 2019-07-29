#!/usr/bin/env bash

combine_dir=$1
#nuaa_dir=$2
#boks_dir=$3



echo nua dir located in $nuaa_dir


# suppose using Detectedface format of NUAA
# TODO: add url to download NUAA detected face data

cd $nuaa_dir
echo working in `pwd` dir

awk '{printf "ClientFace/%s ClientFace/%s ClientFace/%s 1\n",$1,$1,$1}' client_train_face.txt > train_list.txt
awk '{printf "ImposterFace/%s ImposterFace/%s ImposterFace/%s 0\n",$1,$1,$1}' imposter_train_face.txt >> train_list.txt

awk '{printf "ClientFace/%s ClientFace/%s ClientFace/%s 1\n",$1,$1,$1}' client_test_face.txt > val_private_list.txt
awk '{printf "ImposterFace/%s ImposterFace/%s ImposterFace/%s 0\n",$1,$1,$1}' imposter_test_face.txt >> val_private_list.txt

sed -i 's/\\/\//g' train_list.txt
sed -i 's/\\/\//g' val_private_list.txt


#imposter_train_face.txt > casia-surf-lookalike/train_list.txt
#cat client_test_face.txt imposter_test_face.txt > casia-surf-lookalike/val_private_list.txt


