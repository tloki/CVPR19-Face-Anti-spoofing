#!/usr/bin/env bash

nuaa_dir=$1

echo nua dir located in $nuaa_dir


# suppose using Detectedface format of NUAA
# TODO: add url to download NUAA detected face data

cd $nuaa_dir
echo working in `pwd` dir

rm -rf casia-surf-lookalike
mkdir casia-surf-lookalike

awk '{printf "%s %s %s 1\n",$1,$1,$1}' client_train_face.txt > casia-surf-lookalike/train_list.txt
awk '{printf "%s %s %s 0\n",$1,$1,$1}' imposter_train_face.txt >> casia-surf-lookalike/train_list.txt

awk '{printf "%s %s %s 1\n",$1,$1,$1}' client_test_face.txt >> casia-surf-lookalike/val_private_list.txt
awk '{printf "%s %s %s 0\n",$1,$1,$1}' imposter_test_face.txt >> casia-surf-lookalike/val_private_list.txt

#imposter_train_face.txt > casia-surf-lookalike/train_list.txt
#cat client_test_face.txt imposter_test_face.txt > casia-surf-lookalike/val_private_list.txt


