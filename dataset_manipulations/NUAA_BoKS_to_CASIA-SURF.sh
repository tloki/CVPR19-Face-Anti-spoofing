#!/usr/bin/env bash

combine_dir=$1
nuaa_dir=$2
boks_dir=$3

# ignores case
NUA_EXT="jpg"

echo combined dir:      $combine_dir
echo NUAA dataset dir:  $nuaa_dir
echo BoKS dataset dir:  $boks_dir

mkdir -p $combine_dir

######################
# NUAA part ##########
######################

PERSON_TRAIN_THRES=11

rm -rf $combine_dir/nuaa_bonafide.txt
rm -rf $combine_dir/nuaa_presentation.txt
rm -rf $combine_dir/nuaa_train.txt
rm -rf $combine_dir/nuaa_validation.txt

touch $combine_dir/nuaa_bonafide.txt
touch $combine_dir/nuaa_presentation.txt
touch $combine_dir/nuaa_train.txt
touch $combine_dir/nuaa_validation.txt

# bonafide
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ClientFace/ | grep -i "\.${NUA_EXT}\$"`
label=1

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_bonafide.txt

    if [ $person -le $PERSON_TRAIN_THRES ]
    then
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_test.txt
    fi
done

# fake
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ImposterFace/ | grep -i "\.${NUA_EXT}\$"`
label=0

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_presentation.txt

    if [ $person -le $PERSON_TRAIN_THRES ]
    then
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_test.txt
    fi
done

# validation:
# for now assume cross-dataset (test on whole dataset)
cat nuaa_bonafide.txt nuaa_presentation.txt > nuaa_validation.txt



######################
# BoKS part ##########
######################

PERSON_TRAIN_THRES=11

rm -rf $combine_dir/nuaa_bonafide.txt
rm -rf $combine_dir/nuaa_presentation.txt
rm -rf $combine_dir/nuaa_train.txt
rm -rf $combine_dir/nuaa_validation.txt

touch $combine_dir/nuaa_bonafide.txt
touch $combine_dir/nuaa_presentation.txt
touch $combine_dir/nuaa_train.txt
touch $combine_dir/nuaa_validation.txt

# bonafide
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ClientFace/ | grep -i "\.${NUA_EXT}\$"`
label=1

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_bonafide.txt

    if [ $person -le $PERSON_TRAIN_THRES ]
    then
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_test.txt
    fi
done

# fake
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ImposterFace/ | grep -i "\.${NUA_EXT}\$"`
label=0

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_presentation.txt

    if [ $person -le $PERSON_TRAIN_THRES ]
    then
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} ${path} ${path} ${label}" >> $combine_dir/nuaa_test.txt
    fi
done




#
#
#awk '{printf "ClientFace/%s ClientFace/%s ClientFace/%s 1\n",$1,$1,$1}' client_train_face.txt > train_list.txt
#awk '{printf "ImposterFace/%s ImposterFace/%s ImposterFace/%s 0\n",$1,$1,$1}' imposter_train_face.txt >> train_list.txt
#
#awk '{printf "ClientFace/%s ClientFace/%s ClientFace/%s 1\n",$1,$1,$1}' client_test_face.txt > val_private_list.txt
#awk '{printf "ImposterFace/%s ImposterFace/%s ImposterFace/%s 0\n",$1,$1,$1}' imposter_test_face.txt >> val_private_list.txt
#
#sed -i 's/\\/\//g' train_list.txt
#sed -i 's/\\/\//g' val_private_list.txt


#imposter_train_face.txt > casia-surf-lookalike/train_list.txt
#cat client_test_face.txt imposter_test_face.txt > casia-surf-lookalike/val_private_list.txt


