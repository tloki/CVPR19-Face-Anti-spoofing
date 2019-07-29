#!/usr/bin/env bash

combine_dir=$1
nuaa_dir=$2
boks_dir=$3

# BoKS requires BoKS_face_detector.py to be preprocessed !

# ignores case
NUAA_EXT="jpg"
BOKS_EXT="jpg"

echo combined dir:      $combine_dir
echo NUAA dataset dir:  $nuaa_dir
echo BoKS dataset dir:  $boks_dir

mkdir -p $combine_dir

NUAA_PERSON_TRAIN_THRES=11
BOKS_PERSON_TRAIN_THRES=50
# TODO: problem - poslije 20. su svi "youtube"

#######################
## NUAA part ##########
#######################

#NUAA_PERSON_TRAIN_THRES=11

# dataset consists of 15 persons
#

rm -rf $combine_dir/nuaa_bonafide.txt
rm -rf $combine_dir/nuaa_presentation.txt
rm -rf $combine_dir/nuaa_train.txt
rm -rf $combine_dir/nuaa_validation.txt
rm -rf $combine_dir/nuaa_whole_dataset.txt
rm -rf $combine_dir/nuaa_info.txt

touch $combine_dir/nuaa_bonafide.txt
touch $combine_dir/nuaa_presentation.txt
touch $combine_dir/nuaa_train.txt
touch $combine_dir/nuaa_validation.txt
touch $combine_dir/nuaa_whole_dataset.txt
touch $combine_dir/nuaa_info.txt

# bonafide
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ClientFace/ | grep -i "\.${NUAA_EXT}\$"`
label=1

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_bonafide.txt

    if [ $person -le $NUAA_PERSON_TRAIN_THRES ]
    then
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_validation.txt
        # TODO: check if color, ir, depth
    fi
done

# fake
nua_jpg=`ls -LR $nuaa_dir/Detectedface/ImposterFace/ | grep -i "\.${NUAA_EXT}\$"`
label=0

for i in $nua_jpg; do

    person=`echo $i | cut -c 1-4`
    path="${nuaa_dir}/Detectedface/ClientFace/${i}"

    echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_presentation.txt

    if [ $person -le $NUAA_PERSON_TRAIN_THRES ]
    then
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_train.txt
    else
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/nuaa_validation.txt
    fi
done

# validation:
# for now assume cross-dataset (test on whole dataset)
cat $combine_dir/nuaa_bonafide.txt $combine_dir/nuaa_presentation.txt >> $combine_dir/nuaa_whole_dataset.txt

######################
## dataset summary: ##
######################
echo " "

num_files=`ls -LR $nuaa_dir/Detectedface | grep -i "\.${NUAA_EXT}\$" | wc -l`
num_files2=`wc -l $combine_dir/nuaa_whole_dataset.txt`
echo "NUAA: number of samples: ${num_files} == ${num_files2}"
echo "NUAA: number of samples: ${num_files} == ${num_files2}" >> $combine_dir/nuaa_info.txt

num_files_bonafide=`grep -i "1\$" $combine_dir/nuaa_whole_dataset.txt | wc -l`
num_files_presentation=`grep -i "0\$" $combine_dir/nuaa_whole_dataset.txt | wc -l`

num_files_bonafide2=`ls -LR $nuaa_dir/Detectedface/ClientFace | grep -i "\.${NUAA_EXT}\$" | wc -l`
num_files_presentation2=`ls -LR $nuaa_dir/Detectedface/ImposterFace | grep -i "\.${NUAA_EXT}\$" | wc -l`

echo "NUAA: number of bonafide samples: ${num_files_bonafide} == ${num_files_bonafide2}"
echo "NUAA: number of presentation samples: ${num_files_presentation} == ${num_files_presentation2}"

echo "NUAA: number of bonafide samples: ${num_files_bonafide} == ${num_files_bonafide2}" >> $combine_dir/nuaa_info.txt
echo "NUAA: number of presentation samples: ${num_files_presentation} == ${num_files_presentation2}" >> $combine_dir/nuaa_info.txt

num_train=`wc -l $combine_dir/nuaa_train.txt`
echo "NUAA: number of train samples: ${num_train}"
echo "NUAA: number of train samples: ${num_train}" >> $combine_dir/nuaa_info.txt

num_test=`wc -l $combine_dir/nuaa_validation.txt`
echo "NUAA: number of test samples: ${num_test}"
echo "NUAA: number of test samples: ${num_test}" >> $combine_dir/nuaa_info.txt

num_test_bonafide=`grep -i "1\$" $combine_dir/nuaa_validation.txt | wc -l`
num_test_presentation=`grep -i "0\$" $combine_dir/nuaa_validation.txt | wc -l`
echo "NUAA: number of test bonafide samples: ${num_test_bonafide}"
echo "NUAA: number of test presentation samples: ${num_test_presentation}"

echo "NUAA: number of test bonafide samples: ${num_test_bonafide}" >> $combine_dir/nuaa_info.txt
echo "NUAA: number of test presentation samples: ${num_test_presentation}" >> $combine_dir/nuaa_info.txt

num_train_bonafide=`grep -i "1\$" $combine_dir/nuaa_train.txt | wc -l`
num_train_presentation=`grep -i "0\$" $combine_dir/nuaa_train.txt | wc -l`
echo "NUAA: number of train bonafide samples: ${num_train_bonafide}"
echo "NUAA: number of train presentation samples: ${num_train_presentation}"

echo "NUAA: number of train bonafide samples: ${num_train_bonafide}" >> $combine_dir/nuaa_info.txt
echo "NUAA: number of train presentation samples: ${num_train_presentation}" >> $combine_dir/nuaa_info.txt

echo " "
echo " " >> $combine_dir/nuaa_info.txt


######################
# BoKS part ##########
######################

#BOKS_PERSON_TRAIN_THRES=78
# 32 spoof person, 48 person real in test
# 73 spoof in train, 62 real in train
#

rm -rf $combine_dir/boks_bonafide.txt
rm -rf $combine_dir/boks_presentation.txt

rm -rf $combine_dir/boks_train.txt
rm -rf $combine_dir/boks_validation.txt

rm -rf $combine_dir/boks_whole_dataset.txt
rm -rf $combine_dir/boks_info.txt

#TODO: not good test - has train and validation samples in it
rm -rf $combine_dir/boks_test.txt

touch $combine_dir/boks_bonafide.txt
touch $combine_dir/boks_presentation.txt

touch $combine_dir/boks_train.txt
touch $combine_dir/boks_validation.txt

touch $combine_dir/boks_whole_dataset.txt
touch $combine_dir/boks_info.txt

# bonafide
boks_jpg=`ls -LR $boks_dir/Detected/real/ | grep -i "\.${BOKS_EXT}\$"`
label=1

ignored_photos_real=0
ignored_photos_spoof=0


for i in $boks_jpg; do

#    echo $i
    no_extension_name="${i%.*}"

    # count number of "_" ocurences (check if name is not just a [number].jpg)

    just_delims="${no_extension_name//[^_]}"
    num_delims="${#just_delims}"

    if [ $num_delims -ne 3 ]
    then
#        echo "ignoring ${i}"
        ignored_photos_real=$((ignored_photos_real+1))
        continue # TODO: do not just ignore 0000.jpg and simmilar...
        # TODO: This was originally validation - put them all to test?
        # TODO: for now just ignore them because all of them are contained in train set..
    fi

    iter_count=1
    for j in $(echo $no_extension_name | tr "_" "\n"); do
        if [ $iter_count -eq 1 ]; then camera=$j; fi
        if [ $iter_count -eq 2 ]; then person=`echo $j | grep -io "[0-9]\+"`; fi
        if [ $iter_count -eq 3 ]; then session=`echo $j | grep -io "[0-9]\+"`; fi
        if [ $iter_count -eq 4 ]; then num=$j; fi

        iter_count=$((iter_count+1))
    done

#    echo "camera: ${camera}, person id: ${person}, session nr: ${session}, number: ${num}"

#    person=`echo $i | cut -c 1-4`
    path="${boks_dir}/Detected/real/${i}"

    echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_bonafide.txt

    if [ $person -le $BOKS_PERSON_TRAIN_THRES ]
    then
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_train.txt
    else
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_validation.txt
    fi
done

# fake
boks_jpg=`ls -LR $boks_dir/Detected/spoof/ | grep -i "\.${BOKS_EXT}\$"`
label=0


for i in $boks_jpg; do

#    echo $i
    no_extension_name="${i%.*}"

    # count number of "_" ocurences (check if name is not just a [number].jpg)

    just_delims="${no_extension_name//[^_]}"
    num_delims="${#just_delims}"

    if [ $num_delims -ne 5 ]
    then
#        echo "ignoring ${i}"
        ignored_photos_spoof=$((ignored_photos_spoof+1))
        continue # TODO: do not just ignore 0000.jpg and simmilar...
        # TODO: This was originaly validation - put them all to test?
        # TODO: for now just ignore them because all of them are contained in train set..
    fi

#    echo "not ignoring ${i}"
#    exit -1

    iter_count=1
    for j in $(echo $no_extension_name | tr "_" "\n"); do
        if [ $iter_count -eq 1 ]; then m1=$j; fi
        if [ $iter_count -eq 2 ]; then m2=$j; fi
        if [ $iter_count -eq 3 ]; then m3=$j; fi
        if [ $iter_count -eq 4 ]; then person=`echo $j | grep -io "[0-9]\+"`; fi
        if [ $iter_count -eq 5 ]; then session=`echo $j | grep -io "[0-9]\+"`; fi
        if [ $iter_count -eq 6 ]; then num=$j; fi

        iter_count=$((iter_count+1))
    done

#    echo "person id: ${person}, session nr: ${session}, number: ${num}"

#    person=`echo $i | cut -c 1-4`
    path="${boks_dir}/Detected/spoof/${i}"

    echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_presentation.txt

    if [ $person -le $BOKS_PERSON_TRAIN_THRES ]
    then
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_train.txt
    else
        echo "${path} NoIR NoDepth ${label}" >> $combine_dir/boks_validation.txt
    fi
done

ignored_samples=$((ignored_photos_real+ignored_photos_spoof))

#echo ignored spoof, real
#echo $ignored_photos_real
#echo $ignored_photos_spoof


# validation:
# for now assume cross-dataset (test on whole dataset)
cat $combine_dir/boks_bonafide.txt $combine_dir/boks_presentation.txt >> $combine_dir/boks_whole_dataset.txt

######################
## dataset summary: ##
######################
echo " "

num_files=`(ls -LR $boks_dir/Detected/real && ls -LR $boks_dir/Detected/spoof) | grep -i "\.${BOKS_EXT}\$" | wc -l`
num_files=$((num_files-ignored_samples))
num_files2=`wc -l $combine_dir/boks_whole_dataset.txt`
echo "BoKS: number of samples: ${num_files} == ${num_files2}"
echo "BoKS: number of samples: ${num_files} == ${num_files2}" >> $combine_dir/boks_info.txt

num_files_bonafide=`grep -i "1\$" $combine_dir/boks_whole_dataset.txt | wc -l`
num_files_presentation=`grep -i "0\$" $combine_dir/boks_whole_dataset.txt | wc -l`

num_files_bonafide2=`ls -LR $boks_dir/Detected/real | grep -i "\.${BOKS_EXT}\$" | wc -l`
num_files_presentation2=`ls -LR $boks_dir/Detected/spoof | grep -i "\.${BOKS_EXT}\$" | wc -l`
num_files_bonafide2=$((num_files_bonafide2-ignored_photos_real))
num_files_presentation2=$((num_files_presentation2-ignored_photos_spoof))

echo "BoKS: number of bonafide samples: ${num_files_bonafide} == ${num_files_bonafide2}"
echo "BoKS: number of presentation samples: ${num_files_presentation} == ${num_files_presentation2}"

echo "BoKS: number of bonafide samples: ${num_files_bonafide} == ${num_files_bonafide2}" >> $combine_dir/boks_info.txt
echo "BoKS: number of presentation samples: ${num_files_presentation} == ${num_files_presentation2}" >> $combine_dir/boks_info.txt

num_train=`wc -l $combine_dir/boks_train.txt`
echo "BoKS: number of train samples: ${num_train}"
echo "BoKS: number of train samples: ${num_train}" >> $combine_dir/boks_info.txt

num_test=`wc -l $combine_dir/boks_validation.txt`
echo "BoKS: number of test samples: ${num_test}"
echo "BoKS: number of test samples: ${num_test}" >> $combine_dir/boks_info.txt

num_test_bonafide=`grep -i "1\$" $combine_dir/boks_validation.txt | wc -l`
num_test_presentation=`grep -i "0\$" $combine_dir/boks_validation.txt | wc -l`
echo "BoKS: number of test bonafide samples: ${num_test_bonafide}"
echo "BoKS: number of test presentation samples: ${num_test_presentation}"

echo "BoKS: number of test bonafide samples: ${num_test_bonafide}" >> $combine_dir/boks_info.txt
echo "BoKS: number of test presentation samples: ${num_test_presentation}" >> $combine_dir/boks_info.txt

num_train_bonafide=`grep -i "1\$" $combine_dir/boks_train.txt | wc -l`
num_train_presentation=`grep -i "0\$" $combine_dir/boks_train.txt | wc -l`
echo "BoKS: number of train bonafide samples: ${num_train_bonafide}"
echo "BoKS: number of train presentation samples: ${num_train_presentation}"

echo "BoKS: number of train bonafide samples: ${num_train_bonafide}" >> $combine_dir/boks_info.txt
echo "BoKS: number of train presentation samples: ${num_train_presentation}" >> $combine_dir/boks_info.txt

echo " "
echo " " >> $combine_dir/boks_info.txt
