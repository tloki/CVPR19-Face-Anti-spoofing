import os
import random
from utils import *

# dataset root path
DATA_ROOT = r'lol/home/loki/Datasets/spoofing/NUAA/Detectedface/casia-surf-lookalike'


TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Val/'
RESIZE_SIZE = 112


# list of training images
def load_train_list(path=None):

    if path is None:
        path = DATA_ROOT

    path += '/train_list.txt'

    f = open(path)
    lines = f.readlines()

    list = []
    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list


# list of validation images
def load_val_list(path=None):
    if path is None:
        path = DATA_ROOT

    path += "/val_private_list.txt"

    list = []
    f = open(path)
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list


# list of test images
def load_test_list(path=None):
    if path is None:
        path = DATA_ROOT

    path += "/test_public_list.txt"

    list = []
    f = open(path)
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list


def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]


def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list



