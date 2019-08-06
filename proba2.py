#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from copy import deepcopy
from FPSmeter import *
from utils import hisEqulColor
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
from process.augmentation import TTA_36_cropps, color_augumentor
from metric import infer_test, infer_test_simple

# TODO: use contants glbaly (including patch size of 48)
RESIZE_SIZE = 112 # from data _helper import RESIZE_SIZE


from model.FaceBagNet_model_A import Net
net = Net(num_class=2, is_first_bn=True)
net = torch.nn.DataParallel(net) # TODO: alternative?
# net = net.cuda()
# net = net.cpu()
net = net.cuda()
net.load_state_dict(torch.load("./models/model_A_color_48/checkpoint/global_min_acer_model.pth",
                               map_location=lambda storage, loc: storage))

# crop_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

# infer_img = deepcopy(crop_img)
infer_img = cv2.imread('/home/loki/Datasets/spoofing/BoKS/Detected/real/AA5742_id154_s0_112.jpg', 1)
loaded_img = deepcopy(infer_img)

infer_img = cv2.resize(infer_img, (RESIZE_SIZE, RESIZE_SIZE))

infer_img = color_augumentor(infer_img, target_shape=(48, 48, 3), is_infer=True)
n = len(infer_img)

infer_img = np.concatenate(infer_img, axis=0)
infer_img = np.transpose(infer_img, (0, 3, 1, 2))
infer_img = infer_img.astype(np.float32)
infer_img = infer_img.reshape([n, 3, 48, 48])
infer_img = infer_img / 255.0

inpt = torch.FloatTensor([infer_img])
# inpt = torch.FloatTensor()
data = [[inpt, inpt]]

result = infer_test_simple(net, data)
# result = result[0]
# result = round(result)
# result = bool(result)
# result = "OK"*(result) + "FAKE"*(not result)
print(result)