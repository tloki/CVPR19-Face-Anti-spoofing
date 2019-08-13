#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from mtcnn.mtcnn import MTCNN
from copy import deepcopy
from FPSmeter import *
from utils import histogram_eq_color
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
from preprocessing.augmentation import TTA_36_cropps, color_augumentor
from metric import infer_test, infer_test_simple

# TODO: use contants glbaly (including patch size of 48)
RESIZE_SIZE = 112 # from data _helper import RESIZE_SIZE


font = cv2.FONT_HERSHEY_SIMPLEX
TopLeftCornerOfPicture = (10, 40)  # (left, up is origin)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2
fontColor2 = (0, 255, 0)

# TODO: add mtcnn resolution modifier
# TODO: play with MTCNN parameters


def main():
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)

    # load model
    from model.FaceBagNet_model_A import Net
    net = Net(num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net) # TODO: alternative?
    # net = net.cuda()
    # net = net.cpu()
    net = net.cuda()
    net.load_state_dict(torch.load("./models/model_A_color_48/checkpoint/global_min_acer_model.pth",
                                   map_location=lambda storage, loc: storage))



    # if you want video instead of live webcam view:
    # video_capture = cv2.VideoCapture("/home/loki/Desktop/spoofing/Video/snimka4.mp4")

    video_capture.set(3, 640)
    video_capture.set(4, 480)

    minsize = 25  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    sess = tf.Session()
    with sess.as_default():
        # pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        detector = MTCNN()

        while (True):
            fps = FPSmeter()

            # this is currently a bottleneck
            # TODO: implement from https://gitlab.com/tloki/rpi-fejsanje/blob/master/RasPiCamera.py
            ret, frame = video_capture.read()
            if not ret:
                return

            frame2 = deepcopy(frame)
            frame_hist_eq = histogram_eq_color(frame2)

            # frame =

            detection = detector.detect_faces(frame)


            if detection:

                for i, bbox in enumerate(detection):
                    bbox = bbox['box']
                    pt1 = bbox[0], bbox[1]
                    pt2 = bbox[0] + bbox[2], bbox[1] + int(bbox[3])

                    cv2.rectangle(frame_hist_eq, pt1, pt2, color=(0, 255, 0), thickness=lineType)
                    cv2.putText(frame_hist_eq, "id: "+ str(i), (bbox[0] , bbox[1] - 5), font, fontScale, fontColor,
                                lineType)

                    # TODO: do not hardcode face index
                    if i == 0:
                        crop_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

                        # infer_img = deepcopy(crop_img)
                        infer_img = cv2.imread('/home/loki/Datasets/spoofing/BoKS/Detected/real/AA5742_id154_s0_112.jpg', 1)
                        loaded_img = deepcopy(infer_img)
                        try:
                            infer_img = cv2.resize(infer_img, (RESIZE_SIZE, RESIZE_SIZE))
                        except:
                            continue

                        # infer_img = color_augumentor(infer_img, target_shape=(48, 48, 3), is_infer=True)
                        # n = len(infer_img)

                        # infer_img = np.concatenate(infer_img, axis=0)
                        # infer_img = np.transpose(infer_img, (0, 3, 1, 2))
                        # infer_img = infer_img.astype(np.float32)
                        # infer_img = infer_img.reshape([n, 3, 48, 48])
                        # infer_img = infer_img / 255.0

                        # inpt = torch.FloatTensor([infer_img])
                        # inpt = torch.FloatTensor()
                        # data = [[inpt, inpt]]

                        # result = infer_test_simple(net, data)
                        # result = result[0]
                        # result = round(result)
                        # result = bool(result)
                        # result = "OK"*(result) + "FAKE"*(not result)
                        # print(result)

                        cv2.imshow("id: "+ str(i), infer_img)

            cv2.putText(frame_hist_eq, 'Not ' * (not bool(detection)) +'Detected', TopLeftCornerOfPicture, font,
                        fontScale, fontColor2 if detection else fontColor, lineType)

            fps.tick(frame_hist_eq)
            cv2.imshow('Hist EQ', frame_hist_eq)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    from time import time
    t = time()
    main()
    print(time() - t)
