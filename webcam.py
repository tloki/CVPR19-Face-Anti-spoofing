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


font = cv2.FONT_HERSHEY_SIMPLEX
TopLeftCornerOfPicture = (10, 40) # (left, up is origin)
fontScale = 1
fontColor = (0,0,255)
lineType = 2
fontColor2 = (0,255,0)

# TODO: add mtcnn resolution modifier
# TODO: play with MTCNN parameters





def main():
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    # video_capture = cv2.VideoCapture(0)

    # if you want video instead of live webcam view:
    video_capture = cv2.VideoCapture("/home/loki/Desktop/spoofing/Video/snimka4.mp4")

    video_capture.set(3, 1280)
    video_capture.set(4, 720)

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
            frame_hist_eq = hisEqulColor(frame2)

            detection = detector.detect_faces(frame)


            if detection:

                for i, bbox in enumerate(detection):
                    bbox = bbox['box']
                    pt1 = bbox[0], bbox[1]
                    pt2 = bbox[0] + bbox[2], bbox[1] + int(bbox[3])

                    cv2.rectangle(frame_hist_eq, pt1, pt2, color=(0, 255, 0), thickness=lineType)
                    cv2.putText(frame_hist_eq, "id: "+ str(i), (bbox[0] , bbox[1] - 5), font, fontScale, fontColor,
                                lineType)

                    if i == 0:
                        crop_img = frame_hist_eq[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        cv2.imshow('crop', crop_img)


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
