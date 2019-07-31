#!/usr/bin/env python3

import warnings
import argparse
import os
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
import cv2
import os.path
import ntpath

# grep -io "User.*$" missing.txt | grep -i "unable" | grep -io "/.*\.png" > no_dets.txt
# grep -io "User.*$" missing.txt | grep -i "unable" | grep -io "/.*\.png" > multiple_dets.txt
# 18410 + 701 [no dets] + 193 [multi dets]
# 19304


def get_images_list(path):
    root_dir = path

    pngs = []
    for root, dir, files in os.walk(root_dir):

        for file in files:

            # TODO: make it smarter (non png-hardcoded, case insensitive etc..)

            if file.endswith(".png"):
                pngs.append(os.path.join(root, file))

    # TODO: should BoKS dataset length be hardcoded?
    assert len(pngs) == 19304

    return pngs


def main(config):

    # get all files list (including full path)
    root_dir = config.boks_data_path
    pngs = get_images_list(root_dir)

    # destination dir:
    dest = config.dest_path
    if dest is None:
        parent = os.path.abspath(os.path.join(root_dir, os.pardir))

        if not os.path.exists(os.path.join(parent, "Detected")):
            os.mkdir(os.path.join(parent, "Detected"))

        dest = os.path.abspath(os.path.join(parent, "Detected"))

    # initialize MTCNN face detector
    detector = MTCNN()

    num_det_problems = 0

    for image_path in tqdm(pngs):

        image_bgr = cv2.imread(image_path)
        detection = detector.detect_faces(image_bgr)

        # TODO: what to do?
        if len(detection) == 0:
            num_det_problems += 1
            wrn = "{}. unable to detect face on img {} skipping for now".format(num_det_problems, image_path)
            warnings.warn(wrn)
            continue

        # TODO: bound both objects (probably ghosting is in question!)
        if len(detection) != 1:
            num_det_problems += 1
            wrn = "{}. mltpl dets on {}. skipping...".format(num_det_problems, image_path)
            warnings.warn(wrn)
            # raise UserWarning()
            continue


        detected_face = detection[0]

        source_image_path = ntpath.basename(image_path)
        label = ntpath.basename(os.path.abspath(os.path.join(image_path, os.pardir)))

        img_dest_dir = os.path.join(dest, label)
        if not os.path.exists(img_dest_dir):
            os.mkdir(img_dest_dir)

        # TODO: again hardcoded .png
        dest_image_name = source_image_path[0:-3] + "jpg"

        bbox = detected_face['box']
        pt1 = bbox[0], bbox[1]
        pt2 = bbox[0] + bbox[2], bbox[1] + int(bbox[3])

        crop_img = image_bgr[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]


        dest_img_fullpath = os.path.join(img_dest_dir, dest_image_name)

        # if "0145.jpg" in dest_img_fullpath:
        #     continue

        if crop_img.shape[0] * crop_img.shape[1] == 0:
            wrn1 = "image '{}' detected face dimensions: {}".format(dest_img_fullpath, crop_img.shape)
            num_det_problems += 1
            wrn = "{}. det size dim == 0 on {}. skipping...".format(num_det_problems, image_path)
            warnings.warn(wrn1 + "\n" + wrn)
            # raise UserWarning()
            continue

        cv2.imwrite(dest_img_fullpath, crop_img)






if __name__ == '__main__':
    # parse BoKS dataset path
    parser = argparse.ArgumentParser()
    parser.add_argument('--boks_data_path', type=str, default=None, required=True)
    parser.add_argument('--dest_path', type=str, default=None, required=False)
    config = parser.parse_args()

    # print(config)

    main(config)
