from imgaug import augmenters as iaa
import math
import numpy as np
import cv2
# from preprocessing.data_helper import RESIZE_SIZE
from preprocessing.data_helper import *  # random
RESIZE_SIZE = 112


def random_cropping(image, target_shape=(48, 48, 3), is_random=True):
    # randomly crop part of photo. -

    # resize back to 112x112 and then crop random part to target shape
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    target_h, target_w,_ = target_shape
    height, width, _ = image.shape

    if is_random:
        # start_x = random.randint(0, width - target_w)
        # start_y = random.randint(0, height - target_h)
        start_x = np.random.randint(0, width - target_w + 1)
        start_y = np.random.randint(0, height - target_h + 1)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    return zeros


def TTA_5_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],]

    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]
        image_ = zeros.copy()
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images


def TTA_18_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()
        zeros = np.fliplr(zeros)
        image_flip = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images


# ovo ide samo na testu!
# jednu sliku podijeli na 36 podslika
def TTA_36_cropps(image, target_shape=(48, 48, 3)):
    # up to 112x112
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    # half of photo
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y], # middle

              [start_x - target_w, start_y], # left middle
              [start_x, start_y - target_w], # middle up
              [start_x + target_w, start_y], # right middle
              [start_x, start_y + target_w], # middle down

              [start_x + target_w, start_y + target_w], # bottom right
              [start_x - target_w, start_y - target_w], # upper left
              [start_x - target_w, start_y + target_w], # bottom left
              [start_x + target_w, start_y - target_w], # upper right
              ]
    # 9 points

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        # broken to the point dirty fixing is in place
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        # and again, broken...
        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images


def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.5, r1 = 0.5, channel = 3):
    # if random.uniform(0, 1) > probability:
    if np.random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        # target_area = random.uniform(sl, sh) * area
        # aspect_ratio = random.uniform(r1, 1 / r1)

        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            # x1 = random.randint(0, img.shape[0] - h)
            # y1 = random.randint(0, img.shape[1] - w)
            x1 = np.random.randint(0, img.shape[0] - h + 1)
            y1 = np.random.randint(0, img.shape[1] - w + 1)

            noise = np.random.random((h, w, channel))*255
            noise = noise.astype(np.uint8)

            if img.shape[2] == channel:
                img[x1:x1 + h, y1:y1 + w, :] = noise
            else:
                print('wrong')
                return
            return img

    return img


def random_resize(img, probability=0.5, min_ratio=0.2):
    # decide with probability 'probability' to resize the photo
    # absolute crime to do it like this

    # this indeed does produce reproducible results
    r = np.random.uniform(0, 1)
    # print(r)
    # global R
    # if R is True:
    # print("RANDOM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, ", r)
    # R = False
    # print(r)
    if r > probability:
        return img

    # to which resolution to resize (from radio 0.2 to 1.0, with uniform probability)
    ratio = np.random.uniform(min_ratio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio)
    new_w = int(w*ratio)

    # TODO: fix this assertion inv_scale_x > 0 problem
    try:
        img = cv2.resize(img, (new_w, new_h))
    except:

        print("img shape: {}, new_w: {}, new_h: {}, ratio: {}, min_ratio: {}".format(img.shape, new_w, new_h, ratio,
                                                                                     min_ratio))
        exit(-1)

    img = cv2.resize(img, (w, h))
    return img

# def quarter_color_augumentor(image, ):


def color_augumentor(image, target_shape=(48, 48, 3), is_infer=False):
    # print(image.shape)
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ]) # TODO: does not seem to have any effect...

        image = augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)

        # return tensors of 36 patches
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True) #

        # print(image.shape)
        image = augment_img.augment_image(image) # do upper augmentation
        image = random_resize(image) # with probability 50%, resize to 20-100% (uniform distrib)
        image = random_cropping(image, target_shape, is_random=True)  # randomly crop
        # print(image.shape)
        return image


def depth_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image


def ir_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])
        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image


if __name__ == "__main__":
    # import os
    # print(os.getcwd())

    cv2.namedWindow("lala")

    lenna = cv2.imread("lenna.jpeg", 1)
    # print(lenna)q
    cv2.imshow('lala', lenna)
    k = chr(cv2.waitKey())

    # image = np.transpose(lenna, (2, 0, 1))
    # print(image.shape)
    # cv2.imshow('lala', image)
    # k = chr(cv2.waitKey())

    # cv2.waitKey(1) & 0xFF == ord('q')

    dest = TTA_36_cropps(lenna)
    print(len(dest))
    for img in dest:
        print(img.shape)
        cv2.imshow('lala', img[0])
        k = chr(cv2.waitKey())

    cv2.destroyAllWindows()