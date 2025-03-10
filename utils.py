from torch.autograd import Variable
import torch.nn.functional as F
import os
import shutil
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from preprocessing.augmentation import color_augumentor, ir_augumentor, depth_augumentor


# from https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
def histogram_eq_color(img):
    ycr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycr_cb)
    len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycr_cb)
    cv2.cvtColor(ycr_cb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def save(list_or_dict, name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()


def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp


def acc(preds, labels, th=0.0):
    preds = (preds > th).int()
    labels = labels.int()
    return (preds == labels).float().mean()


def dot_numpy(vector1, vector2, emb_size=512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)
    cosV12 = np.dot(vector1, vector2)
    return cosV12


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# this one is used on train
def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    # loss = F.cross_entropy(logit, truth, reduce=is_average) # deprecated
    loss = None
    # try:
    loss = F.cross_entropy(logit, truth, reduction='mean' if is_average else 'none')
    # except:
    #     print("old version of pytorch used")
    # loss = F.cross_entropy(logit, truth, reduction='elementwise_mean' if is_average else 'none')
    # print(loss)

    return loss


def bce_criterion(logit, truth, is_average=True):
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduce=is_average)
    return loss


def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """
    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec'%(min,sec)
    elif mode == 'milis' or mode == 'ms' or mode.startswith("mil"):
        ti = int(t)
        decs = t - ti
        decs = int(decs * 1000)
        sec = ti % 60
        return '%02d s %02d ms' % (sec, decs)
    else:
        raise NotImplementedError


def np_float32_to_uint8(x, scale=255.0):
    return (x*scale).astype(np.uint8)


def np_uint8_to_float32(x, scale=255.0):
    return (x/scale).astype(np.float32)


def dummy(*args, **kwargs):
    pass


def get_model(model_name, num_class, is_first_bn) -> nn.Module:
    if model_name == 'baseline':
        from model.model_baseline import Net
    elif model_name == 'model_A':
        from model.FaceBagNet_model_A import Net
    elif model_name == 'model_B':
        from model.FaceBagNet_model_B import Net
    elif model_name == 'model_C':
        from model.FaceBagNet_model_C import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net


def get_augment(image_mode):

    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        # print("layer")
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp