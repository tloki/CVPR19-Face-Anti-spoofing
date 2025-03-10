import numpy as np
import torch
from scipy import interpolate
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import dummy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)

    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc


def calculate(threshold, dist, actual_issame):
    predict_is_same = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_is_same, actual_issame))
    fp = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_is_same), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_is_same), actual_issame))
    return tp, fp, tn, fn


def calculate_acer(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer, tp, fp, tn,fn


def tpr_fpr(dist, actual_issame, fpr_target = 0.001):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind='slinear')
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    print(str(FPR)+' '+str(TPR))
    return FPR,TPR


def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob


def do_valid(net, test_loader, criterion ):
    valid_num = 0
    losses = []
    corrects = []
    probs = []
    labels = []

    for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    # ----------------------------------------------

    correct = np.concatenate(corrects)
    loss = np.concatenate(losses)
    loss = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer, _, _, _, _ = calculate_acer(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]


def do_valid_test(net, validation_loader, criterion, device="cpu", logger=dummy):
    valid_num  = 0
    losses = []
    corrects = []
    probs = []
    labels = []

    # for i, (input, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
    for i, (inpt, truth) in tqdm(enumerate(validation_loader), desc="valid progress", total=len(validation_loader),
                                 leave=False):
    # for i, (inpt, truth) in enumerate(validation_loader):
        b, n, c, w, h = inpt.size()
        inpt = inpt.view(b*n, c, w, h)

        inpt = inpt.cuda() if torch.cuda.is_available() else input.cpu()
        truth = truth.cuda() if torch.cuda.is_available() else truth.cpu()

        # input = input.to(device)
        # input

        with torch.no_grad():
            logit, _ ,_ = net(inpt)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim=1, keepdim=False)

            truth = truth.view(logit.shape[0])
            loss = criterion(logit, truth, False) # report average loss
            # loss2 = F.cross_entropy(logit, truth, reduce=False)

            # assert loss.data.cpu.numpy() == loss2.data.cpu.numpy()

            correct, prob = metric(logit, truth)

        valid_num += len(inpt)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    # ----------------------------------------------

    correct = np.concatenate(corrects)
    loss = np.concatenate(losses)
    ll = len(losses)
    loss = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer, _, _, _, _ = calculate_acer(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct, tpr, fpr
    ])

    # print()
    # print("[valid] tpr: {:.2f}% fpr: {:.2f}% acc: {:.2f}% acer: {:.2f}% loss: {:.5f} crct: {:.2f}%".format(
    #     tpr*100, fpr*100, acc*100, acer*100, loss, correct*100))

    # validation loss,
    return valid_loss,[probs[:, 1], labels]


def infer_test(net: nn.Module, test_loader, device="cpu"):
    valid_num = 0
    # probs = torch.tensor([], device=device)
    probs = []

    for i, (inpt, truth) in enumerate(tqdm(test_loader)):
        b, n, c, w, h = inpt.size()
        # print(b, n, c, w, h)
        inpt = inpt.view(b*n,c,w,h)
        # input = input.cuda() if torch.cuda.is_available() else input.cpu()
        inpt = inpt.to(device)

        with torch.no_grad():
            logit, _, _ = net(inpt)
            logit = logit.view(b, n, 2)
            logit = torch.mean(logit, dim=1, keepdim=False)
            prob = F.softmax(logit, 1)

        valid_num += len(inpt)
        probs.append(prob.data.cpu().numpy())

        # probs.concatenate()

        # print("ok")
        # probs.append(prob.data.numpy())
    # print(probs)
    out_probs = np.concatenate(probs)
    return out_probs[:, 1]


def infer_test_simple(net, test_loader):
    valid_num = 0
    probs = []

    for inpt, truth in test_loader:
        b, n, c, w, h = input.size()
        inpt = inpt.view(b*n, c, w, h)
        inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()

        with torch.no_grad():
            logit, _, _ = net(inpt)
            logit = logit.view(b, n, 2)
            logit = torch.mean(logit, dim=1, keepdim=False)
            prob = F.softmax(logit, 1)

        valid_num += len(inpt)
        probs.append(prob.data.cpu().numpy())

    probs = np.concatenate(probs)
    probs = probs[:, 1]
    return probs


# def infer_test_realtime(net, test_loader):
#     from FBN import FaceBagNet
#     model48 = FaceBagNet("model48.pth", 48, "cuda")
#
#     # print(model48.neural_net)
#
#     # for param in net.parameters():
#     #     print(param.data)
#     #     exit(-1)
#
#     import cv2
#     valid_num = 0
#     probs = []
#
#     cv2.namedWindow("detected")
#     print("started realtime detection loop:")
#
#     while True:
#         # try:
#         inpt, raw_image = test_loader[0]
#         # except:
#         #     print(".")
#         #     continue
#
#         inp = inpt.size()
#         b, n, c, w, h = inp
#         print(b, n, c, w, h)
#         inpt = inpt.view(b*n, c, w, h)
#         inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()
#
#         with torch.no_grad():
#             logit, _, _ = net(inpt)
#             logit = logit.view(b, n, 2)
#             logit = torch.mean(logit, dim=1, keepdim=False)
#             prob = F.softmax(logit, 1)
#
#         is_real = list(prob.data.cpu().numpy()[:, 1])[0]
#
#         print((is_real > 0.5)*"ok" + (is_real < 0.5)*"FAKE", is_real, model48.predict(raw_image))
#
#         valid_num += len(inpt)
#         probs.append(prob.data.cpu().numpy())
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     probs = np.concatenate(probs)
#     probs = probs[:, 1]
#
#     return probs