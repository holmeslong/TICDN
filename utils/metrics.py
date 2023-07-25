# -*- coding: utf-8 -*-

import numpy as np
import torch


def iou(y_pred, y_true, use_sigmoid=True, threshold=0.5):
    """ 计算一张图像的iou """
    smooth = 0.00001
    y_pred = y_pred.squeeze()  # H, W
    y_true = y_true.squeeze()  # H, W
    if use_sigmoid:
        y_pred = y_pred.sigmoid()
    # y_pred = (y_pred > threshold).to(torch.float32)
    inter = (y_true * y_pred).sum()
    union = (y_true + y_pred - y_true * y_pred).sum()
    iou = (inter + smooth) / (union + smooth)
    return iou


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    
    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


class AverageMeter(object):
    """ Compute and store the average and current value """

    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += 1
        self.avg = self.sum / self.count
