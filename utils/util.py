# -*- coding: utf-8 -*-

import math
import os
import json
import pdb
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F

from grabcut import GCClient
from utils.metrics import scores


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


# ----------
# 学习率更新
# ----------

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=15./32.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_attn_fea(feature, attn, num):
    feature = feature.detach().cpu().numpy()  # (C, H, W)
    weights = attn.detach().cpu().numpy().squeeze()  # (C,)
    rank_weights = np.argsort(weights)
    choose_feas = feature[rank_weights[-num:]]
    return choose_feas


def get_deepgc_mask(fea_lst, img_size, img, bboxes):
    concat_fea = np.reshape(fea_lst, (-1, img_size[0] * img_size[1]))
    concat_fea = np.transpose(concat_fea).reshape((*img_size, -1))
    concat_fea = np.mean(concat_fea, axis=2)
    concat_fea = np.concatenate([img, concat_fea], axis=2)
    GC = GCClient(img, 5, d=concat_fea.shape[-1], bboxes=bboxes, fea=concat_fea)
    GC.run()  # !!!
    return GC._mask


def deep_grabcut(cfg, encoder, dataloader, img_root_dir):
    encoder.eval()

    with torch.no_grad():
        pseudo_label_unmask = {}
        for img_id, img_a, img_b, _, bbox_mask in dataloader:
            img_a, img_b = img_a.to(cfg.device), img_b.to(cfg.device)
            bbox_mask = bbox_mask.to(cfg.device)

            # 保存特征图
            layer_outputs = []    # [a_fea_l1: (B, C, H, W), b_fea_l1: (B, C, H, W)]
            seblock_outputs = []  # [a_fea_l1: (B, C, 1, 1), b_fea_l1: (B, C, 1, 1)]
            def layer_hook(module, input, output):
                if isinstance(output, tuple):
                    seblock_outputs.append(output[1])
                else:
                    output = F.interpolate(output, size=cfg.img_size, mode='bilinear')
                    layer_outputs.append(output)
                
            hook_layer1 = encoder.layer1[-1].register_forward_hook(layer_hook)
            hook_seblock1 = encoder.channel_attn.register_forward_hook(layer_hook)
            _ = encoder(img_a)
            _ = encoder(img_b)

            for i in range(len(img_id)):
                img_names = ['A', 'B']
                label_unmask = np.zeros(cfg.img_size, dtype=np.uint8)
                bboxes = np.loadtxt(os.path.join(cfg.root, cfg.split_bbox, "{}.txt".format(img_id[i][:-4])),
                                    delimiter=',', dtype=np.int64)
                if len(bboxes.shape) == 1:
                    bboxes = np.array([bboxes])
                assert bboxes.shape[-1] == 5, "The shape of bbox is wrong!"
                
                a_path = os.path.join(img_root_dir, "A", img_id[i])
                b_path = os.path.join(img_root_dir, "B", img_id[i])
                image_a = cv2.imread(a_path, cv2.IMREAD_COLOR)
                image_b = cv2.imread(b_path, cv2.IMREAD_COLOR)
                feature_a = get_attn_fea(layer_outputs[0][i], seblock_outputs[0][i], 100)
                feature_b = get_attn_fea(layer_outputs[1][i], seblock_outputs[1][i], 100)
                for bbox in bboxes:
                    if bbox[-1] == 0:
                        mask = get_deepgc_mask(feature_a, cfg.img_size, image_a, [bbox])
                    else:
                        mask = get_deepgc_mask(feature_b, cfg.img_size, image_b, [bbox])
                    label_unmask[(mask == 1) | (mask == 3)] = 1
                pseudo_label_unmask[img_id[i]] = label_unmask
            
            hook_layer1.remove()
            hook_seblock1.remove()

    return pseudo_label_unmask


# -----------------
# 可视化生成的伪标签
# -----------------

def visual_pseudo_label(path, save_path):
    os.makedirs(save_path, exist_ok=True)

    data = pickle.load(open(path, 'rb'))
    for key, value in tqdm(data.items()):
        value[value == 1] = 255
        cv2.imwrite(os.path.join(save_path, key), value.astype(np.uint8))
