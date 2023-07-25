# -*- coding: utf-8 -*-

import torch


class Config(object):

    # network config
    in_channels = 3
    num_classes = 2
    # resnet_pretrained = 'https://download.pytorch.org/models/resnet34-b627a593.pth'
    resnet_pretrained = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    pretrained = ""
    img_size = (256, 256)
    input_shape = (3, 256, 256)

    # dataset config
    root = "E:/Datasets/WHU_Two/whu_256x256"
    ignore_label = 0
    mean_bgr = [104.008, 116.669, 122.675]
    split_train = 'train'
    split_val = 'val'
    split_test = 'test'
    split_bbox = 'annotations'

    # train config
    seed = 613
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 30  # 需要改的值 30
    iter_max = 650  # 需要改的值 1530
    lr_gen1 = 5e-4  # 需要改的值
    lr_dis1 = 5e-6  # 需要改的值
    lr_encoder = 5e-5  # 需要改的值 1e-4
    lr_decoder = 1e-4  # 需要改的值 1e-4
    lr_d2 = 1e-5  # 需要改的值 1e-5
    lr_seg = 1e-4  # 需要改的值 5e-4
    lr_d3 = 2e-5  # 需要改的值 2e-5
    lr_sigma = 1e-2  # 需要改的值
    # lr_other = 1e-4  # 2e-4 bs: 8
    # lr_seg = 5e-4  # 1e-3 bs: 8
    # lr_G = 0.001
    # lr_D = 0.0001
    lr_drop = 7400
    drop_gamma = 0.1
    # iter_train_gen = 2
    labmda_semantic = 0.5
    save_path = "./checkpoint/deepGC_GAN_WHU/backbone_drn_a/ablation/baseline_fcgc_d2_diff/cfm2"
    iter_save_txt = 20  # 需要改的值
    iter_val = 650  # 需要改的值
    save_max_num = 5  # 需要改的值
    batch_size = 2  # 需要改的值
    num_worker = 0  # 表示开启几个子线程加载数据, 0表示不开启子线程, 数据只在主线程被加载
    warm_steps = 500

    # 伪标签生成
    tau_p = 0.65
    p_batch_size = 2
    layer_nums = 15
    
    # test config
    save_pred_dir = "./pred_mask/"
    threshold = 0.5
