# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.decoder import Up
from networks.dfm import CorrelationFuseModule


def deconv2d(in_channels, out_channels, ks=3, stride=2, padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, ks, stride, padding, output_padding),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlockRP(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlockRP, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # 反射填充，不填充0，适用于风格迁移
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),  # 在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResBlockIns(nn.Module):

    def __init__(self, in_features, out_features):
        super(ResBlockIns, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, 3, 1, 1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


# ----------------------------------------------------------------
# Discriminator: discriminate if image is pre image or post image
# ----------------------------------------------------------------

class Discriminator(nn.Module):

    def __init__(self, input_shape) -> None:
        super().__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """ Returns downsampling layers of each discriminator block """
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, 1),  # Nx512x31x31
            nn.ZeroPad2d((1, 1, 1, 1)),         # Nx512x33x33 
            nn.Conv2d(512, 1, 4, padding=1)     # Nx1x32x32
        )

    def forward(self, img):
        return self.model(img)


# ----------------------------------------
# Decoder
# 1. get reconstructed pre image
# 2. get post image stylized as pre image
# ----------------------------------------

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        mid_channels = 32

        # Conv1
        model = [
            nn.Conv2d(in_channels, mid_channels * 4, 3, 1, 1),
            nn.InstanceNorm2d(mid_channels * 4),
            nn.ReLU(inplace=True)
        ]

        # ResBlockIns
        for _ in range(4):
            model += [ResBlockIns(mid_channels * 4, mid_channels * 4)]

        # Deconv2d
        model += [
            deconv2d(mid_channels * 4, mid_channels * 2),
            deconv2d(mid_channels * 2, mid_channels * 2),
            deconv2d(mid_channels * 2, mid_channels * 2),
            deconv2d(mid_channels * 2, mid_channels)
        ]

        # Conv out layer
        model += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# Segmenter: get segmentation mask
# ----------------------------------------

class Segmenter(nn.Module):

    def __init__(self):
        super(Segmenter, self).__init__()

        chn_lst = [256, 512, 1024, 2048]

        # 特征融合部分
        self.diff1 = CorrelationFuseModule(chn_lst[0])
        self.diff2 = CorrelationFuseModule(chn_lst[1])
        self.diff3 = CorrelationFuseModule(chn_lst[2])
        self.diff4 = CorrelationFuseModule(chn_lst[3])

        # 特征解码器
        # Decoder
        self.up4 = Up(chn_lst[3] + chn_lst[2], chn_lst[2])
        self.up3 = Up(chn_lst[2] + chn_lst[1], chn_lst[1])
        self.up2 = Up(chn_lst[1] + chn_lst[0], chn_lst[0])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(chn_lst[0], chn_lst[0] // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chn_lst[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(chn_lst[0] // 4, 1, 1, stride=1, padding=0)
        )

    def forward(self, x1, x2):
        # DFM part
        ex1_1, ex1_2, ex1_3, ex1_4 = x1
        ex2_1, ex2_2, ex2_3, ex2_4 = x2
        df1 = self.diff1(ex1_1, ex2_1)
        df2 = self.diff2(ex1_2, ex2_2)
        df3 = self.diff3(ex1_3, ex2_3)
        df4 = self.diff4(ex1_4, ex2_4)
        
        # Decoder part
        decoder3 = self.up4(df4, df3)
        decoder2 = self.up3(decoder3, df2)
        decoder1 = self.up2(decoder2, df1)
        res = self.up1(decoder1)

        return res
