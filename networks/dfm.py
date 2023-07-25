# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CorrelationFuseModule(nn.Module):

    def __init__(self, dim_in) -> None:
        super().__init__()

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, f1, f2):
        assert f1.shape == f2.shape
        B, C, H, W = f1.shape
        N = H * W

        cor_f1 = f1 - self.conv1(self.avgpool1(f1))  # (B, C, H, W)
        cor_f2 = f2 - self.conv1(self.avgpool1(f2))  # (B, C, H, W)
        cor_f1 = cor_f1.view(B, C, N).transpose(-1, -2)  # (B, N, C)
        cor_f2 = cor_f2.view(B, C, N).transpose(-1, -2)  # (B, N, C)

        cor = torch.sum(cor_f1 * cor_f2, dim=2, keepdim=True)  # (B, N, 1)
        cor = cor.transpose(-2, -1).contiguous().view(B, 1, H, W)

        correlation = torch.sigmoid(self.conv2(cor))
        out = torch.abs(f1 - f2) * (1 - correlation)
        return out
