# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


class SureRegionLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, y_pred, y_true, sure_mask):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        loss = loss[sure_mask].sum() / sure_mask.sum().clamp(min=1).float()
        return loss


class ContentLoss(nn.Module):
    def __init__(self, feature_layer=1) -> None:
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vgg = vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.net = vgg
        self.net.to(device)
        feature_layer_list = [29, 22, 15, 8, 3]
        feature_layer = feature_layer if feature_layer > 0 else 1
        feature_layer = feature_layer if feature_layer < 6 else 5
        self.feature_layer_list = feature_layer_list[:feature_layer]
        self.loss = nn.MSELoss()

    def forward(self, target_image, generate_image):
        content_loss = 0
        x = target_image.clone()
        y = generate_image.clone()
        layer_num = len(self.feature_layer_list)
        for i in range(len(self.net)):
            x = self.net[i](x.clone())
            y = self.net[i](y.clone())
            for i in self.feature_layer_list:
                content_loss += self.loss(x, y) / layer_num

        return content_loss
