import numpy as np
import torch
from torch.nn import functional as F
from monai.networks.layers import Conv, ChannelPad
from torch import nn
from typing import Callable, Tuple, List
import pytorch_lightning as pl

'''
   BN 就是批量归一化

   RELU 就是激活函数

   lambda x:x 这个函数的意思是输出等于输入

   identity 就是残差

   1个resnet block 包含2个basic block
   1个resnet block 需要添加2个残差

   在resnet block之间残差形式是1*1conv，在resnet block内部残差形式是lambda x:x
   resnet block之间的残差用粗箭头表示，resnet block内部的残差用细箭头表示

   3*3conv s=2，p=1 特征图尺寸会缩小
   3*3conv s=1，p=1 特征图尺寸不变
'''

'''SA '''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        aux = x
        x = self.model.layer4(x)
        return aux, x


class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = nn.functional.interpolate(self.features(x), size[2:], mode='bilinear', align_corners=True)
        return output


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=2):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


# 构建一个FCN分割头，用于计算辅助损失
class Aux_Head(nn.Module):
    def __init__(self, in_channels=1024, num_classes=2):
        super(Aux_Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decode_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 2, self.in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 4, self.num_classes, kernel_size=3, padding=1),

        )

    def forward(self, x):
        return self.decode_head(x)


class my_pspnet(nn.Module):
    def __init__(self,
                 ):
        super(my_pspnet, self).__init__()
        self.fcn = base_resnet()

        self.aux_head = Aux_Head(in_channels=1024)
        self.cls_seg = nn.Sequential(
            PSPHEAD(in_channels=2048, out_channels=512, pool_sizes=[1, 2, 3, 6]),
            nn.Conv2d(512, 2, kernel_size=3, padding=1)
        )


    def forward(self, x):
        size = x.shape
        aux, encoder_x_first = self.fcn(x)

        if self.training:
            aux = self.aux_head(aux)
            aux = nn.functional.interpolate(aux, size[2:], mode='bilinear', align_corners=True)
            x = self.cls_seg(encoder_x_first)
            x = nn.functional.interpolate(x, size[2:], mode='bilinear', align_corners=True)
            return aux, x
        else:
            x = self.cls_seg(encoder_x_first)
            x = nn.functional.interpolate(x, size[2:], mode='bilinear', align_corners=True)
            return x
