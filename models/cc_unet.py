import warnings
import numpy as np
import torch
import functools
from torch.nn import functional as F
from monai.networks.layers import Conv, ChannelPad
from torch import nn
from typing import Callable, Tuple, List
import pytorch_lightning as pl
import common_tools.utils as myut
from common_tools import ramps
from torch.cuda.amp import autocast
from torch.autograd import Variable

'''SA '''
import torch
from torch import nn
from common_tools.attentions import CoordAttention, SimAM, CBAM, PsAAttention, PiaxlAttention, NAM
import math
import cv2
from torch.nn import init


class conv_1X1(nn.Module):
    '''
    使用conv_1X1 改变维度通道数
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 strides: int = 1,
                 num_groups=32
                 ):
        super(conv_1X1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class conv_3X3(nn.Module):
    '''
    使用conv_3X3 改变维度通道数
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 num_groups=32,
                 dilation=1,
                 ):
        super(conv_3X3, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class conv_7X7(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups=32,
                 ):
        super(conv_7X7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        channels = [32, 64, 128, 256, 512]

        self.layer_1 = nn.Sequential(
            conv_3X3(1, channels[0]),
            conv_3X3(channels[0], channels[0]),
        )

        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_3X3(channels[0], channels[1]),
            conv_3X3(channels[1], channels[1]),
        )

        self.layer_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_3X3(channels[1], channels[2]),
            conv_3X3(channels[2], channels[2]),
        )

        self.layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_3X3(channels[2], channels[3]),
            conv_3X3(channels[3], channels[3]),
        )

        self.layer_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_3X3(channels[3], channels[4]),
            conv_3X3(channels[4], channels[4]),
            # PAM(),
        )

    def forward(self, x):
        features = []

        x = self.layer_1(x)
        features.append(x)  # (128, 128, 128)  32

        x = self.layer_2(x)
        features.append(x)

        x = self.layer_3(x)
        features.append(x)

        x = self.layer_4(x)
        features.append(x)

        x = self.layer_5(x)

        return x, features

class U_decoder(nn.Module):
    def __init__(self, class_num=2):
        super(U_decoder, self).__init__()
        channels = [32, 64, 128, 256, 512]

        self.res_1 = nn.Sequential(
            conv_3X3(channels[4] + channels[3], channels[3]),
            conv_3X3(channels[3], channels[3]),

        )
        self.res_2 = nn.Sequential(
            conv_3X3(channels[3] + channels[2], channels[2]),
            conv_3X3(channels[2], channels[2]),
        )

        self.res_3 = nn.Sequential(
            conv_3X3(channels[2] + channels[1], channels[1]),
            conv_3X3(channels[1], channels[1]),

        )
        self.res_4 = nn.Sequential(
            conv_3X3(channels[1] + channels[0], channels[0]),
            conv_3X3(channels[0], channels[0]),
        )

        self.classify_seg_conv_0 = nn.Conv2d(channels[0], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_1 = nn.Conv2d(channels[1], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_2 = nn.Conv2d(channels[2], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_3 = nn.Conv2d(channels[3], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, feature):

        _, _, h3, w3 = feature[3].shape  # 512
        _, _, h2, w2 = feature[2].shape  # 256
        _, _, h1, w1 = feature[1].shape  # 128
        _, _, h0, w0 = feature[0].shape  # 64
        x = F.interpolate(x, size=(h3, w3), mode='bilinear', align_corners=True)
        x = self.res_1(torch.cat([x, feature[3]], dim=1))

        x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)

        x = self.res_2(torch.cat([x, feature[2]], dim=1))

        x = F.interpolate(x, size=(h1, w1), mode='bilinear', align_corners=True)

        x = self.res_3(torch.cat([x, feature[1]], dim=1))

        x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=True)

        x = self.res_4(torch.cat([x, feature[0]], dim=1))
        output = self.classify_seg_conv_0(x)


        # number of spatial dimensions of the input image is 2
        return output

class U_decoder_f1(nn.Module):
    def __init__(self, class_num=2):
        super(U_decoder_f1, self).__init__()
        channels = [32, 64, 128, 256, 512]

        self.res_1 = nn.Sequential(
            conv_3X3(channels[4] , channels[3]),
            conv_3X3(channels[3], channels[3]),

        )
        self.res_2 = nn.Sequential(
            conv_3X3(channels[3] + channels[2], channels[2]),
            conv_3X3(channels[2], channels[2]),

        )
        self.res_3 = nn.Sequential(
            conv_3X3(channels[2] + channels[1], channels[1]),
            conv_3X3(channels[1], channels[1]),

        )
        self.res_4 = nn.Sequential(
            conv_3X3(channels[1] , channels[0]),
            conv_3X3(channels[0], channels[0]),
        )

        self.classify_seg_conv_0 = nn.Conv2d(channels[0], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_1 = nn.Conv2d(channels[1], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_2 = nn.Conv2d(channels[2], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_3 = nn.Conv2d(channels[3], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, feature):

        _, _, h3, w3 = feature[3].shape  # 512
        _, _, h2, w2 = feature[2].shape  # 256
        _, _, h1, w1 = feature[1].shape  # 128
        _, _, h0, w0 = feature[0].shape  # 64
        x = F.interpolate(x, size=(h3, w3), mode='bilinear', align_corners=True)
        x = self.res_1(x)

        x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)

        x = self.res_2(torch.cat([x, feature[2]], dim=1))

        x = F.interpolate(x, size=(h1, w1), mode='bilinear', align_corners=True)

        x = self.res_3(torch.cat([x, feature[1]], dim=1))

        x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=True)

        x = self.res_4(x)
        output = self.classify_seg_conv_0(x)


        # number of spatial dimensions of the input image is 2
        return output

class U_decoder_f2(nn.Module):
    def __init__(self, class_num=2):
        super(U_decoder_f2, self).__init__()
        channels = [32, 64, 128, 256, 512]

        self.res_1 = nn.Sequential(
            conv_3X3(channels[4] , channels[3]),
            conv_3X3(channels[3], channels[3]),

        )
        self.res_2 = nn.Sequential(
            conv_3X3(channels[3] + channels[2], channels[2]),
            conv_3X3(channels[2], channels[2]),

        )
        self.res_3 = nn.Sequential(
            conv_3X3(channels[2] + channels[1], channels[1]),
            conv_3X3(channels[1], channels[1]),

        )
        self.res_4 = nn.Sequential(
            conv_3X3(channels[1], channels[0]),
            conv_3X3(channels[0], channels[0]),
        )

        self.classify_seg_conv_0 = nn.Conv2d(channels[0], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_1 = nn.Conv2d(channels[1], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_2 = nn.Conv2d(channels[2], class_num, kernel_size=1, stride=1, padding=0)
        self.classify_seg_conv_3 = nn.Conv2d(channels[3], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, feature):
        feature_ = []
        _, _, h3, w3 = feature[3].shape  # 512
        _, _, h2, w2 = feature[2].shape  # 256
        _, _, h1, w1 = feature[1].shape  # 128
        _, _, h0, w0 = feature[0].shape  # 64
        x = F.interpolate(x, size=(h3, w3), mode='bilinear', align_corners=True)
        x = self.res_1(x)

        x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)

        x = self.res_2(torch.cat([x, feature[2]], dim=1))

        x = F.interpolate(x, size=(h1, w1), mode='bilinear', align_corners=True)

        x = self.res_3(torch.cat([x, feature[1]], dim=1))

        x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=True)
        x = self.res_4(x)
        output = self.classify_seg_conv_0(x)

        # number of spatial dimensions of the input image is 2
        return output


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_encoder = U_encoder()
        self.u_decoder = U_decoder()
        self.u_decoder_f1 = U_decoder_f1()
        self.u_decoder_f2 = U_decoder_f2()


    def forward(self, x):

        encoder_x_first, encoder_skips_first = self.u_encoder(x)
        outputs = self.u_decoder(encoder_x_first, encoder_skips_first)
        outputs_f1 = self.u_decoder_f1(encoder_x_first, encoder_skips_first)
        outputs_f2 = self.u_decoder_f2(encoder_x_first, encoder_skips_first)

        return outputs,outputs_f1,outputs_f2


if __name__ == '__main__':
    x = torch.randn(2, 1, 256, 256)
    model = Unet()
    y = model(x)
    print(y[0].shape)
