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


class Resblock(nn.Module):
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
        super(Resblock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      bias=False),
        )
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      bias=False),
        )
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        short_cut = self.short_cut(x)
        return self.bn_relu(conv_2 + short_cut)


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


def channel_shuffle(x, groups=2):
    b, c, h, w = x.shape
    x = x.reshape(b, groups, -1, h, w)
    x = x.permute(0, 2, 1, 3, 4)
    # flatten
    x = x.reshape(b, -1, h, w)
    return x


class CAM(nn.Module):
    '''
    下采样模块
    '''

    def __init__(self, in_channels, class_num=2):
        super(CAM, self).__init__()
        self.class_num = class_num
        self.conv_1x1 = nn.Conv2d(in_channels, self.class_num, kernel_size=1, stride=1, padding=0)
        num_channels_reduced = in_channels // 2
        self.fc1 = nn.Linear(in_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, in_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        class_score = self.conv_1x1(x).softmax(dim=1).reshape(b, -1, self.class_num)
        reshaped_feature = x.reshape(b, c, -1)
        with autocast(enabled=False):
            channel_class_relevance = torch.bmm(reshaped_feature.float(), class_score.float())
            relevance_list = []
            for i in range(self.class_num):
                relevance_of_class_i = channel_class_relevance[:, :, i]
                scaled_num = relevance_of_class_i.abs().max(dim=1, keepdim=True).values / 2
                relevance_of_class_i = relevance_of_class_i / scaled_num
                relevance_of_class_i = self.relu(self.fc1(relevance_of_class_i))
                relevance_of_class_i = self.fc2(relevance_of_class_i)
                relevance_list.append(relevance_of_class_i)
            attention_weights = functools.reduce(lambda x, y: x + y, relevance_list)  # 列表累加
            attention_weights = self.sigmoid(attention_weights)
            x = x * attention_weights.view(b, c, 1, 1)
        return x


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class sharp_aware_block(nn.Module):
    def __init__(self, class_num=2):
        super(sharp_aware_block, self).__init__()
        self.epsilon = 1e-09
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_out = self.avg_pool(x)  # avg_out:[n,c*h*w]=>[n,c]
        max_out = self.max_pool(x)  # max_out:[n,c*h*w]=>[n,c]
        joint = torch.softmax(avg_out * max_out, dim=1)
        entorpy = -(joint * torch.log(joint + self.epsilon))
        certainty = 1 - torch.tanh(entorpy)
        x = x * certainty
        return x


def sharpening(p):
    T = 0.1
    p_sharpen = p ** T / (p ** T + (1 - p) ** T)
    return p_sharpen


class SAAM(nn.Module):
    def __init__(self):
        super(SAAM, self).__init__()
        self.epsilon = 1e-09

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out:[n,1,h,w]
        joint = avg_out
        K = joint.reshape(b, 1, -1).permute(0, 2, 1)
        Q = joint.reshape(b, 1, -1)
        V = x.reshape(b, c, -1)
        joint_soft = torch.softmax(torch.bmm(K, Q), dim=-1)
        outputs = torch.bmm(V, joint_soft.permute(0, 2, 1)).reshape(b, -1, h, w)
        return outputs

class BEM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BEM, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=3,dilation=3,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=5,dilation=5,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=7,dilation=7,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.short = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def boundary(self,x):
        erode = (-F.max_pool2d(-x, kernel_size=7, stride=1, padding=3))
        return x - erode
    def forward(self, x):
        eem = self.boundary(x)
        #short = self.short(x)
        conv_0 = self.conv_0(eem)
        conv_1 = self.conv_1(eem)
        conv_2 = self.conv_2(eem)
        conv_3 = self.conv_3(eem)
        output = torch.cat([conv_0, conv_1 , conv_2 ,conv_3],dim=1)
        avg_out = torch.mean(output, dim=1, keepdim=True)  # avg_out:[n,1,h,w]
        max_out = torch.max(output, dim=1, keepdim=True)[0]  # avg_out:[n,1,h,w]
        return torch.sigmoid(avg_out + max_out) * x + x


def sharp(self,x):
    short_cut_avg =  torch.mean(x, dim=1, keepdim=True)
    short_cut_max =  torch.max(x, dim=1, keepdim=True)[0]
    # dilate = (F.max_pool2d(short_cut, kernel_size=3, stride=1, padding=1))
    erode = (-F.max_pool2d(-short_cut_avg, kernel_size=3, stride=1, padding=1))
    sharp_avg = short_cut_avg - erode
    erode = (-F.max_pool2d(-short_cut_avg, kernel_size=3, stride=1, padding=1))
    sharp_max = short_cut_max - erode
    sharp = torch.sigmoid(sharp_avg + sharp_max)
    return sharp


class sharp_weight(nn.Module):
    def __init__(self):
        super(sharp_weight, self).__init__()
        self.kernel_size = 5
        self.padding = 2
        self.zero = 0.0
        self.one = 1.0
        self.threshold =  0.7
        self.eps = 1e-16
        self.alpha = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, x):
        #
        x = torch.sigmoid(x)
        erode_1 = (-F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        erode_2 = (-F.max_pool2d(-x, kernel_size=3, stride=1, padding=1))
        erode_3 = (-F.max_pool2d(-x, kernel_size=7, stride=1, padding=3))
        sharp = x - erode_1 + x - erode_2 + x - erode_3 + x
        sharp = sharp * x + sharp

        # x = torch.sigmoid(x)
        # dilate = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # erode = (-F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        # sharp = dilate - erode
        # sharp = F.avg_pool2d(sharp, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # sharp = torch.where((sharp > self.threshold), (1.0/(sharp + self.eps)), self.zero) + x
        # sharp = torch.where(( (self.one - 0.1)  <= sharp) & (sharp <= (self.one + 0.1) ), sharp * self.alpha, sharp * self.beta) + x
        return sharp


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        channels = [32, 64, 128, 256, 512]

        self.layer_1 = nn.Sequential(
            # Resblock(1, channels[0]),
            conv_3X3(1, channels[0]),
            conv_3X3(channels[0], channels[0]),
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Resblock(channels[0], channels[1]),
            conv_3X3(channels[0], channels[1]),
            conv_3X3(channels[1], channels[1]),
        )
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Resblock(channels[1], channels[2]),
            conv_3X3(channels[1], channels[2]),
            conv_3X3(channels[2], channels[2]),
        )
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Resblock(channels[2], channels[3]),
            conv_3X3(channels[2], channels[3]),
            conv_3X3(channels[3], channels[3]),
        )

        self.layer_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Resblock(channels[3], channels[4]),
            conv_3X3(channels[3], channels[4]),
            conv_3X3(channels[4], channels[4]),
            # SAAM(),
        )

        # self.res_path_1 = res_path()
        # self.res_path_2 = res_path()
        # self.res_path_3 = res_path()
        # self.res_path_4 = res_path()

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
            # Resblock(channels[4] + channels[3], channels[3]),
            conv_3X3(channels[4] + channels[3], channels[3]),
            conv_3X3(channels[3], channels[3]),
            # BEM(channels[3], channels[3]),

        )
        self.res_2 = nn.Sequential(
            # Resblock(channels[3] + channels[2], channels[2]),
            conv_3X3(channels[3] + channels[2], channels[2]),
            conv_3X3(channels[2], channels[2]),
            # BEM(channels[2], channels[2]),

        )
        self.res_3 = nn.Sequential(
            # Resblock(channels[2] + channels[1], channels[1]),
            conv_3X3(channels[2] + channels[1], channels[1]),
            conv_3X3(channels[1], channels[1]),
            # BEM(channels[1], channels[1]),

        )
        self.res_4 = nn.Sequential(
            # Resblock(channels[1] + channels[0], channels[0]),
            conv_3X3(channels[1] + channels[0], channels[0]),
            conv_3X3(channels[0], channels[0]),
            # BEM(channels[0], channels[0]),
        )

        self.classify_seg_conv_0 = nn.Conv2d(channels[0], class_num, kernel_size=1, stride=1, padding=0)

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


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_encoder = U_encoder()
        self.u_decoder = U_decoder()

    def forward(self, x):
        encoder_x_first, encoder_skips_first = self.u_encoder(x)
        outputs = self.u_decoder(encoder_x_first, encoder_skips_first)
        return outputs


def get_current_consistency_weight(epoch):
    consistency = 0.1
    consistency_rampup = 200.0
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class SingleConv1(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size=3, padding=1):
        super().__init__()
        self.Single_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Single_Conv(x)


if __name__ == '__main__':
    x = torch.randn(2, 1, 256, 256)
    model = Unet()
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    # 假设每个参数是一个 32 位浮点数（4 字节）
    bytes_per_param = 4

    # 计算总字节数
    total_bytes = total_trainable_params * bytes_per_param

    # 转换为兆字节（MB）和千字节（KB）
    total_megabytes = total_bytes / (1024 * 1024)
    total_kilobytes = total_bytes / 1024

    print("Total parameters in MB:", total_megabytes)
    print("Total parameters in KB:", total_kilobytes)
