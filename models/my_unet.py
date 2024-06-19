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
            # nn.Dropout2d(0.1),
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


def channel_shuffle(x, groups=4):
    b, c, h, w = x.shape
    x = x.reshape(b, groups, -1, h, w)
    x = x.permute(0, 2, 1, 3, 4)
    # flatten
    x = x.reshape(b, -1, h, w)
    return x



class space_aware_block(nn.Module):
    def __init__(self):
        super(space_aware_block, self).__init__()
        self.epsilon = 1e-09

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out:[n,1,h,w]
        # max_out = torch.max(x, dim=1, keepdim=True)[0]  # max_out:[n,1,h,w]
        joint = avg_out
        K = joint.reshape(b, 1, -1).permute(0, 2, 1)
        Q = joint.reshape(b, 1, -1)
        V = x.reshape(b, c, -1)
        joint = torch.softmax(torch.bmm(K, Q), dim=-1)
        return torch.bmm(V, joint.permute(0, 2, 1)).reshape(b, -1, h, w)

class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        channels = [8, 16, 32, 64, 128]

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
            space_aware_block(),
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
    def __init__(self):
        super(U_decoder, self).__init__()
        channels = [8, 16, 32, 64, 128]

        self.res_1 = nn.Sequential(
            conv_3X3(channels[4] + channels[3], channels[3]),

        )
        self.res_2 = nn.Sequential(
            conv_3X3(channels[3] + channels[2], channels[2]),

        )
        self.res_3 = nn.Sequential(
            conv_3X3(channels[2] + channels[1], channels[1]),
        )
        self.res_4 = nn.Sequential(
            conv_3X3(channels[1] + channels[0], channels[0]),
        )

        self.classify_seg_conv_0 = nn.Conv2d(channels[0], 2, kernel_size=1, stride=1, padding=0)

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


def create_model(is_teacher_model=False):
    # Network definition
    net = Unet()
    model = net.cuda()
    if is_teacher_model:
        for param in model.parameters():
            param.detach_()  # teacher 不参与反向传播 所以切断反向传播
    return model


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
        output = self.u_decoder(encoder_x_first, encoder_skips_first)
        return output







class my_unet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 extra_gap_weight: float,
                 learning_rate: float = 1.0e-3,
                 loss_func: Callable = nn.CrossEntropyLoss(reduction='none'),
                 total_iterations: int = 1000,
                 ):
        super(my_unet, self).__init__()
        self.student_model = create_model()
        self.teacher_model = create_model(is_teacher_model=True)

        self.extra_gap_weight = extra_gap_weight
        self.learning_rate = learning_rate

        self.loss_func = loss_func
        self.total_iterations = total_iterations
        self.gama = 0.5

    def forward(self, x):
        fetures = []
        output_student = self.student_model(x)

        output_student = self.student_model(x)
        return

    def training_step(self, batch, batch_idx):
        loss = 0.
        images, labels, state = myut.get_batch_data(batch, ('image', 'label', 'state'))
        b, c, h, w = images.shape
        images, labels = images.float(), labels.long()
        # inferer = myut.get_inferer(use_sliding_window=False)
        # student_preds = inferer(inputs=images, network=self.student_model)  # 原图像分割结果
        student_preds = self.student_model(images)
        for i in range(b):
            if state[i] == '0':
                loss_seg = self.loss_func(student_preds[i].unsqueeze(0), labels[i].unsqueeze(0))
                loss = loss + loss_seg
            else:
                # 一致性损失(一般是均方差)
                x = torch.clamp(torch.randn_like(images[i].unsqueeze(0)) * 0.1, -0.2, 0.2) + images[i].unsqueeze(0)
                with torch.no_grad():
                    teacher_preds = self.teacher_model(x)
                loss_consistency = softmax_mse_loss(student_preds[i].unsqueeze(0), teacher_preds)
                loss = loss + loss_consistency * get_current_consistency_weight(self.current_epoch)
        # print("loss_consistency * get_current_consistency_weight(self.current_epoch)", loss_.mean())
        self.log('train_loss', loss.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        loss = None
        images, labels, state = myut.get_batch_data(batch, ('image', 'label', 'state'))
        images, labels = images.float(), labels.long()
        # inferer = myut.get_inferer(use_sliding_window=True)  # use_sliding_window=True
        # preds = inferer(inputs=images, network=self.student_model)  # 原图像分割结果
        preds = self.student_model(images)
        loss = self.loss_func(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
            on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        initial_learning_rate = self.learning_rate
        current_iteration = self.trainer.global_step

        total_iteration = self.total_iterations
        for pg in optimizer.param_groups:
            pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
        optimizer.step(closure=optimizer_closure)
        update_ema_variables(self.student_model, self.teacher_model, alpha=0.99, global_step=current_iteration)

    def configure_optimizers(self):
        return myut.configure_optimizers(self.student_model, self.learning_rate)
