import torch
import pytorch_lightning as pl
from common_tools import ramps
import random
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable, Tuple, List


def cal_dice(output, target, eps=1e-3):
    output = torch.argmax(output, dim=1)
    inter = torch.sum(output * target) + eps
    union = torch.sum(output) + torch.sum(target) + eps * 2
    dice = 2 * inter / union
    return dice


def get_current_consistency_weight(epoch, consistency: float = 0.1, consistency_rampup: float = 200.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# 损失函数


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


class PAM(nn.Module):
    def __init__(self):
        super(PAM, self).__init__()
        self.epsilon = 1e-09

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out:[n,1,h,w]
        # max_out = torch.max(x, dim=1, keepdim=True)[0]  # max_out:[n,1,h,w]
        joint = avg_out
        K = joint.reshape(b, 1, -1).permute(0, 2, 1)
        Q = joint.reshape(b, 1, -1)
        V = x.reshape(b, c, -1)
        joint_soft = torch.softmax(torch.bmm(K, Q), dim=-1)
        outputs = torch.bmm(V, joint_soft.permute(0, 2, 1)).reshape(b, -1, h, w)
        return outputs + x


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


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_encoder = U_encoder()
        self.u_decoder = U_decoder()

    def forward(self, x):
        encoder_x_first, encoder_skips_first = self.u_encoder(x)
        outputs = self.u_decoder(encoder_x_first, encoder_skips_first)
        return outputs


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss = loss + dice * weight[i]
        return loss / self.n_classes


class mt_unet(pl.LightningModule):
    def __init__(self, lr: float = 0.001,
                 batch_size: int = 4,
                 labeled_bs: int = 2,
                 is_semi: bool = True,
                 max_epoch: int = 10000,
                 alpha: float = 0.99,
                 consistency: float = 0.1,
                 consistency_rampup: float = 200.0,
                 loss_func: Callable = nn.CrossEntropyLoss(reduction='none'),
                 ):
        super().__init__()
        self.semi_train = is_semi
        self.learning_rate = lr
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs
        self.glob_step = 0
        self.alpha = alpha
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        # networks
        self.model_stu = Unet()
        self.model_tea = Unet()

        for para in self.model_tea.parameters():
            para.detach_()

        self.dice_loss = DiceLoss()
        self.ce_loss = loss_func
        # self.mse_loss = mse_loss
        self.eval_dict = dict({"dice": []})

    def forward(self, x):
        return self.model_stu(x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        images, labels = batch['weak_image'], batch['label']
        images, labels = images.float(), labels.long()
        # train teacher network
        # if optimizer_idx == 0:
        outputs = self.model_stu(images)
        outputs_soft = torch.softmax(outputs, dim=1)
        consistency_loss = 0.0
        consistency_weight = get_current_consistency_weight(self.current_epoch, self.consistency,
                                                            self.consistency_rampup)
        if self.semi_train:
            unsup_images = images[self.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unsup_images) * 0.1, -0.2, 0.2)
            unsup_images = unsup_images + noise
            with torch.no_grad():
                ema_output = self.model_tea(unsup_images)

            ema_output_soft = torch.softmax(ema_output, dim=1)
            consistency_loss = torch.mean((outputs_soft[self.labeled_bs:] - ema_output_soft) ** 2)

        ce_loss = self.ce_loss(outputs[:self.labeled_bs], labels[:self.labeled_bs])
        dice_loss = self.dice_loss(outputs[:self.labeled_bs], labels[:self.labeled_bs].unsqueeze(1))
        loss_sup = 0.5 * (ce_loss + dice_loss)
        self.log('train_consistency_loss_loss', consistency_loss, on_step=False, on_epoch=True)
        self.log('train_ce_loss', ce_loss, on_step=False, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('train_sup_loss', loss_sup, on_step=False, on_epoch=True)
        return loss_sup + consistency_weight * consistency_loss

    def validation_step(self, batch, batch_idx):
        # if self.current_epoch > 150:
        self.model_stu.eval()
        images, labels = batch['weak_image'], batch['label']
        images, labels = images.float(), labels.long()
        with torch.no_grad():
            outputs = self.model_stu(images)
        self.eval_dict["dice"].append(cal_dice(outputs, labels))

    def on_validation_epoch_end(self):

        mean_dice = sum(self.eval_dict["dice"]) / 50
        self.log('val_mean_dice', mean_dice)
        self.eval_dict = dict({"dice": []})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model_stu.parameters(), lr=self.learning_rate, momentum=0.9,
                                    weight_decay=0.0001)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, param in zip(self.model_tea.parameters(), self.model_stu.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
