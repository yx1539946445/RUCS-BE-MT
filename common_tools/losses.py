import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.measure as measure
import gala.evaluate as gala_ev
import skimage
import cv2
from scipy.ndimage import distance_transform_edt as distance


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def kl_loss(inputs, targets, ep=1e-8):
    kl_loss = nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs + ep), targets)
    return consist_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs + ep)
    return torch.mean(-(target[:, 0, ...] * logprobs[:, 0, ...] + target[:, 1, ...] * logprobs[:, 1, ...]))


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


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

    mse_loss = F.mse_loss(input_softmax, target_softmax)
    return mse_loss


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
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
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
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
            loss += dice * weight[i]
        return loss / self.n_classes


from torch.autograd import Variable
import math


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


from PIL import Image
import numpy as np


def save(input, save_dir):
    input = input.softmax(dim=1).argmax(dim=1)
    image = input.squeeze(0)
    image = image.cpu().numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))  # *255
    image.save(save_dir)


def to_one(input):
    return torch.where(input != 0, torch.tensor(1).to(torch.device('cuda:0')), input).to(torch.device('cuda:0'))


class SE_ME_loss(nn.Module):
    def __init__(self, n_classes=2, ignore_index=-1, lambda_sep=0.5, lambda_merge=0.5, smooth=1e-09):
        super(SE_ME_loss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.lambda_sep = lambda_sep
        self.lambda_merge = lambda_merge
        self.smooth = 1e-09

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def separation_error_loss(self, predicted_probs, target_one_hot):
        # 计算分离错误损失
        vertical_diff = torch.abs(predicted_probs[:, :, 1:, :] - predicted_probs[:, :, :-1, :])
        horizontal_diff = torch.abs(predicted_probs[:, :, :, 1:] - predicted_probs[:, :, :, :-1])

        sep_error = torch.sum(vertical_diff * target_one_hot[:, :, 1:, :]) + torch.sum(
            horizontal_diff * target_one_hot[:, :, :, 1:])
        sep_error /= torch.sum(target_one_hot[:, :, 1:, :]) + torch.sum(target_one_hot[:, :, :, 1:] + self.smooth)

        return sep_error

    def merge_error_loss(self, predicted_probs, target_one_hot):
        # 计算合并错误损失
        vertical_sum = predicted_probs[:, :, 1:, :] + predicted_probs[:, :, :-1, :]
        horizontal_sum = predicted_probs[:, :, :, 1:] + predicted_probs[:, :, :, :-1]

        merge_error = torch.sum(torch.abs(vertical_sum - 1.0) * target_one_hot[:, :, 1:, :]) + torch.sum(
            torch.abs(horizontal_sum - 1.0) * target_one_hot[:, :, :, 1:])
        merge_error /= torch.sum(target_one_hot[:, :, 1:, :]) + torch.sum(target_one_hot[:, :, :, 1:] + self.smooth)

        return merge_error

    # def forward(self, pred, gt):
    #     loss = self.calculate_information_change(pred, gt)
    #     return loss
    def forward(self, predicted_logits, target_segmentation):
        # 将预测的 logits 转换为概率分布
        predicted_probs = F.softmax(predicted_logits, dim=1)

        # 将目标分割标签进行 one-hot 编码
        target_one_hot = F.one_hot(target_segmentation, num_classes=predicted_probs.shape[1]).permute(0, 3, 1,
                                                                                                      2).float()

        # 计算分离错误损失
        sep_loss = self.separation_error_loss(predicted_probs, target_one_hot)

        # 计算合并错误损失
        merge_loss = self.merge_error_loss(predicted_probs, target_one_hot)

        # 总的信息变化量损失
        total_loss = sep_loss + merge_loss

        return total_loss


class Sharp_aware_loss(nn.Module):
    def __init__(self, n_classes=2, ignore_index=-1):
        super(Sharp_aware_loss, self).__init__()
        self.n_classes = n_classes
        self.smooth = 1e-09
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def calculate_information_change(self, pred, gt, smooth=1e-09):
        # 将 logits 转换为概率分布
        probs = F.softmax(pred, dim=1)
        # 将真实标签转换为 one-hot 编码
        labels_one_hot = F.one_hot(gt, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        # 计算分离错误和合并错误
        # labels_one_hot = F.softmax(gt,dim=1)

        distances = labels_one_hot - pred

        true_entropy = -torch.sum(labels_one_hot * torch.log(labels_one_hot + smooth), dim=0)
        prob_entropy = -torch.sum(probs * torch.log(probs + smooth), dim=0)
        joint_entropy = -torch.sum(true_entropy * prob_entropy * torch.log(true_entropy * prob_entropy + smooth), dim=0)
        cross_entropy = -torch.sum(labels_one_hot * torch.log(probs + smooth), dim=0)
        # joint_entropy_ = -torch.sum(prob_entropy*torch.log(true_entropy+smooth),dim=0)
        # merging_error = (torch.abs(1 - labels_one_hot - probs))  # 计算每个样本的合并错误
        # merging_loss = torch.mean(merging_error)  # 计算平均合并错误
        # pred_dist = (labels_one_hot-intersection + smooth)
        # 返回 VI 损失
        loss = cross_entropy + torch.abs((true_entropy + prob_entropy - 2 * joint_entropy) * distances)
        # weights = loss * self.weights_aware(pred, gt)
        # split_term = split_term + split_term * weights
        # merge_term = merge_term + merge_term * weights
        # loss = loss + weights
        return loss.mean()

    def weights_aware(self, pred, gt, smooth=1e-09):
        b, c, h, w = pred.shape
        pred = pred.softmax(dim=1).argmax(dim=1)
        intersection = torch.sum(gt.unsqueeze(1) * pred.unsqueeze(1), dim=1)
        total = torch.sum(gt.unsqueeze(1) + pred.unsqueeze(1), dim=1)
        union = total - intersection
        weights = 1 - (((intersection + smooth)) / (union + smooth))
        return weights.view(b, 1, h, w)

    def sharp_aware(self, pred, gt, smooth=1e-09):
        b, c, h, w = pred.shape
        pred_erode = -(self.maxpool(-pred))
        pred_boudary = self.maxpool(pred - pred_erode)
        gt = gt.unsqueeze(1)
        gt_erode = -(self.maxpool(-gt.float()))
        gt_boudary = self.maxpool(gt - gt_erode)
        pred_boudary = pred_boudary.softmax(dim=1).argmax(dim=1)
        intersection = torch.sum(gt_boudary * pred_boudary.unsqueeze(1), dim=1)
        total = torch.sum(pred_boudary + pred_boudary.unsqueeze(1), dim=1)
        union = total - intersection
        weights = 1 - (((intersection + smooth)) / (union + smooth))
        return weights.view(b, 1, h, w)

    def certainty_map(self, map):
        map_softmax = torch.softmax(map, dim=1).argmax(dim=1)
        uncertainty = -(map_softmax * torch.log(map_softmax + self.smooth))
        certainty = 1 - uncertainty
        return certainty

    def forward(self, pred, gt):
        loss = self.calculate_information_change(pred, gt)
        return loss


class Uncertainty_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, epsilon=1e-5):
        super(Uncertainty_Attention, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, smooth=1e-09):
        b, c, h, w = x.shape
        sigmoid_x = torch.softmax(x, dim=1)
        sigmoid_y = torch.softmax(y, dim=1)
        uncertainty_x = -torch.sum(sigmoid_x * torch.log(sigmoid_x + smooth), dim=1)
        uncertainty_x = self.sigmoid(uncertainty_x)
        certainty_x = 1 - uncertainty_x
        uncertainty_y = -torch.sum(sigmoid_y * torch.log(sigmoid_y + smooth), dim=1)
        uncertainty_y = self.sigmoid(uncertainty_y)
        certainty_y = 1 - uncertainty_y
        sharp = x * certainty_y.view(b, 1, h, w) + y * certainty_x.view(b, 1, h, w)
        return sharp


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)

    def forward(self, pred, gt, smooth=1e-09):
        # 去除细小颗粒
        erode_gt = (-self.maxpool(-gt))
        dilate_gt = self.maxpool(erode_gt)
        filtered_gt = dilate_gt

        # 计算概率
        probs = torch.softmax(pred, dim=1)
        labels_one_hot = torch.softmax(filtered_gt, dim=1)

        # 计算伪标签准确性损失
        loss = F.mse_loss(probs, labels_one_hot)
        return loss


class to_boundary(nn.Module):
    def __init__(self):
        super(to_boundary, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def detect_edge(self, inputs, sobel_kernel, kernel_num):
        kernel = np.array(sobel_kernel, dtype='float32')
        kernel = kernel.reshape((1, 1, kernel_num, kernel_num))
        weight = Variable(torch.from_numpy(kernel)).to(torch.device('cuda:0'))
        edge = torch.zeros(inputs.size()[1], inputs.size()[0], inputs.size()[2], inputs.size()[3]).to(
            torch.device('cuda:0'))
        for k in range(inputs.size()[1]):
            fea_input = inputs[:, k, :, :]
            fea_input = fea_input.unsqueeze(1)
            edge_c = F.conv2d(fea_input, weight, padding=kernel_num // 2)
            edge[k] = edge_c.squeeze(1)
        edge = edge.permute(1, 0, 2, 3)
        return edge

    def sobel_conv2d(self, x):
        f, w, h = -0.01, -0.1, 1

        sobel_x = self.detect_edge(x, [
            [f, f, f, f, f],
            [f, w, w, w, f],
            [f, w, h, w, f],
            [f, w, w, w, f],
            [f, f, f, f, f],

        ], 5)

        sobel_y = self.detect_edge(x, [
            [w, w, w],
            [w, h, w],
            [w, w, w],
        ], 3)

        return sobel_x + sobel_y

    def forward(self, x):
        x = self.sobel_conv2d(x)
        # extend boundary map
        x = self.maxpool(x)
        x = torch.abs(x)
        return x


def _one_hot_encoder(input_tensor):
    tensor_list = []
    n_classes = 2
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, n_classes=2, ignore_index=-1):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def calculate_entropy(self, cluster):
        n = cluster.numel()
        unique, counts = torch.unique(cluster, return_counts=True)
        prob = counts.numel() / n
        prob = torch.tensor(prob)
        return -torch.sum(prob * torch.log(prob))

    def calculate_joint_entropy(self, cluster1, cluster2):
        n = cluster1.numel()
        unique1, counts1 = torch.unique(cluster1, return_counts=True)
        unique2, counts2 = torch.unique(cluster2, return_counts=True)
        joint_counts = torch.zeros((len(unique1), len(unique2)))
        for i in range(n):
            idx1 = torch.where(unique1, cluster1[i])[0][0]
            idx2 = torch.where(unique2, cluster2[i])[0][0]
            joint_counts[idx1, idx2] += 1
        joint_prob = joint_counts / n
        return -torch.sum(joint_prob * torch.log2(joint_prob))

    def calculate_variation_of_information(self, cluster1, cluster2):
        entropy1 = self.calculate_entropy(cluster1)
        entropy2 = self.calculate_entropy(cluster2)
        joint_entropy = self.calculate_joint_entropy(cluster1, cluster2)
        conditional_entropy1 = joint_entropy - entropy2
        conditional_entropy2 = joint_entropy - entropy1
        return conditional_entropy1 + conditional_entropy2

    def forward(self, pred, gt, smooth=1e-8):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        gt_ = gt
        gt = self._one_hot_encoder(gt.unsqueeze(dim=1))
        # softmax so that predicted map can be distributed in [0, 1]
        # one-hot vector of ground truth
        #  计算欧式距离
        pred = torch.sigmoid(pred)
        # distance = torch.sqrt(torch.sum((pred - gt) ** 2, dim=(1, 2)))
        # distance = torch.mean(distance) * 0.01
        # distance = torch.sigmoid(distance)
        # print("distance",distance)
        loss1 = self.loss_func(pred, gt_)
        # print("loss1",loss1.shape)
        # loss2 = distance
        # loss2 = se + me
        pred_gt_intersection = torch.sum((pred * gt), dim=(2, 3))
        pred_gt_total = torch.sum((pred + gt), dim=(2, 3))
        # union = pred_gt_total - pred_gt_intersection
        dice = (2 * (pred_gt_intersection + smooth)) / (pred_gt_total + smooth)
        loss2 = 1 - dice
        loss = loss1.mean() + loss2
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
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
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
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
            loss += dice * weight[i]
        return loss / self.n_classes


def get_vi_loss(pred, mask, bg_value: int = 0, method: int = 1):
    """
    Referenced by:
    Marina Meilă (2007), Comparing clusterings—an information based distance,
    Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
    :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
    :return Tuple = (VI, merger_error, split_error)
    """
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    vi, merger_error, split_error = 0.0, 0.0, 0.0

    label_pred, num_pred = measure.label(pred, connectivity=2, background=bg_value, return_num=True)
    label_mask, num_mask = measure.label(mask, connectivity=2, background=bg_value, return_num=True)
    if method == 0:
        # scikit-image
        split_error, merger_error = skimage.variation_of_information(label_mask, label_pred)
    elif method == 1:
        # gala
        merger_error, split_error = gala_ev.split_vi(label_pred, label_mask)
    vi = merger_error + split_error
    if math.isnan(vi):
        return 10, 5, 5
    return vi


import torch
import torch.nn as nn
import cv2, os
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
import numpy as np


def get_obj_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    获得前景（晶界）权重图,基于正态分布曲线在[-2.58*sigma, 2.58*sigma]处概率密度为99%
    因此每次求取最大值max_dis，反推sigma = max_dis / 2.58
    并根据Unet的原文计算Loss

    Obtain a foreground (grain boundary) weight map based on a normal distribution curve with a probability density of 99% at [-2.58*sigma, 2.58*sigma]
  So each time you get the maximum value max_dis, and then calculate sigma = max_dis / 2.58
  finally calculate Loss based on the original paper of U-Net
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow(dis_map, 2) / (2 * pow(std, 2)))
    return weight_matrix


def get_bck_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    获得背景（晶粒内部）权重图   Obtain background (inside grain) weight map
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow((max_dis - dis_map), 2) / (2 * pow(std, 2)))
    return weight_matrix


def caculate_weight_map(maskAddress, saveAddress='', weight_cof=30):
    """
    计算真值图对应的权重图  Calculate the weight map corresponding to the mask image
    :param maskAddress:  Address for mask image or np array
    :param saveAddress:  save directory
    :param weight_cof:  weight for class balance plus w0
    :return:  "adaptive_dis_weight" is the weight map for loss   "adaptive_bck_dis_weight_norm" is the weight map for last information
    """
    if isinstance(maskAddress, str):
        mask = cv2.imread(maskAddress, 0)
    else:
        mask = maskAddress
    labeled, label_num = label(mask, background=255, return_num=True, connectivity=1)
    image_props = regionprops(labeled, cache=False)
    dis_trf = ndimage.distance_transform_edt(255 - mask)
    adaptive_obj_dis_weight = np.zeros(mask.shape, dtype=np.float32)
    adaptive_obj_dis_weight = adaptive_obj_dis_weight + (mask / 255) * weight_cof
    adaptive_bck_dis_weight = np.ones(mask.shape, dtype=np.float32)

    for num in range(1, label_num + 1):
        image_prop = image_props[num - 1]
        bool_dis = np.zeros(image_prop.image.shape)
        bool_dis[image_prop.image] = 1.0
        (min_row, min_col, max_row, max_col) = image_prop.bbox
        temp_dis = dis_trf[min_row: max_row, min_col: max_col] * bool_dis

        adaptive_obj_dis_weight[min_row: max_row, min_col: max_col] = adaptive_obj_dis_weight[min_row: max_row,
                                                                      min_col: max_col] + get_obj_dis_weight(
            temp_dis) * bool_dis
        adaptive_bck_dis_weight[min_row: max_row, min_col: max_col] = adaptive_bck_dis_weight[min_row: max_row,
                                                                      min_col: max_col] + get_bck_dis_weight(
            temp_dis) * bool_dis

    # get weight map for loss
    adaptive_bck_dis_weight = adaptive_bck_dis_weight[:, :, np.newaxis]
    adaptive_obj_dis_weight = adaptive_obj_dis_weight[:, :, np.newaxis]
    adaptive_dis_weight = np.concatenate((adaptive_bck_dis_weight, adaptive_obj_dis_weight), axis=2)

    # np.save(os.path.join(saveAddress, "weight_map_loss.npy"), adaptive_dis_weight)

    # print("adaptive_obj_dis_weight range ", np.max(adaptive_obj_dis_weight), " ", np.min(adaptive_obj_dis_weight))
    # print("adaptive_bck_dis_weight range ", np.max(adaptive_bck_dis_weight), " ", np.min(adaptive_bck_dis_weight))

    # get weight for last information
    adaptive_bck_dis_weight = adaptive_bck_dis_weight[:, :, 0]
    bck_maxinum = np.max(adaptive_bck_dis_weight)
    bck_mininum = np.min(adaptive_bck_dis_weight)
    adaptive_bck_dis_weight_norm = (adaptive_bck_dis_weight - bck_mininum) / (bck_maxinum - bck_mininum)
    adaptive_bck_dis_weight_norm = (1 - adaptive_bck_dis_weight_norm) * (-7) + 1

    # np.save(os.path.join(saveAddress, "weight_map_last.npy"), adaptive_bck_dis_weight_norm)

    return adaptive_dis_weight, adaptive_bck_dis_weight_norm


class WeightMapLoss(nn.Module):
    """
    calculate weighted loss with weight maps in two channels
    """

    def __init__(self, _eps=1e-20, _dilate_cof=1):
        super(WeightMapLoss, self).__init__()
        self._eps = _eps
        # Dilate Coefficient of Mask
        self._dilate_cof = _dilate_cof
        # class balance weight, which is adjusted according to the dilate coefficient. The dilate coefficient can be 1, 3, 5, 7, 9 ....
        self._weight_cof = torch.Tensor([_dilate_cof, 20]).cuda()

    def _calculate_maps(self, mask, weight_maps, method):
        if -1 < method <= 6:  # WCE  LOSS
            weight_bck = torch.zeros_like(mask)
            weight_obj = torch.zeros_like(mask)
            if method == 1:  # class balance weighted loss
                weight_bck = (1 - mask) * self._weight_cof[0]
                weight_obj = mask * self._weight_cof[1]
            elif method == 2:  # 自适应膨胀 晶界加权（bck也加权） Adaptive weighted loss with dilated mask (bck is also weighted)
                weight_bck = (1 - mask) * weight_maps[:, 0, :, :]
                weight_obj = mask * weight_maps[:, 1, :, :]
            elif method == 3:  # 自适应膨胀 晶界加权（bck为1） Adaptive weighted loss with dilated mask (bck is set to 1)
                weight_bck = (1 - mask) * self._weight_cof[0]
                weight_obj = mask * weight_maps[:, 1, :, :]
            elif method == 4:  # 自适应对比晶界加权（bck也加权） Adaptive weighted loss described in our paper (bck is is also weighted)
                temp_weight_bck = weight_maps[:, 0, :, :]
                temp_weight_obj = weight_maps[:, 1, :, :]
                #                 print('WeightMapLoss mask ', mask.shape)
                #                 print('WeightMapLoss temp_weight_bck ', temp_weight_bck.shape)
                weight_obj[temp_weight_obj >= temp_weight_bck] = temp_weight_obj[temp_weight_obj >= temp_weight_bck]
                weight_obj = mask * weight_obj
                weight_bck[weight_obj <= temp_weight_bck] = temp_weight_bck[weight_obj <= temp_weight_bck]
            #                 print('WeightMapLoss weight_bck ', weight_bck.shape)

            elif method == 5:  # 自适应对比晶界加权（bck为1）  Adaptive weighted loss described in our paper (bck is set to 1)
                temp_weight_bck = weight_maps[:, 0, :, :]
                temp_weight_obj = weight_maps[:, 1, :, :]
                weight_obj[temp_weight_obj >= temp_weight_bck] = temp_weight_obj[temp_weight_obj >= temp_weight_bck]
                weight_obj = mask * weight_obj
                weight_bck[weight_obj <= temp_weight_bck] = 1
            return weight_bck, weight_obj
        elif method >= 7:  # MSE  LOSS
            weight_map = torch.zeros_like(mask)
            # MSE LOSS
            if method == 7:  # class banlance weighted loss
                weight_map = mask * (self._weight_cof[1] - self._weight_cof[0])
                weight_map = weight_map + self._weight_cof[0]
            elif method == 8:  # Adaptive weighted loss described in our paper
                weight_map = weight_map + mask * 30
                weight_map = weight_map + (1 - mask) * weight_maps[:, 0, :, :]
            weight_map[weight_map < 1] = 1
            return weight_map

    def forward(self, input, target, weight_maps, method=0):
        """
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：Select the type of loss function
        """
        mask = target.float()
        if -1 < method <= 6:  # WCE
            weight_bck, weight_obj = self._calculate_maps(mask, weight_maps, method)
            logit = torch.softmax(input, dim=1)
            logit = logit.clamp(self._eps, 1. - self._eps)
            loss = -1 * weight_bck * torch.log(logit[:, 0, :, :]) - weight_obj * torch.log(logit[:, 1, :, :])
            weight_sum = weight_bck + weight_obj
            return loss.sum() / weight_sum.sum()
        elif method >= 7:  # MSE
            weight_map = self._calculate_maps(mask, weight_maps, method)
            loss = weight_map * torch.pow(input - mask, 2)
            return torch.mean(loss.sum(dim=(1, 2)) / weight_map.sum(dim=(1, 2)))

    def show_weight(self, target, weight_maps, method=0):
        """
        For insurance purposes, visualize weight maps
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：Select the type of loss function
        """
        mask = target.float()
        if -1 < method <= 6:  # WCE
            weight_bck, weight_obj = self._calculate_maps(mask, weight_maps, method)
            return weight_bck, weight_obj
        elif method >= 7:  # MSE
            weight_map = self._calculate_maps(mask, weight_maps, method)
            return weight_map
