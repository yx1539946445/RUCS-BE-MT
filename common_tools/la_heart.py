import h5py
import torch
import numpy as np
import itertools
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import cv2
from scipy import ndimage
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
import albumentations as A
from common_tools.losses import caculate_weight_map
from random import randint
import random
from torch import nn
import torch.nn.functional as F


def _gen_file_path(file_params, filename):
    return os.path.join(file_params[0], filename + '.' + file_params[1])


class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir + '/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]  # 108
        image_params = (os.path.join('../', 'data', "images"), 'png')
        label_params = (os.path.join('../', 'data', "labels"), 'png')
        # weight_params = {'image': _gen_file_path(image_params, image_name), 'label': _gen_file_path(label_params, image_name)}
        sample = {'weak_image': cv2.imread(_gen_file_path(image_params, image_name), cv2.IMREAD_GRAYSCALE),
                  'strong_image': cv2.imread(_gen_file_path(image_params, image_name), cv2.IMREAD_GRAYSCALE),
                  'label': cv2.imread(_gen_file_path(label_params, image_name), cv2.IMREAD_GRAYSCALE),
                  'weights': None,
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


def instance_to_mask(instance):
    '''
    Args:
        instance: PIL or numpy image
    '''
    instance = np.array(instance, dtype=np.uint32)
    # instance = 256 * (256 * instance[:, :, 0] + instance[:, :, 1]) + instance[:, :, 2]
    object_list = np.unique(instance[instance != 0])  # 去除重复的值
    current_num = 1
    for obj in object_list:
        instance[instance == obj] = current_num
        current_num += 1
    return instance.astype(np.uint8)


def instance_mask_to_binary_label(instance_mask):
    '''
    Args:
        instance_mask: PIL or numpy image
    '''
    instance_mask = np.array(instance_mask)
    instance_mask[instance_mask != 0] = 1
    return instance_mask


class InstanceToBinaryLabel(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        label = sample['label']
        label = torch.from_numpy(instance_mask_to_binary_label(instance_to_mask(label)))
        filter = sharp()
        return {'weak_image': sample['weak_image'], 'strong_image': sample['strong_image'], 'label': label,
                'weights': filter(label)}



def add_scratch_noise(image, mask):
    # # 二值化处理
    H, W = mask.shape
    if mask.sum() > 0:
        ret, thresh = cv2.threshold(mask, 0, 1, 0)
    else:
        ret, thresh = cv2.threshold(image, 0, 1, 0)
        # 寻找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓
    for contour in contours:
        mid_x, mid_y = int(W / 2), int(H / 2)
        x, y, w, h = cv2.boundingRect(contour)
        # 获取外接圆
        (_, _), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        gap_region = 10
        position = 1
        move_x = min(radius, int(w * (random.randint(position, 4) / 4)))
        move_y = min(radius, int(h * (random.randint(position, 4) / 4)))
        if x < mid_x and y > mid_y:
            # 左下
            new_x = x + move_x  # 在x方向上向右偏移3个像素
            new_y = y - move_y  # 在y方向上向下偏移10个像素
            if 0 <= x - gap_region <= W and 0 <= y + h <= H - gap_region:
                cell_image = image[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w + 2 * gap_region <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region,
                            new_x:new_x + w + 2 * gap_region].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_pred
            if x == 0 and y != 0:
                new_x = x  # 在x方向上向右偏移3个像素
                new_y = y - move_y  # 在y方向上向下偏移10个像素
                cell_image = image[y - gap_region:y + h + gap_region, x:x + w].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x:x + w].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_pred
            # 将细胞复制到相邻位置
        if x < mid_x and y < mid_y:
            # 左上
            # 将细胞复制到相邻位置
            new_x = x + move_x  # 在x方向上向右偏移3个像素
            new_y = y + move_y  # 在y方向上向下偏移10个像素
            if 0 <= x - gap_region <= W and 0 <= y - gap_region <= H:
                cell_image = image[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                if 0 <= new_y + h < H and 0 <= new_x + w <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region,
                            new_x:new_x + w + 2 * gap_region].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_pred
            else:
                new_x = x  # 在x方向上向右偏移3个像素
                new_y = y + move_y  # 在y方向上向下偏移10个像素
                cell_image = image[y - gap_region:y + h + gap_region, x:x + w].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x:x + w].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_pred
        if x > mid_x and y < mid_y:
            # 右上
            # 将细胞复制到相邻位置
            new_x = x - move_x  # 在x方向上向右偏移3个像素
            new_y = y + move_y  # 在y方向上向下偏移10个像素

            if 0 <= x + w <= W - gap_region and 0 <= y - gap_region <= H:
                cell_image = image[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w + 2 * gap_region <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region,
                            new_x:new_x + w + 2 * gap_region].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_pred
            else:
                new_x = x  # 在x方向上向右偏移3个像素
                new_y = y + move_y  # 在y方向上向下偏移10个像素
                cell_image = image[y - gap_region:y + h + gap_region, x:x + w].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x:x + w].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_pred
        if x > mid_x and y > mid_y:
            # 右下
            # 将细胞复制到相邻位置
            new_x = x - move_x  # 在x方向上向右偏移3个像素
            new_y = y - move_y  # 在y方向上向下偏移10个像素
            if 0 <= x + w <= W - gap_region and 0 <= y + h <= H - gap_region:
                cell_image = image[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x - gap_region:x + w + gap_region].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w + 2 * gap_region <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region,
                            new_x:new_x + w + 2 * gap_region].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w + 2 * gap_region] = cell_pred
            else:
                new_x = x  # 在x方向上向右偏移3个像素
                new_y = y - move_y  # 在y方向上向下偏移10个像素
                cell_image = image[y - gap_region:y + h + gap_region, x:x + w].copy()
                cell_pred = mask[y - gap_region:y + h + gap_region, x:x + w].copy()
                if 0 <= new_y + h + 2 * gap_region <= H and 0 <= new_x + w <= W and 0 <= new_y <= H and 0 <= new_x <= W and cell_pred.sum() != 0 \
                        and mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w].shape == cell_pred.shape:
                    image[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_image
                    mask[new_y:new_y + h + 2 * gap_region, new_x:new_x + w] = cell_pred
    return image, mask


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = np.array(img, dtype=np.float32)
    img -= mean
    img *= denominator
    return img


class Normalize():

    def __call__(self, sample):
        mean = torch.tensor([0.6306])
        std = torch.tensor([0.2603])
        weak_image = normalize(img=sample['weak_image'], mean=mean, std=std)
        strong_image = normalize(img=sample['strong_image'], mean=mean, std=std)
        return {'weak_image': torch.from_numpy(weak_image), 'strong_image': torch.from_numpy(strong_image),
                'label': sample['label']}


from PIL import Image


class sharp(nn.Module):
    def __init__(self):
        super(sharp, self).__init__()
        self.kernel_size = 5
        self.padding = 2
        self.zero = 0.0
        self.one = 1.0
        self.threshold = 0.7
        self.alpha = 1
        self.beta = 9

    def forward(self, x):

        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
        x = torch.where(x != 0, 1.0, 0.0)
        dilate = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        erode = (-F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        sharp = dilate - erode
        sharp = F.avg_pool2d(sharp, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        sharp = torch.where((sharp > self.threshold), (1.0/sharp), self.zero)
        sharp = torch.where(sharp > self.one, sharp * self.alpha, sharp * self.beta)
        sharp = torch.squeeze(torch.squeeze(sharp, dim=0), dim=0)
        return sharp

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        weak_image = sample['weak_image']
        weak_image = weak_image.reshape(1, weak_image.shape[0], weak_image.shape[1])
        strong_image = sample['strong_image']
        strong_image = strong_image.reshape(1, strong_image.shape[0], strong_image.shape[1])
        return {'weak_image': weak_image, 'strong_image': strong_image, 'label': sample['label'],
                'weights': sample['weights']}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def color_jitter(image):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class Augumentation():
    '''
    Data augumentation.
    '''

    def __init__(self):
        self.weak_trans = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.Flip(p=0.5),
            A.RandomCrop(width=256, height=256, p=1),
        ])
        self.strong_trans = A.Compose([
            A.ColorJitter(0.8, 0.8, 0.8, 0.2, p=1),
        ])

    def __call__(self, sample):
        # 有标签数据增强
        # sample['weak_image'], sample['label'] = add_scratch_noise(image=sample['weak_image'], mask=sample['label'])
        weak_trans = self.weak_trans(image=sample['weak_image'], mask=sample['label'])
        strong_trans = self.strong_trans(image=weak_trans['image'])
        return {'weak_image': weak_trans['image'], 'strong_image': strong_trans['image'], 'label': weak_trans['mask']}


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["weak_image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        filter = sharp()
        weights = filter(torch.from_numpy(label))
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"weak_image": image,"strong_image":image, "label": label,"weights":weights}
        return sample



class CenterCropForVisualize():
    def __init__(self):
        self.trans_resize = A.Resize(512,512)
        self.trans_centercrop = A.CenterCrop(448,448)

    def __call__(self, sample):
        weak_trans = self.trans_resize(image=sample['weak_image'], mask=sample['label'])
        weak_trans = self.trans_centercrop(image=weak_trans['image'], mask=weak_trans['mask'])
        filter = sharp()
        return {'weak_image': weak_trans['image'], 'strong_image': sample['strong_image'],'label': weak_trans['mask'],'weights': weak_trans['mask']}


if __name__ == '__main__':
    train_set = LAHeart('E:/data/LASet/data')
    print(len(train_set))
    # data = train_set[0]
    # image, label = data['image'], data['label']
    # print(image.shape, label.shape)
    labeled_idxs = list(range(25))
    unlabeled_idxs = list(range(25, 123))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 4, 2)
    i = 0
    for x in batch_sampler:
        i += 1
        print('%02d' % i, '\t', x)
