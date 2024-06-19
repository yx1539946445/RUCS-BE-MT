import sys
import os
import torch
import torchmetrics
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../..")
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.vnet import VNet
from models.unet import Unet
from common_tools import ramps, losses
from common_tools.la_heart import *
import numpy as np
import torch.nn as nn
import common_tools.utils as myut
import random
from common_tools.losses import SSIM, BoundaryLoss, Sharp_aware_loss, WeightMapLoss, ConfidenceLoss, SE_ME_loss

loss_func = nn.CrossEntropyLoss(reduction='mean')



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='AL_SI', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/omnisky/postgraduate/Yb/data_set/LASet/data',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='unet', help='model_name')
parser.add_argument('--model', type=str, default='UAMT', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=40, help='trained samples')
parser.add_argument('--max_samples', type=int, default=400, help='all samples')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
args = parser.parse_args()

snapshot_path = "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

device = "cuda:2"
max_iterations = args.max_iterations
batch_size = args.batch_size
base_lr = args.base_lr
labeled_bs = args.labeled_bs


class DiceLoss(nn.Module):
    def __init__(self, n_classes = 2):
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
            loss += dice * weight[i]
        return loss / self.n_classes

def cal_dice(output, target, eps=1e-3):
    output = torch.argmax(output, dim=1)
    inter = torch.sum(output * target) + eps
    union = torch.sum(output) + torch.sum(target) + eps * 2
    dice = 2 * inter / union
    return dice


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True  #
        cudnn.deterministic = False  #
    else:
        cudnn.benchmark = False  # True #
        cudnn.deterministic = True  # False #

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = Unet()
        model = net
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model().to(device)
    ema_model = create_model(True).to(device)
    # mean = 0.6306
    # std = 0.2603
    train_data_path = "../data/txt_file"
    train_data = LAHeart(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             Augumentation(),
                             InstanceToBinaryLabel(),
                             # Normalize(),
                             ToTensor(),
                         ]))

    val_data = LAHeart(base_dir=train_data_path,
                       split='val',
                       transform=transforms.Compose([
                           InstanceToBinaryLabel(),
                           # Normalize(),
                           ToTensor()
                       ]))

    labeled_idxs = list(range(0,args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn)

    # optimizer = myut.configure_optimizers(model, base_lr)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # 结构性损失函数

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = base_lr
    val_result_list = []
    weights_num = 0.
    dice_loss = DiceLoss()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        model.train()
        time1 = time.time()
        train_epoch_loss = 0.
        train_num = 0
        for i_batch, sampled_batch in enumerate(train_loader):
            time2 = time.time()
            print('fetch data cost {}'.format(time2 - time1))
            volume_batch_weak, volume_batch_strong, label_batch, weights_batch = sampled_batch['weak_image'], \
                                                                                 sampled_batch['strong_image'], \
                                                                                 sampled_batch['label'], sampled_batch[
                                                                                     'weights'],

            volume_batch_weak, volume_batch_strong, label_batch, weights_batch = volume_batch_weak.float().to(
                device), volume_batch_strong.float().to(device), label_batch.long().to(device), weights_batch.long().to(
                device)

            unlabeled_volume_batch_weak = volume_batch_weak[labeled_bs:]
            unlabeled_volume_batch_strong = volume_batch_strong[labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch_weak) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch_weak + noise

            outputs = model(volume_batch_weak)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            ema_output_soft = torch.softmax(ema_output,dim=1)
            T = 8
            _, _, w, h = unlabeled_volume_batch_weak.shape
            volume_batch_r = unlabeled_volume_batch_weak.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, w, h]).to(device)
            for i in range(T//2):
                ema_inputs = volume_batch_r + \
                             torch.clamp(torch.randn_like(
                                 volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                                         (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 *  torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)

            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            # calculate the loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # 归一化
            loss_sup = 0.5 * (loss_func(outputs[:labeled_bs], label_batch[:labeled_bs]) + dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1)))
            # 伪标签
            loss_sup = loss_sup

            consistency_dist = torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            loss_unsup = consistency_weight * consistency_loss

            loss = loss_sup + loss_unsup
            train_epoch_loss = train_epoch_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            train_num = train_num + 1
            # writer.add_scalar('train_loss/train_loss', loss, iter_num)
            writer.add_scalar('lr/lr', lr_, iter_num)
            writer.add_scalar('info/train_step_total_loss', loss, iter_num)
            writer.add_scalar('info/train_step_sup_loss', loss_sup, iter_num)
            writer.add_scalar('info/train_step_unsup_loss', loss_unsup, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f  loss_sup: %f   loss_unsup: %f   ' %
                (iter_num, loss.mean(), loss_sup.mean(), loss_unsup))
            ## change lr
        writer.add_scalar('train_loss/train_loss', train_epoch_loss.mean() / train_num, iter_num)

        if iter_num > 0 and iter_num % 200 == 0:
            model.eval()
            with torch.no_grad():
                dice_sample = 0
                for sampled_batch in val_loader:
                    image, label = sampled_batch['weak_image'].float().to(device), sampled_batch['label'].float().to(device)
                    outputs = model(image)
                    dice_once = cal_dice(outputs, label)
                    dice_sample += dice_once
                dice_sample = dice_sample / len(val_loader)
                print('Average center dice:{:.3f}'.format(dice_sample))

            if dice_sample > best_dice:
                best_dice = dice_sample
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
            writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
            writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
            model.train()
        if iter_num >= max_iterations:
            break
        time1 = time.time()
    writer.close()
    print("finished")



