'''
Author       : sphc
Description  : ---
Email        : jinkai0916@outlook.com
Date         : 2020-08-20 12:25:36
LastEditors  : sphc
LastEditTime : 2022-05-11 11:23:23
'''

import sys
import torch
import torchmetrics
import pandas as pd

sys.path.append("..")

# 避免除零
_SMOOTH = 1e-06


class ClassificationMetricsCalculator():
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def get_metrics(self, confusion_matrix) -> pd.Series:
        self.__current_confusion_matrix = confusion_matrix
        return self.info()

    def __pixel_accuracy(self) -> torch.Tensor:
        return self.__current_confusion_matrix.trace() / (self.__current_confusion_matrix.sum() + _SMOOTH)

    def __class_pixel_accuracy(self) -> torch.Tensor:
        return torch.diag(self.__current_confusion_matrix) / (self.__current_confusion_matrix.sum(axis=1) + _SMOOTH)

    def __mean_pixel_accuracy(self) -> torch.Tensor:
        return self.__class_pixel_accuracy().mean()

    def __adjusted_rand_score(self) -> torch.Tensor:
        return 2.0 * (self.__tp() * self.__tn() - self.__fn() * self.__fp()) / ((self.__tp() + self.__fn()) * (self.__fn() + self.__tn()) + (self.__tp() + self.__fp()) * (self.__fp() + self.__tn())+_SMOOTH)

    def __dice(self) -> torch.Tensor:
        if 2 != self.num_classes:
            raise RuntimeError('SegmentationMetrics.dice_index(): num_classes is not 2!')
        return (2 * self.__tp()) / (2 * self.__tp() + self.__fp() + self.__fn() + _SMOOTH)

    def __jaccard(self) -> torch.Tensor:
        if 2 != self.num_classes:
            raise RuntimeError('SegmentationMetrics.dice_index(): num_classes is not 2!')
        return self.__tp() / (self.__tp() + self.__fp() + self.__fn() + _SMOOTH)

    def __sensitivity(self) -> torch.Tensor:
        if 2 != self.num_classes:
            raise RuntimeError('SegmentationMetrics.dice_index(): num_classes is not 2!')
        return self.__tp() / (self.__tp() + self.__fn() + _SMOOTH)

    def __specificity(self) -> torch.Tensor:
        if 2 != self.num_classes:
            raise RuntimeError('SegmentationMetrics.dice_index(): num_classes is not 2!')
        return self.__tn() / (self.__fp() + self.__tn() + _SMOOTH)

    def info(self) -> pd.Series:
        d = {}
        # 类别数为2时才计算这些指标
        if self.num_classes == 2:
            d['Dice'] = self.__dice().item()
            d['Jaccard'] = self.__jaccard().item()
            d['Sensitivity'] = self.__sensitivity().item()
            d['Specificity'] = self.__specificity().item()

        d['Accuracy'] = self.__pixel_accuracy().item()
        # d['MPA'] = self.__mean_pixel_accuracy().item()
        # d['ARI'] = self.__adjusted_rand_score().item()
        return pd.Series(d)

    def __tp(self) -> torch.Tensor:
        return self.__current_confusion_matrix[1][1]

    def __tn(self) -> torch.Tensor:
        return self.__current_confusion_matrix[0][0]

    def __fp(self) -> torch.Tensor:
        return self.__current_confusion_matrix[0][1]

    def __fn(self) -> torch.Tensor:
        return self.__current_confusion_matrix[1][0]


class ObjectDice(torchmetrics.Metric):
    '''
        reference: https://github.com/WenYanger/Medical-Segmentation-Metrics
    '''

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "current_object_dice",
            default=torch.zeros(1),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total",
            default=torch.zeros(1),
            dist_reduce_fx="sum"
        )

    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        self.current_object_dice += self.__object_dice(pred, label)
        self.total += 1

    def compute(self) -> pd.Series:
        return self.info()

    def __dice(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        temp = A & B
        return 2 * torch.sum(temp, dtype=torch.float32) / \
               (torch.sum(A, dtype=torch.float32) + torch.sum(B, dtype=torch.float32))

    def __object_dice(self, S: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # 获取S中object列表及非0元素个数
        listLabelS = torch.unique(S)
        listLabelS = listLabelS[listLabelS != 0]
        numS = len(listLabelS)
        # 获取G中object列表及非0元素个数
        listLabelG = torch.unique(G)
        listLabelG = listLabelG[listLabelG != 0]
        numG = len(listLabelG)

        if numS == 0 & numG == 0:
            return 1
        elif numS == 0 | numG == 0:
            return 0

        # 记录omega_i*Dice(G_i,S_i)
        temp1 = 0.0
        # S中object总面积
        totalAreaS = torch.sum(S > 0, dtype=torch.float32)

        for iLabelS in range(numS):
            # Si为S中值为iLabelS的区域, boolean矩阵
            Si = S == listLabelS[iLabelS]
            # 找到G中对应区域并去除背景
            intersectlist = G[Si]
            intersectlist = intersectlist[intersectlist != 0]

            if len(intersectlist) != 0:
                indexGi = torch.argmax(torch.bincount(intersectlist))
                # Gi为gt中能与Si面积重合最大的object
                Gi = G == indexGi.item()
            else:
                Gi = torch.ones_like(G)
                Gi = Gi == 0

            omegai = torch.sum(Si, dtype=torch.float32) / totalAreaS
            temp1 = temp1 + omegai * self.__dice(Gi, Si)

        # 记录tilde_omega_i*Dice(tilde_G_i,tilde_S_i)
        temp2 = 0.0
        # G中object总面积
        totalAreaG = torch.sum(G > 0)

        for iLabelG in range(numG):
            # Si为S中值为iLabelS的区域, boolean矩阵
            tildeGi = G == listLabelG[iLabelG]
            # 找到G中对应区域并去除背景
            intersectlist = S[tildeGi]
            intersectlist = intersectlist[intersectlist != 0]

            if len(intersectlist) != 0:
                indextildeSi = torch.argmax(torch.bincount(intersectlist))
                tildeSi = S == indextildeSi.item()
            else:
                tildeSi = torch.ones_like(S)
                tildeSi = tildeSi == 0

            tildeOmegai = torch.sum(tildeGi, dtype=torch.float32) / totalAreaG
            temp2 = temp2 + tildeOmegai * self.__dice(tildeGi, tildeSi)

        objDice = (temp1 + temp2) / 2
        return objDice

    def info(self) -> pd.Series:
        d = {}
        d['ObjDice'] = (self.current_object_dice / (self.total + _SMOOTH)).item()
        return pd.Series(d)
