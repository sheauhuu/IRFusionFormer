import torch
from torch import Tensor
from typing import Tuple
from pycm import *


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.device = device
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred = pred.argmax(dim=1)
        # target = target.squeeze(1)
        # print(pred.shape, target.shape)
        keep = target != self.ignore_label
        # print(pred[keep].shape, target[keep].shape)
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
    
    def reset(self) -> None:
        self.hist.zero_()

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()]=0.
        miou = ious.mean().item()
        # miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()]=0.
        mf1 = f1.mean().item()
        # mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()]=0.
        macc = acc.mean().item()
        # macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def compute_metrics(self):
        conf_total = self.hist.cpu().numpy().astype(int)
        cm = ConfusionMatrix(matrix=conf_total)
        return {
            "Dice": cm.F1[1],
            "Specificity": cm.TNR[1],
            "Precision": cm.PPV[1],
            "Recall": cm.TPR[1],
            "Accuracy": cm.Overall_ACC,
            "Jaccard": cm.J[1]
        }