"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch import optim
from torch.optim import lr_scheduler

from ..core import register

__all__ = ["SGD", "Adam", "AdamW", "CosineAnnealingLR", "LambdaLR", "MultiStepLR", "OneCycleLR"]


SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
