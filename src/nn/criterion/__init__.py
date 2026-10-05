"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch import nn

from ...core import register
from .det_criterion import DetCriterion

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
