"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch.cuda import amp

from ..core import register

__all__ = ["GradScaler"]

GradScaler = register()(amp.grad_scaler.GradScaler)
