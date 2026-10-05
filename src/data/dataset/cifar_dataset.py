"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from collections.abc import Callable
from typing import Optional

import torchvision

from ...core import register


@register()
class CIFAR10(torchvision.datasets.CIFAR10):
    __inject__ = ["transform", "target_transform"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
