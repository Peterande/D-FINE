"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._config import BaseConfig
from .workspace import GLOBAL_CONFIG, create, register
from .yaml_config import YAMLConfig
from .yaml_utils import *
