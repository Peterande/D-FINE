#!/usr/bin/env python3
"""
Base Segmentation Head for DFINE
Abstract base class defining the interface for all segmentation heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.models import BaseSegmentationHead


class SegmentationHeadInterface(BaseSegmentationHead):
    """Interface for all segmentation heads with common functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def _build_head(self):
        """Build the specific segmentation head architecture"""
        pass
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through segmentation head
        
        Args:
            features: List of feature tensors from backbone
            
        Returns:
            Segmentation logits tensor [N, num_classes, H, W]
        """
        pass
    
    def get_complexity_info(self) -> Dict[str, Any]:
        """Get model complexity information"""
        info = self.get_model_info()
        
        # Estimate FLOPs (rough approximation)
        sample_input = [torch.randn(1, ch, 64, 64) for ch in self.in_channels_list]
        
        # Count convolutions
        conv_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_count += 1
        
        info.update({
            'conv_layers': conv_count,
            'estimated_flops': conv_count * 64 * 64 * 1000,  # Very rough estimate
            'memory_efficient': self.tier == 'lightweight'
        })
        
        return info


# Common building blocks for segmentation heads
class ConvBlock(nn.Module):
    """Standard convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False,
                 norm: bool = True, activation: bool = True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for lightweight models"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 
                                  groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class SimpleFPN(nn.Module):
    """Simple FPN for lightweight models"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply final convs and return highest resolution feature
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        return outputs[0]  # Return highest resolution


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int] = [1, 6, 12, 18]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 3, 
                               padding=dilation, dilation=dilation, bias=False)
            self.convs.append(nn.Sequential(
                conv, 
                nn.BatchNorm2d(out_channels//len(dilations)), 
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False),
            nn.BatchNorm2d(out_channels//len(dilations)),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        features = []
        
        # Apply dilated convolutions
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # Combine and project
        combined = torch.cat(features, dim=1)
        return self.project(combined)


class LightweightASPP(nn.Module):
    """Lightweight ASPP using depthwise separable convolutions"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int] = [1, 6, 12]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for dilation in dilations[:3]:  # Limit to 3 dilations for efficiency
            if dilation == 1:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False)
            else:
                conv = DepthwiseSeparableConv(in_channels, out_channels//len(dilations), 3, padding=dilation)
            self.convs.append(conv)
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        features = []
        
        # Apply dilated convolutions
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # Combine and project
        combined = torch.cat(features, dim=1)
        return self.project(combined)


def create_decoder(in_channels: int, num_classes: int, tier: str = 'standard', dropout_rate: float = 0.1) -> nn.Module:
    """Create decoder based on tier"""
    
    if tier == 'lightweight':
        return nn.Sequential(
            ConvBlock(in_channels, in_channels//2, 3, 1, 1),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels//2, num_classes, 1)
        )
    
    elif tier == 'standard':
        return nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels//2, 3, 1, 1),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels//2, num_classes, 1)
        )
    
    elif tier == 'advanced':
        return nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels//2, 3, 1, 1),
            ConvBlock(in_channels//2, in_channels//4, 3, 1, 1),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels//4, num_classes, 1)
        )
    
    else:
        raise ValueError(f"Unknown tier: {tier}")


# Export commonly used components
__all__ = [
    'SegmentationHeadInterface',
    'ConvBlock',
    'DepthwiseSeparableConv', 
    'SimpleFPN',
    'ASPP',
    'LightweightASPP',
    'create_decoder'
]