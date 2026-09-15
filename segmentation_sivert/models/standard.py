#!/usr/bin/env python3
"""
Standard Segmentation Head for DFINE
Refactored from the original implementation with improved structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.base import SegmentationHeadInterface, ConvBlock, ASPP, create_decoder


class StandardFPN(nn.Module):
    """Standard FPN with feature fusion for balanced performance"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
        
        # Feature fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
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
        
        # Apply final convs
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Weighted feature fusion
        target_size = outputs[0].shape[-2:]
        fused_features = []
        
        for i, feat in enumerate(outputs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(feat * self.fusion_weights[i])
        
        # Return weighted sum of features
        return sum(fused_features)


class StandardASPP(nn.Module):
    """Standard ASPP for multi-scale context"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int] = [1, 6, 12]):
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


class StandardSegmentationHead(SegmentationHeadInterface):
    """Standard segmentation head with balanced performance and accuracy
    
    This is the refactored version of the original segmentation head,
    providing good performance with reasonable computational cost.
    """
    
    def __init__(self, 
                 in_channels_list: List[int],
                 num_classes: int = 7,
                 feature_dim: int = 256,
                 dropout_rate: float = 0.1):
        """Initialize standard segmentation head
        
        Args:
            in_channels_list: List of input channel counts from backbone
            num_classes: Number of segmentation classes (default: 7 for Pascal Person Parts)
            feature_dim: Feature dimension for processing (default: 256)
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            tier='standard'
        )
    
    def _build_head(self):
        """Build standard segmentation head architecture"""
        
        # Feature Pyramid Network for multi-scale feature fusion
        self.fpn = StandardFPN(self.in_channels_list, self.feature_dim)
        
        # Atrous Spatial Pyramid Pooling for multi-scale context
        self.aspp = StandardASPP(self.feature_dim, self.feature_dim)
        
        # Decoder for final segmentation prediction
        self.decoder = nn.Sequential(
            # First stage - maintain feature richness
            ConvBlock(self.feature_dim, self.feature_dim, 3, 1, 1),
            
            # Second stage - reduce dimensions
            ConvBlock(self.feature_dim, self.feature_dim//2, 3, 1, 1),
            
            # Dropout for regularization
            nn.Dropout2d(self.dropout_rate),
            
            # Final classification layer
            nn.Conv2d(self.feature_dim//2, self.num_classes, 1)
        )
        
        print(f"🏗️  Built standard segmentation head:")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Dropout rate: {self.dropout_rate}")
        print(f"   Classes: {self.num_classes}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through standard segmentation head
        
        Args:
            features: List of feature tensors from backbone [feat1, feat2, feat3, feat4]
            
        Returns:
            Segmentation logits [N, num_classes, H, W] where H, W are 1/4 of input size
        """
        # Multi-scale feature fusion with FPN
        fused_feature = self.fpn(features)
        
        # Apply ASPP for multi-scale context
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation logits
        seg_logits = self.decoder(context_feature)
        
        return seg_logits
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        base_info = super().get_model_info()
        
        # Add standard-specific info
        base_info.update({
            'fpn_feature_dim': self.feature_dim,
            'aspp_dilations': [1, 6, 12],
            'decoder_stages': 3,
            'supports_multiscale': True,
            'optimized_for': 'balanced_performance'
        })
        
        return base_info


# Backward compatibility - alias for the original name
EnhancedSegmentationHead = StandardSegmentationHead


def create_standard_segmentation_head(in_channels_list: List[int], 
                                    num_classes: int = 7,
                                    feature_dim: int = 256,
                                    dropout_rate: float = 0.1) -> StandardSegmentationHead:
    """Factory function to create standard segmentation head"""
    return StandardSegmentationHead(
        in_channels_list=in_channels_list,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )


# Export for use in other modules
__all__ = [
    'StandardSegmentationHead',
    'StandardFPN', 
    'StandardASPP',
    'EnhancedSegmentationHead',  # Backward compatibility
    'create_standard_segmentation_head'
]