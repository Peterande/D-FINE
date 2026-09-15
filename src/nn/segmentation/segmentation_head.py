"""
Body Parts Segmentation Head for D-FINE
Designed for HGNetv2 backbone features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.common import ConvNormLayer, get_activation
from ...core import register

__all__ = ["BodyPartsSegmentationHead"]


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of features [P2, P3, P4, P5] from HGNetv2
        """
        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply final convs
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        return outputs


@register()
class BodyPartsSegmentationHead(nn.Module):
    """
    Body Parts Segmentation Head for D-FINE
    6 body parts: head, torso, arms, hands, legs, feet
    """
    
    def __init__(
        self, 
        in_channels_list,  # HGNetv2 output channels [128, 256, 512, 1024] for example
        num_classes=7,     # 6 body parts + background
        feature_dim=256,
        dropout_rate=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        
        print(f"🏗️  Creating segmentation head with channels: {in_channels_list}")
        
        # Feature Pyramid Network
        self.fpn = FPN(in_channels_list, feature_dim)
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            ConvNormLayer(feature_dim, feature_dim, 3, 1, act='relu'),
            ConvNormLayer(feature_dim, feature_dim//2, 3, 1, act='relu'),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(feature_dim//2, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from HGNetv2 backbone
        Returns:
            Segmentation logits
        """
        # Use FPN to fuse multi-scale features
        fpn_features = self.fpn(features)
        
        # Use the highest resolution feature (P2) for final prediction
        final_feature = fpn_features[0]
        
        # Generate segmentation prediction
        seg_logits = self.decoder(final_feature)
        
        return seg_logits