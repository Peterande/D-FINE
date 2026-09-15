#!/usr/bin/env python3
"""
Advanced Segmentation Head for DFINE
Maximum accuracy and robustness, suitable for high-end GPU setups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.base import SegmentationHeadInterface, ConvBlock, ASPP


class AttentionGate(nn.Module):
    """Attention gate for better feature selection"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SelfAttentionModule(nn.Module):
    """Self-attention module for long-range dependencies"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        return out


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class AdvancedASPP(nn.Module):
    """Advanced ASPP with attention and multi-scale context"""
    
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
        
        # Channel and spatial attention
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.attention = CBAM(total_channels)
        
        # Final projection
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
        
        # Combine features
        combined = torch.cat(features, dim=1)
        
        # Apply attention
        attended = self.attention(combined)
        
        return self.project(attended)


class RobustFPN(nn.Module):
    """Robust FPN with attention gates and multi-scale fusion"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # Lateral connections with batch normalization
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        # Output convolutions with attention
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                CBAM(out_channels)
            ) for _ in in_channels_list
        ])
        
        # Attention gates for better feature selection
        self.attention_gates = nn.ModuleList([
            AttentionGate(out_channels, out_channels, out_channels//2)
            for _ in range(len(in_channels_list)-1)
        ])
        
        # Multi-scale fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
        # Self-attention for global context
        self.self_attention = SelfAttentionModule(out_channels)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway with attention
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            
            # Apply attention gate
            att_feat = self.attention_gates[i-1](upsampled, laterals[i-1])
            laterals[i-1] = laterals[i-1] + att_feat
        
        # Apply final convs with attention
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Multi-scale weighted feature fusion
        target_size = outputs[0].shape[-2:]
        fused_features = []
        
        for i, feat in enumerate(outputs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            # Apply learnable weights
            weighted_feat = feat * torch.sigmoid(self.fusion_weights[i])
            fused_features.append(weighted_feat)
        
        # Combine with self-attention
        fused = sum(fused_features)
        attended = self.self_attention(fused)
        
        return attended


class AdvancedDecoder(nn.Module):
    """Advanced decoder with skip connections and attention"""
    
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Multi-stage decoder with progressive refinement
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            CBAM(in_channels),
            nn.Dropout2d(dropout_rate)
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(in_channels, in_channels//2, 3, 1, 1),
            CBAM(in_channels//2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(in_channels//2, in_channels//4, 3, 1, 1),
            CBAM(in_channels//4),
            nn.Dropout2d(dropout_rate)
        )
        
        # Final classification
        self.classifier = nn.Conv2d(in_channels//4, num_classes, 1)
        
        # Auxiliary outputs for deep supervision
        self.aux_classifier1 = nn.Conv2d(in_channels, num_classes, 1)
        self.aux_classifier2 = nn.Conv2d(in_channels//2, num_classes, 1)
    
    def forward(self, x):
        # Progressive decoding with auxiliary outputs
        feat1 = self.stage1(x)
        aux1 = self.aux_classifier1(feat1)
        
        feat2 = self.stage2(feat1)
        aux2 = self.aux_classifier2(feat2)
        
        feat3 = self.stage3(feat2)
        main_out = self.classifier(feat3)
        
        return main_out, aux1, aux2


class AdvancedSegmentationHead(SegmentationHeadInterface):
    """Advanced segmentation head with maximum accuracy and robustness
    
    Incorporates state-of-the-art attention mechanisms, multi-scale processing,
    and deep supervision for the highest possible segmentation quality.
    """
    
    def __init__(self, 
                 in_channels_list: List[int],
                 num_classes: int = 7,
                 feature_dim: int = 384,  # Increased for advanced model
                 dropout_rate: float = 0.15):  # Slightly higher for regularization
        """Initialize advanced segmentation head
        
        Args:
            in_channels_list: List of input channel counts from backbone
            num_classes: Number of segmentation classes (default: 7 for Pascal Person Parts)
            feature_dim: Feature dimension for processing (default: 384, increased for capacity)
            dropout_rate: Dropout rate (default: 0.15, higher for robustness)
        """
        super().__init__(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            tier='advanced'
        )
    
    def _build_head(self):
        """Build advanced segmentation head architecture"""
        
        # Robust FPN with attention mechanisms
        self.fpn = RobustFPN(self.in_channels_list, self.feature_dim)
        
        # Advanced ASPP with attention
        self.aspp = AdvancedASPP(self.feature_dim, self.feature_dim, dilations=[1, 6, 12, 18])
        
        # Advanced decoder with deep supervision
        self.decoder = AdvancedDecoder(self.feature_dim, self.num_classes, self.dropout_rate)
        
        # Boundary refinement branch
        self.boundary_head = nn.Sequential(
            ConvBlock(self.feature_dim, 64, 3, 1, 1),
            CBAM(64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        print(f"🏗️  Built advanced segmentation head:")
        print(f"   Feature dim: {self.feature_dim} (increased)")
        print(f"   Dropout rate: {self.dropout_rate}")
        print(f"   Classes: {self.num_classes}")
        print(f"   🎯 Features: Attention, multi-scale, deep supervision, boundary refinement")
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through advanced segmentation head
        
        Args:
            features: List of feature tensors from backbone [feat1, feat2, feat3, feat4]
            
        Returns:
            Tuple of (main_logits, aux1_logits, aux2_logits, boundary_map)
        """
        # Multi-scale feature fusion with attention
        fused_feature = self.fpn(features)
        
        # Apply advanced ASPP with attention
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation with deep supervision
        main_logits, aux1_logits, aux2_logits = self.decoder(context_feature)
        
        # Generate boundary map for auxiliary loss
        boundary_map = self.boundary_head(context_feature)
        
        return main_logits, aux1_logits, aux2_logits, boundary_map
    
    def forward_inference(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for inference (returns only main output)"""
        main_logits, _, _, _ = self.forward(features)
        return main_logits
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        base_info = super().get_model_info()
        
        # Add advanced-specific info
        base_info.update({
            'fpn_feature_dim': self.feature_dim,
            'aspp_dilations': [1, 6, 12, 18],
            'decoder_stages': 3,
            'uses_attention': True,
            'uses_self_attention': True,
            'uses_cbam': True,
            'deep_supervision': True,
            'boundary_refinement': True,
            'supports_multiscale': True,
            'optimized_for': 'maximum_accuracy',
            'target_platforms': ['high_end_gpu', 'server', 'workstation'],
            'estimated_accuracy': 'highest',
            'estimated_memory_mb': '200-400'
        })
        
        return base_info


def create_advanced_segmentation_head(in_channels_list: List[int], 
                                    num_classes: int = 7,
                                    feature_dim: int = 384,
                                    dropout_rate: float = 0.15) -> AdvancedSegmentationHead:
    """Factory function to create advanced segmentation head"""
    return AdvancedSegmentationHead(
        in_channels_list=in_channels_list,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )


# Export for use in other modules
__all__ = [
    'AdvancedSegmentationHead',
    'RobustFPN',
    'AdvancedASPP',
    'AdvancedDecoder',
    'AttentionGate',
    'SelfAttentionModule',
    'CBAM',
    'create_advanced_segmentation_head'
]