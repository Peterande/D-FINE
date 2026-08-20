#!/usr/bin/env python3
"""
Model tier implementations for DFINE Segmentation
Three specialized tiers: lightweight, standard, advanced
"""

# Import tier-specific models
from .lightweight import create_lightweight_segmentation_head, LightweightSegmentationHead
from .standard import create_standard_segmentation_head, StandardSegmentationHead
from .advanced import create_advanced_segmentation_head, AdvancedSegmentationHead

# Import base classes
from .base import SegmentationHeadInterface, ConvBlock, ASPP, LightweightASPP, SimpleFPN

__version__ = "1.0.0"
__all__ = [
    # Factory functions
    'create_lightweight_segmentation_head',
    'create_standard_segmentation_head', 
    'create_advanced_segmentation_head',
    
    # Model classes
    'LightweightSegmentationHead',
    'StandardSegmentationHead',
    'AdvancedSegmentationHead',
    
    # Base classes
    'SegmentationHeadInterface',
    'ConvBlock',
    'ASPP',
    'LightweightASPP',
    'SimpleFPN'
]

def create_segmentation_head(tier: str, in_channels_list, **kwargs):
    """Factory function to create segmentation head by tier"""
    
    if tier == 'lightweight':
        return create_lightweight_segmentation_head(in_channels_list, **kwargs)
    elif tier == 'standard':
        return create_standard_segmentation_head(in_channels_list, **kwargs)
    elif tier == 'advanced':
        return create_advanced_segmentation_head(in_channels_list, **kwargs)
    else:
        raise ValueError(f"Unknown tier: {tier}")