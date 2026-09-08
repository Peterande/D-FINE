#!/usr/bin/env python3
"""
Core Model Classes for DFINE Segmentation
Base classes for segmentation heads and model combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Add src to path for DFINE imports (src is at parent level)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up from core/ to project root
src_path = os.path.join(project_root, 'src')
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)


class BaseSegmentationHead(nn.Module, ABC):
    """Abstract base class for segmentation heads"""
    
    def __init__(self, 
                 in_channels_list: List[int],
                 num_classes: int = 7,
                 feature_dim: int = 256,
                 dropout_rate: float = 0.1,
                 tier: str = 'standard'):
        """Initialize base segmentation head
        
        Args:
            in_channels_list: List of input channel counts from backbone
            num_classes: Number of segmentation classes
            feature_dim: Feature dimension for processing
            dropout_rate: Dropout rate
            tier: Model tier ('lightweight', 'standard', 'advanced')
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.tier = tier
        
        # Build the segmentation head
        self._build_head()
        
        # Initialize weights
        self._init_weights()
        
        print(f"🏗️  Created {tier} segmentation head with channels: {in_channels_list}")
    
    @abstractmethod
    def _build_head(self):
        """Build the segmentation head architecture"""
        pass
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through segmentation head"""
        pass
    
    def _init_weights(self):
        """Initialize weights using standard initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'tier': self.tier,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        print(f"📊 {info['tier'].capitalize()} Segmentation Head Info:")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Feature dim: {info['feature_dim']}")
        print(f"   Dropout: {info['dropout_rate']}")
        print(f"   Total params: {info['total_params']:,}")
        print(f"   Model size: {info['model_size_mb']:.2f} MB")


class BaseCombinedModel(nn.Module, ABC):
    """Base class for DFINE + Segmentation combination"""
    
    def __init__(self, 
                 dfine_model: nn.Module,
                 seg_head: BaseSegmentationHead,
                 freeze_detection: bool = True):
        """Initialize combined model
        
        Args:
            dfine_model: Pretrained DFINE detection model
            seg_head: Segmentation head
            freeze_detection: Whether to freeze detection weights
        """
        super().__init__()
        
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
        
        if freeze_detection:
            self._freeze_detection()
    
    def _freeze_detection(self):
        """Freeze detection model parameters"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'seg_head' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        trainable_params = total_params - frozen_params
        
        print(f"🔒 Frozen detection components:")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   ❄️  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   🎯 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def unfreeze_detection(self):
        """Unfreeze detection model parameters"""
        unfrozen_count = 0
        for name, param in self.dfine_model.named_parameters():
            # Only unfreeze parameters that can require gradients (floating point and complex dtypes)
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = True
                unfrozen_count += 1
        
        self.freeze_detection = False
        print(f"🔓 Unfrozen detection components: {unfrozen_count} parameters")
    
    def forward(self, x: torch.Tensor, targets: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through combined model"""
        # Get backbone features (these will have gradients if backbone is unfrozen)
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch - always keep in inference mode for segmentation training
        # Even when backbone is unfrozen, we don't want to trigger detection training
        with torch.no_grad():
            # Temporarily set to eval mode to avoid denoising training
            was_training = self.dfine_model.training
            if was_training:
                self.dfine_model.eval()
            
            det_outputs = self.dfine_model(x)
            outputs.update(det_outputs)
            
            # Restore training mode
            if was_training:
                self.dfine_model.train()
        
        # Segmentation branch (uses backbone_features which may have gradients)
        seg_outputs = self.seg_head(backbone_features)
        
        # Handle different segmentation head output formats
        if isinstance(seg_outputs, tuple):
            # Advanced head returns (main_logits, aux1_logits, aux2_logits, boundary_map)
            seg_logits = seg_outputs[0]  # Use main output
            # Store auxiliary outputs for potential use in loss calculation
            if len(seg_outputs) >= 4:
                outputs['aux1_logits'] = seg_outputs[1]
                outputs['aux2_logits'] = seg_outputs[2] 
                outputs['boundary_map'] = seg_outputs[3]
        else:
            # Standard/lightweight heads return single tensor
            seg_logits = seg_outputs
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get combined model information"""
        det_params = sum(p.numel() for p in self.dfine_model.parameters())
        seg_params = sum(p.numel() for p in self.seg_head.parameters())
        total_params = det_params + seg_params
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        seg_info = self.seg_head.get_model_info()
        
        return {
            'tier': seg_info['tier'],
            'detection_params': det_params,
            'segmentation_params': seg_params,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'freeze_detection': self.freeze_detection,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


def get_actual_backbone_channels(model: nn.Module, input_size: Tuple[int, int] = (640, 640)) -> List[int]:
    """Get actual backbone output channels by running a forward pass"""
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size)
    
    with torch.no_grad():
        try:
            backbone_features = model.backbone(dummy_input)
            actual_channels = [feat.shape[1] for feat in backbone_features]
            return actual_channels
        except Exception as e:
            print(f"Warning: Could not determine backbone channels: {e}")
            # Return default channels for HGNetv2
            return [64, 128, 256, 512]


def load_pretrained_dfine(config_path: str, checkpoint_path: str) -> nn.Module:
    """Load pretrained DFINE model"""
    print(f"🚀 Loading pretrained DFINE from {checkpoint_path}")
    
    try:
        from src.core import YAMLConfig
        cfg = YAMLConfig(config_path)
        model = cfg.model
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict from various checkpoint formats
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained DFINE model")
        return model
    
    except Exception as e:
        print(f"❌ Failed to load DFINE model: {e}")
        raise


def create_combined_model(dfine_model: nn.Module, 
                         seg_head: BaseSegmentationHead,
                         freeze_detection: bool = True) -> BaseCombinedModel:
    """Create combined DFINE + Segmentation model"""
    
    class CombinedModel(BaseCombinedModel):
        """Concrete implementation of combined model"""
        pass
    
    return CombinedModel(dfine_model, seg_head, freeze_detection)


# Utility functions for model management
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def get_model_memory_usage(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> Dict[str, float]:
    """Estimate model memory usage"""
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * 4 / (1024 * 1024)  # MB, assuming float32
    
    # Estimate activation memory with a forward pass
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            # This is a rough estimate
            activation_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        except:
            activation_memory = 0
    
    return {
        'parameters_mb': param_memory,
        'estimated_activation_mb': activation_memory,
        'estimated_total_mb': param_memory + activation_memory
    }