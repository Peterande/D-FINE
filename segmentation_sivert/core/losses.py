#!/usr/bin/env python3
"""
Fixed Core Loss Functions for DFINE Segmentation
Includes the original AdvancedSegmentationLoss functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for better boundary handling"""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        
        # Apply softmax to get probabilities
        inputs_soft = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets * valid_mask.long(), num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        inputs_soft = inputs_soft * valid_mask.unsqueeze(1).float()
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1).float()
        
        # Calculate dice coefficient
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class AdvancedSegmentationLoss(nn.Module):
    """Advanced loss combining multiple components for robust training - ORIGINAL VERSION"""
    
    def __init__(self, num_classes=7, focal_alpha=0.25, focal_gamma=2.0, ignore_index=255, 
                 dice_weight=0.4, ce_weight=0.2, boundary_weight=0.3, size_weight=0.1):
        super().__init__()
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.size_weight = size_weight
        self.ignore_index = ignore_index
    
    def size_sensitive_loss(self, pred, target):
        """Loss that gives more weight to smaller objects (distant people)"""
        # Calculate object sizes
        unique_labels = torch.unique(target)
        size_weights = torch.ones_like(target, dtype=torch.float32)
        
        for label in unique_labels:
            if label == self.ignore_index:
                continue
            
            mask = (target == label)
            size = mask.sum().float()
            
            # Inverse size weighting - smaller objects get higher weight
            if size > 0:
                weight = 1.0 / (torch.sqrt(size) + 1e-6)
                size_weights[mask] = weight
        
        # Apply size weights to cross entropy loss
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        weighted_ce = (ce_loss * size_weights).mean()
        
        return weighted_ce
    
    def forward(self, pred, target, boundary_pred=None):
        # Focal loss (main loss)
        focal = self.focal_loss(pred, target)
        
        # Dice loss (boundary preservation)
        dice = self.dice_loss(pred, target)
        
        # Standard CE loss
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
        
        # Size-sensitive loss for distant objects
        size_loss = self.size_sensitive_loss(pred, target)
        
        # Boundary loss if boundary prediction is provided
        boundary_loss_val = 0
        if boundary_pred is not None:
            # Create boundary target from segmentation mask
            boundary_target = self.create_boundary_target(target)
            boundary_loss_val = F.binary_cross_entropy(boundary_pred.squeeze(1), boundary_target)
        
        # Combine losses
        total_loss = (focal + 
                     self.dice_weight * dice + 
                     self.ce_weight * ce +
                     self.size_weight * size_loss +
                     self.boundary_weight * boundary_loss_val)
        
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'ce': ce.item(),
            'size': size_loss.item(),
            'boundary': boundary_loss_val.item() if isinstance(boundary_loss_val, torch.Tensor) else boundary_loss_val,
            'total': total_loss.item()
        }
    
    def create_boundary_target(self, target):
        """Create boundary target from segmentation mask"""
        # Use morphological operations to detect boundaries
        target_np = target.cpu().numpy()
        boundary_targets = []
        
        for i in range(target_np.shape[0]):
            mask = target_np[i]
            
            # Detect edges using morphological gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            boundary = (boundary > 0).astype(np.float32)
            
            boundary_targets.append(torch.from_numpy(boundary))
        
        return torch.stack(boundary_targets).to(target.device)


class BaseLoss(nn.Module):
    """Base loss class for different model tiers"""
    
    def __init__(self, 
                 num_classes: int = 7,
                 ignore_index: int = 255,
                 tier: str = 'standard'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tier = tier
        
        # Configure loss based on tier
        self._configure_loss()
    
    def _configure_loss(self):
        """Configure loss components based on tier"""
        if self.tier == 'lightweight':
            self._configure_lightweight_loss()
        elif self.tier == 'standard':
            self._configure_standard_loss()
        elif self.tier == 'advanced':
            self._configure_advanced_loss()
        else:
            self._configure_standard_loss()
    
    def _configure_lightweight_loss(self):
        """Configure lightweight loss (simple, fast)"""
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=self.ignore_index)
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        
        self.weights = {
            'focal': 1.0,
            'dice': 0.3,
            'ce': 0.1
        }
    
    def _configure_standard_loss(self):
        """Configure standard loss (enhanced to match original)"""
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=self.ignore_index)
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        
        # Use the same weights as the original AdvancedSegmentationLoss
        self.weights = {
            'focal': 1.0,
            'dice': 0.4,     # Same as original
            'ce': 0.2,       # Same as original
            'boundary': 0.3, # Same as original
            'size': 0.1      # Same as original
        }
    
    def _configure_advanced_loss(self):
        """Configure advanced loss (comprehensive)"""
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=self.ignore_index)
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        
        self.weights = {
            'focal': 1.0,
            'dice': 0.4,
            'ce': 0.2,
            'boundary': 0.3,
            'size': 0.1
        }
    
    def size_sensitive_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss that gives more weight to smaller objects (distant people)"""
        # Calculate object sizes
        unique_labels = torch.unique(target)
        size_weights = torch.ones_like(target, dtype=torch.float32)
        
        for label in unique_labels:
            if label == self.ignore_index:
                continue
            
            mask = (target == label)
            size = mask.sum().float()
            
            # Inverse size weighting - smaller objects get higher weight
            if size > 0:
                weight = 1.0 / (torch.sqrt(size) + 1e-6)
                size_weights[mask] = weight
        
        # Apply size weights to cross entropy loss
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        weighted_ce = (ce_loss * size_weights).mean()
        
        return weighted_ce
    
    def create_boundary_target(self, target: torch.Tensor) -> torch.Tensor:
        """Create boundary target from segmentation mask"""
        target_np = target.cpu().numpy()
        boundary_targets = []
        
        for i in range(target_np.shape[0]):
            mask = target_np[i]
            
            # Detect edges using morphological gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            boundary = (boundary > 0).astype(np.float32)
            
            boundary_targets.append(torch.from_numpy(boundary))
        
        return torch.stack(boundary_targets).to(target.device)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                boundary_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass of loss computation"""
        
        loss_dict = {}
        total_loss = 0.0
        
        # Focal loss (main loss)
        if 'focal' in self.weights:
            focal = self.focal_loss(pred, target)
            loss_dict['focal'] = focal.item()
            total_loss += self.weights['focal'] * focal
        
        # Dice loss
        if 'dice' in self.weights:
            dice = self.dice_loss(pred, target)
            loss_dict['dice'] = dice.item()
            total_loss += self.weights['dice'] * dice
        
        # Standard CE loss
        if 'ce' in self.weights:
            ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
            loss_dict['ce'] = ce.item()
            total_loss += self.weights['ce'] * ce
        
        # Size-sensitive loss
        if 'size' in self.weights:
            size_loss = self.size_sensitive_loss(pred, target)
            loss_dict['size'] = size_loss.item()
            total_loss += self.weights['size'] * size_loss
        
        # Boundary loss
        if 'boundary' in self.weights:
            if boundary_pred is not None:
                # Use provided boundary prediction
                boundary_target = self.create_boundary_target(target)
                boundary_loss_val = F.binary_cross_entropy(boundary_pred.squeeze(1), boundary_target)
            else:
                # No boundary prediction, skip boundary loss
                boundary_loss_val = torch.tensor(0.0, device=pred.device)
            
            loss_dict['boundary'] = boundary_loss_val.item()
            total_loss += self.weights['boundary'] * boundary_loss_val
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Factory function for creating losses
def create_loss(tier: str, num_classes: int = 7, ignore_index: int = 255, **kwargs) -> BaseLoss:
    """Factory function to create loss based on tier"""
    return BaseLoss(num_classes=num_classes, ignore_index=ignore_index, tier=tier)


# Keep the original AdvancedSegmentationLoss for backward compatibility
def create_advanced_loss(num_classes=7, **kwargs):
    """Create the original advanced loss"""
    return AdvancedSegmentationLoss(num_classes=num_classes, **kwargs)