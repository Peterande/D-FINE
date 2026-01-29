#!/usr/bin/env python3
"""
Loss/criterion factory for D-FINE pose estimation.

We re-use D-FINE's built-in matcher + criterion, and add the "keypoints" loss
that we implemented in `src/zoo/dfine/dfine_criterion.py`.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2  # used by segmentation boundary utilities (optional)
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


def create_pose_criterion(
    num_classes: int = 80,
    weight_dict: Dict[str, float] | None = None,
    losses: List[str] | None = None,
):
    """
    Create a DFINECriterion that includes keypoint losses.

    Notes:
    - Keypoints are supervised using the SAME Hungarian matching as the boxes (no extra matching).
    - Keypoints are bbox-relative: outputs["pred_keypoints"][..., :2] are in [0,1] within bbox.
    """
    from src.zoo.dfine.matcher import HungarianMatcher
    from src.zoo.dfine.dfine_criterion import DFINECriterion

    if weight_dict is None:
        weight_dict = {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_keypoints": 10.0,
            "loss_keypoints_vis": 1.0,
            "loss_pose_lqe": 1.0,
        }

    if losses is None:
        losses = ["vfl", "boxes", "local", "keypoints", "pose_lqe"]

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2, "cost_oks": 2.0},
        alpha=0.25,
        gamma=2.0,
    )
        
    return DFINECriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
        reg_max=32,
        boxes_weight_format=None,
        keypoints_box_mode="pred_detached",
        vfl_target="oks",
    )


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