#!/usr/bin/env python3
"""
Enhanced D-FINE Segmentation Training Script with Superior Robustness
Optimized for distant objects, legs/feet segmentation, and overall robustness
Features advanced augmentations, multi-scale training, and progressive strategies
"""

import os
import sys
import yaml
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from pathlib import Path
import json
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import threading
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math

# Add src to path
sys.path.append('src')

from src.core import YAMLConfig

# Global variables to track best across trials
best_global_miou = 0.0
best_global_lock = threading.Lock()


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Set random seed to {seed}")


class AdvancedPascalPersonPartsDataset(torch.utils.data.Dataset):
    """Advanced Pascal Person Parts Dataset with robust augmentations for distant objects and lower body"""
    
    def __init__(self, root_dir, split='train', image_size=640, multi_scale=False, progressive_epoch=0):
        self.root_dir = root_dir
        self.split = split
        self.base_image_size = image_size
        self.multi_scale = multi_scale
        self.progressive_epoch = progressive_epoch
        
        # Get image paths
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'masks', split)
        
        self.image_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        print(f"📊 Loaded {len(self.image_names)} {split} samples")
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # Enhanced multi-scale options for better distant object handling
        self.scales = [0.5, 0.6, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5] if multi_scale else [1.0]
        
        # Progressive training: start smaller, go bigger
        self.current_size = self._get_progressive_size()
        
        # Advanced augmentations
        self.setup_augmentations()
    
    def _get_progressive_size(self):
        """Progressive resizing: start smaller for better distant object learning"""
        if self.split != 'train':
            return self.base_image_size
            
        # Progressive sizing over epochs
        min_size = int(self.base_image_size * 0.75)  # Start at 75% of target
        max_size = self.base_image_size
        
        # Gradually increase size over first 40 epochs
        progress = min(self.progressive_epoch / 40.0, 1.0)
        current_size = int(min_size + (max_size - min_size) * progress)
        
        # Ensure divisible by 32 for model compatibility
        return ((current_size + 31) // 32) * 32
    
    def setup_augmentations(self):
        """Setup advanced augmentations for robustness"""
        if self.split == 'train':
            self.spatial_transform = A.Compose([
                # Geometric augmentations for robustness
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT, p=0.7
                ),
                # Perspective transform for distant object simulation
                A.Perspective(scale=(0.05, 0.15), p=0.3),
                # Elastic transform for natural deformation
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50,
                    border_mode=cv2.BORDER_REFLECT, p=0.2
                ),
                # Grid distortion for robustness
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
                # Optical distortion
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
            ], additional_targets={'mask': 'mask'})
            
            self.color_transform = A.Compose([
                # Enhanced color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.ChannelShuffle(p=0.1),
                # Weather and lighting simulation
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1), 
                    num_shadows_lower=1, num_shadows_upper=2, p=0.3
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=6, num_flare_circles_upper=10, p=0.1
                ),
            ])
            
            self.noise_transform = A.Compose([
                # Noise and blur for distant object simulation
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.GaussianBlur(blur_limit=3, sigma_limit=0, p=0.1),
                # JPEG compression simulation
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            ])
        else:
            # No augmentations for validation
            self.spatial_transform = None
            self.color_transform = None
            self.noise_transform = None
    
    def __len__(self):
        return len(self.image_names)
    
    def apply_mixup_cutmix(self, image, mask, alpha=0.2):
        """Apply MixUp or CutMix for robustness"""
        if random.random() > 0.3:  # 30% chance
            return image, mask
            
        # Get another random sample
        idx2 = random.randint(0, len(self.image_names) - 1)
        img_name2 = self.image_names[idx2]
        mask_name2 = img_name2.replace('.jpg', '.png')
        
        img_path2 = os.path.join(self.img_dir, img_name2)
        mask_path2 = os.path.join(self.mask_dir, mask_name2)
        
        image2 = cv2.imread(img_path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
        
        # Resize to match current image
        h, w = image.shape[:2]
        image2 = cv2.resize(image2, (w, h))
        mask2 = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if random.random() < 0.5:  # MixUp
            lam = np.random.beta(alpha, alpha)
            mixed_image = (lam * image + (1 - lam) * image2).astype(np.uint8)
            # For masks, choose based on lambda threshold
            mixed_mask = np.where(np.random.random((h, w)) < lam, mask, mask2)
        else:  # CutMix
            lam = np.random.beta(alpha, alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
            
            mixed_image = image.copy()
            mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
            
            mixed_mask = mask.copy()
            mixed_mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]
        
        return mixed_image, mixed_mask
    
    def rand_bbox(self, W, H, lam):
        """Generate random bounding box for CutMix"""
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Enhanced multi-scale training with focus on distant objects
        if self.split == 'train' and self.multi_scale:
            # Bias towards smaller scales for better distant object learning
            scale_weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05]  # Favor smaller scales
            scale = np.random.choice(self.scales, p=scale_weights)
            scaled_size = int(self.current_size * scale)
            
            # Resize to scaled size first
            image = cv2.resize(image, (scaled_size, scaled_size))
            mask = cv2.resize(mask, (scaled_size, scaled_size), interpolation=cv2.INTER_NEAREST)
            
            # Smart crop/pad to current_size for consistent batching
            if scaled_size > self.current_size:
                # Random crop with bias towards center (where people usually are)
                margin = scaled_size - self.current_size
                # Center bias: 70% chance to crop near center
                if random.random() < 0.7:
                    start_x = random.randint(margin//4, 3*margin//4)
                    start_y = random.randint(margin//4, 3*margin//4)
                else:
                    start_x = random.randint(0, margin)
                    start_y = random.randint(0, margin)
                
                image = image[start_y:start_y+self.current_size, start_x:start_x+self.current_size]
                mask = mask[start_y:start_y+self.current_size, start_x:start_x+self.current_size]
            elif scaled_size < self.current_size:
                # Center pad with reflection
                pad_x = (self.current_size - scaled_size) // 2
                pad_y = (self.current_size - scaled_size) // 2
                image = cv2.copyMakeBorder(
                    image, pad_y, self.current_size-scaled_size-pad_y,
                    pad_x, self.current_size-scaled_size-pad_x, cv2.BORDER_REFLECT
                )
                mask = cv2.copyMakeBorder(
                    mask, pad_y, self.current_size-scaled_size-pad_y,
                    pad_x, self.current_size-scaled_size-pad_x, cv2.BORDER_CONSTANT, value=0
                )
        else:
            # Standard resize
            target_size = self.current_size if self.split == 'train' else self.base_image_size
            image = cv2.resize(image, (target_size, target_size))
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply advanced augmentations for training
        if self.split == 'train':
            # Spatial augmentations
            if self.spatial_transform:
                transformed = self.spatial_transform(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask']
            
            # MixUp/CutMix for robustness
            image, mask = self.apply_mixup_cutmix(image, mask)
            
            # Color augmentations
            if self.color_transform:
                image = self.color_transform(image=image)['image']
            
            # Noise and blur
            if self.noise_transform:
                image = self.noise_transform(image=image)['image']
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def update_progressive_epoch(self, epoch):
        """Update progressive training epoch"""
        self.progressive_epoch = epoch
        old_size = self.current_size
        self.current_size = self._get_progressive_size()
        if old_size != self.current_size:
            print(f"📏 Progressive resize: {old_size} -> {self.current_size}")


class AttentionGate(nn.Module):
    """Attention gate for better feature selection"""
    
    def __init__(self, F_g, F_l, F_int):
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


class EnhancedASPP(nn.Module):
    """Enhanced ASPP with attention and multi-scale context"""
    
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
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
        
        # Channel attention
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels//8, total_channels, 1),
            nn.Sigmoid()
        )
        
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
        
        # Apply channel attention
        att = self.channel_attention(combined)
        combined = combined * att
        
        return self.project(combined)


class RobustFPN(nn.Module):
    """Robust FPN with attention gates and multi-scale fusion"""
    
    def __init__(self, in_channels_list, out_channels=256):
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
        
        # Attention gates for better feature selection
        self.attention_gates = nn.ModuleList([
            AttentionGate(out_channels, out_channels, out_channels//2)
            for _ in range(len(in_channels_list)-1)
        ])
        
        # Multi-scale fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
    def forward(self, features):
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
        
        # Apply final convs
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
        
        return sum(fused_features)


class SuperiorSegmentationHead(nn.Module):
    """Superior segmentation head optimized for distant objects and lower body parts"""
    
    def __init__(self, in_channels_list, num_classes=7, feature_dim=256, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        print(f"🏗️  Creating superior segmentation head with channels: {in_channels_list}")
        
        # Robust FPN with attention
        self.fpn = RobustFPN(in_channels_list, feature_dim)
        
        # Enhanced ASPP for multi-scale context
        self.aspp = EnhancedASPP(feature_dim, feature_dim)
        
        # Multi-scale decoder for better detail preservation
        self.decoder = nn.Sequential(
            # First stage - preserve details
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            # Second stage - refine features
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            
            # Third stage - final refinement
            nn.Conv2d(feature_dim//2, feature_dim//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim//4),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(feature_dim//4, num_classes, 1)
        )
        
        # Boundary refinement branch
        self.boundary_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features):
        # Multi-scale feature fusion
        fused_feature = self.fpn(features)
        
        # Apply enhanced ASPP
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation
        seg_logits = self.decoder(context_feature)
        
        # Generate boundary map for auxiliary loss
        boundary_map = self.boundary_head(context_feature)
        
        return seg_logits, boundary_map


class DFineWithRobustSegmentation(nn.Module):
    """D-FINE with robust segmentation optimized for distant objects"""
    
    def __init__(self, dfine_model, seg_head, freeze_detection=True):
        super().__init__()
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
        
        if freeze_detection:
            self._freeze_detection()
    
    def _freeze_detection(self):
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
    
    def forward(self, x, targets=None):
        # Get backbone features
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch (frozen during training)
        if not self.training or not self.freeze_detection:
            with torch.no_grad() if self.freeze_detection else torch.enable_grad():
                det_outputs = self.dfine_model(x)
                outputs.update(det_outputs)
        
        # Segmentation branch
        seg_logits, boundary_map = self.seg_head(backbone_features)
        
        # Multi-scale inference for better distant object detection
        if not self.training:
            seg_logits = self.multi_scale_inference(x, seg_logits)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        boundary_map = F.interpolate(
            boundary_map, size=x.shape[-2:],
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        outputs['boundary'] = boundary_map
        
        return outputs
    
    def multi_scale_inference(self, x, base_logits):
        """Multi-scale test-time inference for better results"""
        scales = [0.75, 1.25]  # Additional scales for inference
        multi_scale_logits = [base_logits]
        
        for scale in scales:
            # Resize input
            h, w = x.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Forward pass
            with torch.no_grad():
                scaled_features = self.dfine_model.backbone(scaled_x)
                scaled_logits, _ = self.seg_head(scaled_features)
                
                # Resize back to original scale
                scaled_logits = F.interpolate(
                    scaled_logits, size=base_logits.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                multi_scale_logits.append(scaled_logits)
        
        # Average multi-scale predictions
        return torch.stack(multi_scale_logits).mean(dim=0)


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge preservation"""
    
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        """
        pred: [N, C, H, W] - predicted logits
        target: [N, H, W] - ground truth labels
        """
        # Get prediction probabilities
        pred_soft = torch.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate gradients for boundary detection
        def get_boundaries(tensor):
            # Sobel operator for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device)
            
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            grad_x = F.conv2d(tensor, sobel_x, padding=1)
            grad_y = F.conv2d(tensor, sobel_y, padding=1)
            
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate boundary regions
        pred_boundaries = get_boundaries(pred_soft.max(dim=1)[0].unsqueeze(1))
        target_boundaries = get_boundaries(target_one_hot.max(dim=1)[0].unsqueeze(1))
        
        # Boundary loss
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
        
        return boundary_loss


class AdvancedSegmentationLoss(nn.Module):
    """Advanced loss combining multiple components for robust training"""
    
    def __init__(self, num_classes=7, focal_alpha=0.25, focal_gamma=2.0, ignore_index=255, 
                 dice_weight=0.4, ce_weight=0.2, boundary_weight=0.3, size_weight=0.1):
        super().__init__()
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss()
        
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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for better boundary handling"""
    
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
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


def compute_miou(pred, target, num_classes=7):
    """Compute mean IoU"""
    pred = torch.argmax(pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            ious.append(intersection / union)
    
    return torch.tensor(ious).mean() if ious else torch.tensor(0.0)


def get_actual_backbone_channels(model):
    """Get actual backbone output channels"""
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)
    
    actual_channels = [feat.shape[1] for feat in backbone_features]
    return actual_channels


def load_pretrained_dfine(config_path, checkpoint_path):
    """Load pretrained D-FINE model"""
    print(f"🚀 Loading pretrained D-FINE from {checkpoint_path}")
    
    cfg = YAMLConfig(config_path)
    model = cfg.model
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded pretrained D-FINE model")
    return model


def create_robust_segmentation_model(config_path, checkpoint_path, num_classes=7, 
                                   feature_dim=256, dropout_rate=0.1):
    """Create robust segmentation model"""
    
    det_model = load_pretrained_dfine(config_path, checkpoint_path)
    backbone_channels = get_actual_backbone_channels(det_model)
    
    print(f"🏗️  Using backbone channels: {backbone_channels}")
    
    # Create superior segmentation head
    seg_head = SuperiorSegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )
    
    model = DFineWithRobustSegmentation(
        dfine_model=det_model,
        seg_head=seg_head,
        freeze_detection=True
    )
    
    return model


def unfreeze_backbone_gradually(model, optimizer, epoch, unfreeze_epoch1=35, unfreeze_epoch2=45):
    """Gradually unfreeze backbone layers"""
    
    if epoch == unfreeze_epoch1:
        print("🔓 Unfreezing backbone stage 4 (highest level features)")
        new_params = []
        for name, param in model.dfine_model.backbone.named_parameters():
            if 'stages.3' in name:  # Stage 4 of HGNetv2
                param.requires_grad = True
                new_params.append(param)
        
        if new_params:
            optimizer.add_param_group({
                'params': new_params,
                'lr': optimizer.param_groups[0]['lr'] * 0.1  # 10x lower LR
            })
            print(f"   Added {len(new_params)} parameters to optimizer")
        
    elif epoch == unfreeze_epoch2:
        print("🔓 Unfreezing backbone stage 3")
        new_params = []
        for name, param in model.dfine_model.backbone.named_parameters():
            if 'stages.2' in name:
                param.requires_grad = True
                new_params.append(param)
        
        if new_params:
            optimizer.add_param_group({
                'params': new_params,
                'lr': optimizer.param_groups[0]['lr'] * 0.1
            })
            print(f"   Added {len(new_params)} parameters to optimizer")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Enhanced training epoch with progressive training"""
    model.train()
    
    # Update progressive training
    if hasattr(dataloader.dataset, 'update_progressive_epoch'):
        dataloader.dataset.update_progressive_epoch(epoch)
    
    total_loss = 0
    total_miou = 0
    loss_components = {'focal': 0, 'dice': 0, 'ce': 0, 'size': 0, 'boundary': 0, 'total': 0}
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        seg_pred = outputs['segmentation']
        boundary_pred = outputs.get('boundary', None)
        
        # Calculate loss
        loss, loss_dict = criterion(seg_pred, masks, boundary_pred)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            miou = compute_miou(seg_pred, masks)
        
        total_loss += loss.item()
        total_miou += miou.item()
        
        # Accumulate loss components
        for key in loss_components:
            loss_components[key] += loss_dict.get(key, 0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'mIoU': f'{miou.item():.3f}',
            'FL': f'{loss_dict.get("focal", 0):.3f}',
            'DL': f'{loss_dict.get("dice", 0):.3f}',
            'BL': f'{loss_dict.get("boundary", 0):.3f}'
        })
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_miou, avg_loss_components


def validate_epoch(model, dataloader, criterion, device):
    """Enhanced validation"""
    model.eval()
    
    total_loss = 0
    total_miou = 0
    loss_components = {'focal': 0, 'dice': 0, 'ce': 0, 'size': 0, 'boundary': 0, 'total': 0}
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            seg_pred = outputs['segmentation']
            boundary_pred = outputs.get('boundary', None)
            
            loss, loss_dict = criterion(seg_pred, masks, boundary_pred)
            miou = compute_miou(seg_pred, masks)
            
            total_loss += loss.item()
            total_miou += miou.item()
            
            for key in loss_components:
                loss_components[key] += loss_dict.get(key, 0)
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_miou, avg_loss_components


def save_best_trial_results(output_dir, trial_number, trial_miou, trial_params, model, optimizer, scheduler, args, epoch):
    """Save best trial results and model"""
    global best_global_miou, best_global_lock
    
    with best_global_lock:
        if trial_miou > best_global_miou:
            best_global_miou = trial_miou
            
            print(f"🏆 New best trial found! Trial {trial_number} with mIoU: {trial_miou:.4f}")
            
            # Save best model
            model_save_path = f'{output_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'trial_number': trial_number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': trial_miou,
                'hyperparameters': trial_params,
                'config': vars(args),
                'seed': args.seed
            }, model_save_path)
            
            # Save best results JSON
            best_results = {
                'best_miou': trial_miou,
                'best_trial_number': trial_number,
                'best_hyperparameters': trial_params,
                'best_epoch': epoch,
                'seed': args.seed,
                'config': vars(args),
                'timestamp': str(torch.cuda.Event().query() if torch.cuda.is_available() else 'cpu')
            }
            
            results_save_path = f'{output_dir}/best_results.json'
            with open(results_save_path, 'w') as f:
                json.dump(best_results, f, indent=2)
            
            print(f"💾 Saved best model to: {model_save_path}")
            print(f"💾 Saved best results to: {results_save_path}")
            
            return True  # Indicates this was a new best
    
    return False  # Not a new best


def objective(trial, args, device, train_loader, val_loader, backbone_channels):
    """Optuna objective function for hyperparameter optimization"""
    
    # Set seed for each trial
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)
    
    # Enhanced hyperparameters for robust training
    hyperparams = {
        'lr': trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]),
        'batch_size': trial.suggest_categorical('batch_size', [4, 6, 8, 10, 12]),  # Smaller batches for stability
        'feature_dim': trial.suggest_categorical('feature_dim', [256, 320, 384, 448]),
        'dropout_rate': trial.suggest_categorical('dropout_rate', [0.05, 0.1, 0.15, 0.2]),
        'focal_alpha': trial.suggest_categorical('focal_alpha', [0.15, 0.2, 0.25, 0.3]),
        'focal_gamma': trial.suggest_categorical('focal_gamma', [1.5, 2.0, 2.5]),
        'dice_weight': trial.suggest_categorical('dice_weight', [0.3, 0.4, 0.5, 0.6]),
        'ce_weight': trial.suggest_categorical('ce_weight', [0.1, 0.15, 0.2]),
        'boundary_weight': trial.suggest_categorical('boundary_weight', [0.2, 0.3, 0.4]),
        'size_weight': trial.suggest_categorical('size_weight', [0.05, 0.1, 0.15, 0.2]),
        'weight_decay': trial.suggest_categorical('weight_decay', [1e-5, 5e-5, 1e-4, 2e-4]),
        'scheduler_t0': trial.suggest_categorical('scheduler_t0', [12, 15, 18, 20]),
        'unfreeze_epoch1': trial.suggest_categorical('unfreeze_epoch1', [25, 30, 35]),
        'unfreeze_epoch2': trial.suggest_categorical('unfreeze_epoch2', [35, 40, 45]),
    }
    
    # Ensure unfreeze_epoch2 > unfreeze_epoch1
    if hyperparams['unfreeze_epoch2'] <= hyperparams['unfreeze_epoch1']:
        hyperparams['unfreeze_epoch2'] = hyperparams['unfreeze_epoch1'] + 10
    
    print(f"\n🔬 Trial {trial.number} hyperparameters:")
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")
    
    # Initialize wandb for this trial
    wandb.init(
        project=f'{args.project_name}-robust',
        config={**vars(args), **hyperparams, 'trial_number': trial.number},
        name=f'robust_trial_{trial.number}',
        reinit=True
    )
    
    try:
        # Create model with trial hyperparameters
        model = create_robust_segmentation_model(
            args.config, args.checkpoint, 
            num_classes=7,
            feature_dim=hyperparams['feature_dim'],
            dropout_rate=hyperparams['dropout_rate']
        )
        model = model.to(device)
        
        # Create advanced loss function
        criterion = AdvancedSegmentationLoss(
            num_classes=7,
            focal_alpha=hyperparams['focal_alpha'],
            focal_gamma=hyperparams['focal_gamma'],
            dice_weight=hyperparams['dice_weight'],
            ce_weight=hyperparams['ce_weight'],
            boundary_weight=hyperparams['boundary_weight'],
            size_weight=hyperparams['size_weight']
        )
        
        # Create optimizer with trial hyperparameters
        seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            seg_params, 
            lr=hyperparams['lr'], 
            weight_decay=hyperparams['weight_decay']
        )
        
        # Create scheduler with trial hyperparameters
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams['scheduler_t0'], T_mult=2, eta_min=1e-6
        )
        
        best_trial_miou = 0
        best_trial_epoch = 0
        patience_counter = 0
        patience = 30  # Increased patience for robust training
        
        # Extended training for robust results
        max_epochs = max(args.epochs, 100)  # Minimum 100 epochs for robust training
        
        for epoch in range(max_epochs):
            # Gradual backbone unfreezing
            unfreeze_backbone_gradually(
                model, optimizer, epoch,
                hyperparams['unfreeze_epoch1'],
                hyperparams['unfreeze_epoch2']
            )
            
            # Train
            train_loss, train_miou, train_loss_components = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_miou, val_loss_components = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/miou': train_miou,
                'val/loss': val_loss,
                'val/miou': val_miou,
                'lr': optimizer.param_groups[0]['lr'],
                'trial_number': trial.number,
                'progressive_size': getattr(train_loader.dataset, 'current_size', args.image_size)
            })
            
            # Report to Optuna
            trial.report(val_miou, epoch)
            
            # Check if this is the best for this trial
            if val_miou > best_trial_miou:
                best_trial_miou = val_miou
                best_trial_epoch = epoch
                patience_counter = 0
                
                # Save if this is globally the best
                save_best_trial_results(
                    args.output_dir, trial.number, val_miou, hyperparams, 
                    model, optimizer, scheduler, args, epoch
                )
            else:
                patience_counter += 1
            
            # Prune unpromising trials
            if trial.should_prune():
                wandb.finish()
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
        
        print(f"🏁 Trial {trial.number} completed. Best mIoU: {best_trial_miou:.4f} at epoch {best_trial_epoch}")
        
        wandb.finish()
        return best_trial_miou
    
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        wandb.finish()
        raise e


def main():
    parser = argparse.ArgumentParser(description='Robust D-FINE Segmentation with Advanced Augmentations')
    parser.add_argument('--config', default='configs/dfine_hgnetv2_x_obj2coco.yml')
    parser.add_argument('--checkpoint', default='models/dfine_x_obj2coco.pth')
    parser.add_argument('--dataset', default='datasets/pascal_person_parts')
    parser.add_argument('--batch_size', type=int, default=32)  # Smaller for stability
    parser.add_argument('--epochs', type=int, default=250)  # More epochs for robust training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--output_dir', default='outputs/dfine_segmentation_robust')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials')
    parser.add_argument('--project_name', default='dfine-segmentation-robust', help='Wandb project name')
    parser.add_argument('--study_name', default='dfine_seg_robust_distant', help='Optuna study name')
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load datasets with advanced augmentations
    print("📊 Loading datasets with advanced augmentations...")
    train_dataset = AdvancedPascalPersonPartsDataset(
        args.dataset, split='train', image_size=args.image_size, multi_scale=True
    )
    val_dataset = AdvancedPascalPersonPartsDataset(
        args.dataset, split='val', image_size=args.image_size, multi_scale=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Get backbone channels for model creation
    temp_model = load_pretrained_dfine(args.config, args.checkpoint)
    backbone_channels = get_actual_backbone_channels(temp_model)
    del temp_model  # Free memory
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=f'sqlite:///{args.output_dir}/optuna_study.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    print(f"🔬 Starting robust optimization with {args.n_trials} trials")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Focus: Distant objects, legs/feet robustness, advanced augmentations")
    print(f"⏱️  Using progressive training and multi-scale inference")
    print(f"💾 Best model and results will be saved after each better trial")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, device, train_loader, val_loader, backbone_channels),
        n_trials=args.n_trials,
        timeout=None,
        gc_after_trial=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    print(f"\n🏆 Robust optimization completed!")
    print(f"📊 Best mIoU: {best_value:.4f}")
    print(f"🎯 Best trial number: {best_trial.number}")
    print(f"🎯 Best hyperparameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Save final optimization summary
    optimization_summary = {
        'best_miou': best_value,
        'best_trial_number': best_trial.number,
        'best_params': best_params,
        'total_trials': args.n_trials,
        'seed': args.seed,
        'optimization_config': vars(args),
        'features': [
            'Progressive training',
            'Advanced augmentations',
            'Multi-scale inference',
            'Size-sensitive loss',
            'Boundary-aware loss',
            'Attention mechanisms'
        ]
    }
    
    with open(f'{args.output_dir}/optimization_summary.json', 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    print(f"\n✅ Robust optimization summary saved to: {args.output_dir}/optimization_summary.json")
    print(f"🏆 Best model saved at: {args.output_dir}/best_model.pth")
    print(f"📊 Best results saved at: {args.output_dir}/best_results.json")


if __name__ == '__main__':
    main()