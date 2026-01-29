#!/usr/bin/env python3
"""
Fixed Enhanced D-FINE Segmentation Training Script
Stable multi-scale training without batch collation issues
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

# Add src to path
sys.path.append('src')

from src.core import YAMLConfig


class PascalPersonPartsDataset(torch.utils.data.Dataset):
    """Fixed Pascal Person Parts Dataset with stable multi-scale training"""
    
    def __init__(self, root_dir, split='train', image_size=640, multi_scale=False):
        self.root_dir = root_dir
        self.split = split
        self.base_image_size = image_size
        self.multi_scale = multi_scale
        
        # Get image paths
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'masks', split)
        
        self.image_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        print(f"📊 Loaded {len(self.image_names)} {split} samples")
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # Multi-scale options - but we'll resize everything to base_image_size for batching
        self.scales = [0.75, 0.85, 1.0, 1.15, 1.3] if multi_scale else [1.0]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image
        import cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Multi-scale training with random crop/resize
        if self.split == 'train' and self.multi_scale:
            # Choose random scale
            scale = random.choice(self.scales)
            scaled_size = int(self.base_image_size * scale)
            
            # Resize to scaled size first
            image = cv2.resize(image, (scaled_size, scaled_size))
            mask = cv2.resize(mask, (scaled_size, scaled_size), interpolation=cv2.INTER_NEAREST)
            
            # Random crop/pad to base_image_size for consistent batching
            if scaled_size > self.base_image_size:
                # Random crop
                start_x = random.randint(0, scaled_size - self.base_image_size)
                start_y = random.randint(0, scaled_size - self.base_image_size)
                image = image[start_y:start_y+self.base_image_size, start_x:start_x+self.base_image_size]
                mask = mask[start_y:start_y+self.base_image_size, start_x:start_x+self.base_image_size]
            elif scaled_size < self.base_image_size:
                # Center pad
                pad_x = (self.base_image_size - scaled_size) // 2
                pad_y = (self.base_image_size - scaled_size) // 2
                image = cv2.copyMakeBorder(
                    image, pad_y, self.base_image_size-scaled_size-pad_y,
                    pad_x, self.base_image_size-scaled_size-pad_x, cv2.BORDER_REFLECT
                )
                mask = cv2.copyMakeBorder(
                    mask, pad_y, self.base_image_size-scaled_size-pad_y,
                    pad_x, self.base_image_size-scaled_size-pad_x, cv2.BORDER_CONSTANT, value=0
                )
        else:
            # Standard resize for val or non-multi-scale
            image = cv2.resize(image, (self.base_image_size, self.base_image_size))
            mask = cv2.resize(mask, (self.base_image_size, self.base_image_size), interpolation=cv2.INTER_NEAREST)
        
        # Training augmentations
        if self.split == 'train':
            # Color augmentations
            if random.random() < 0.6:
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.randint(-20, 20)    # brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Horizontal flip
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # Gaussian blur (helps with distance)
            if random.random() < 0.3:
                kernel_size = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Random rotation (small angles)
            if random.random() < 0.4:
                angle = random.uniform(-10, 10)
                h, w = image.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        mask = torch.from_numpy(mask).long()
        
        # Now all tensors have consistent size for batching
        return image, mask


class SimplifiedASPP(nn.Module):
    """Simplified ASPP for better speed/accuracy balance"""
    
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12]):
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


class ImprovedFPN(nn.Module):
    """Improved FPN with better feature fusion"""
    
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
        
        # Feature fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
    def forward(self, features):
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
        
        # Return highest resolution feature with weighted fusion
        return sum(fused_features)


class EnhancedSegmentationHead(nn.Module):
    """Enhanced but stable segmentation head"""
    
    def __init__(self, in_channels_list, num_classes=7, feature_dim=256, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        print(f"🏗️  Creating enhanced segmentation head with channels: {in_channels_list}")
        
        # Improved FPN
        self.fpn = ImprovedFPN(in_channels_list, feature_dim)
        
        # Simplified ASPP for multi-scale context
        self.aspp = SimplifiedASPP(feature_dim, feature_dim)
        
        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(feature_dim//2, num_classes, 1)
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
        
        # Apply simplified ASPP
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation
        seg_logits = self.decoder(context_feature)
        
        return seg_logits


class DFineWithSegmentation(nn.Module):
    """D-FINE with enhanced segmentation"""
    
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
        seg_logits = self.seg_head(backbone_features)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs


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


class AdvancedSegmentationLoss(nn.Module):
    """Combined advanced loss for better segmentation"""
    
    def __init__(self, num_classes=7, focal_alpha=0.25, focal_gamma=2.0, ignore_index=255):
        super().__init__()
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, pred, target):
        # Focal loss (main loss)
        focal = self.focal_loss(pred, target)
        
        # Dice loss (boundary preservation)
        dice = self.dice_loss(pred, target)
        
        # Standard CE loss
        ce = F.cross_entropy(pred, target, ignore_index=255)
        
        # Combine losses
        total_loss = focal + 0.4 * dice + 0.2 * ce
        
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'ce': ce.item(),
            'total': total_loss.item()
        }


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


def create_enhanced_segmentation_model(config_path, checkpoint_path, num_classes=7):
    """Create enhanced segmentation model"""
    
    det_model = load_pretrained_dfine(config_path, checkpoint_path)
    backbone_channels = get_actual_backbone_channels(det_model)
    
    print(f"🏗️  Using backbone channels: {backbone_channels}")
    
    # Create enhanced segmentation head
    seg_head = EnhancedSegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=256
    )
    
    model = DFineWithSegmentation(
        dfine_model=det_model,
        seg_head=seg_head,
        freeze_detection=True
    )
    
    return model


def unfreeze_backbone_gradually(model, optimizer, epoch):
    """Gradually unfreeze backbone layers"""
    
    if epoch == 35:
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
        
    elif epoch == 45:
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
    """Enhanced training epoch"""
    model.train()
    
    total_loss = 0
    total_miou = 0
    loss_components = {'focal': 0, 'dice': 0, 'ce': 0, 'total': 0}
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        seg_pred = outputs['segmentation']
        
        # Calculate loss
        loss, loss_dict = criterion(seg_pred, masks)
        
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
            'DL': f'{loss_dict.get("dice", 0):.3f}'
        })
        
        # Log to wandb
        if batch_idx % 20 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_miou': miou.item(),
                'train/focal_loss': loss_dict.get('focal', 0),
                'train/dice_loss': loss_dict.get('dice', 0),
                'train/ce_loss': loss_dict.get('ce', 0),
                'train/lr': optimizer.param_groups[0]['lr']
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
    loss_components = {'focal': 0, 'dice': 0, 'ce': 0, 'total': 0}
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            seg_pred = outputs['segmentation']
            
            loss, loss_dict = criterion(seg_pred, masks)
            miou = compute_miou(seg_pred, masks)
            
            total_loss += loss.item()
            total_miou += miou.item()
            
            for key in loss_components:
                loss_components[key] += loss_dict.get(key, 0)
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_miou, avg_loss_components


def main():
    parser = argparse.ArgumentParser(description='Fixed Enhanced D-FINE Segmentation Training')
    parser.add_argument('--config', default='configs/dfine_hgnetv2_x_obj2coco.yml')
    parser.add_argument('--checkpoint', default='models/dfine_x_obj2coco.pth')
    parser.add_argument('--dataset', default='datasets/pascal_person_parts')
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced for stability
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=4)  # Reduced for stability
    parser.add_argument('--output_dir', default='outputs/dfine_segmentation_enhanced')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project='dfine-enhanced-segmentation-fixed',
        config=vars(args),
        name=f'fixed_enhanced_dfine_seg_{args.epochs}ep'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create enhanced model
    print("🏗️  Creating enhanced segmentation model...")
    model = create_enhanced_segmentation_model(
        args.config, args.checkpoint, num_classes=7
    )
    model = model.to(device)
    
    # Enhanced datasets with FIXED multi-scale training
    print("📊 Loading enhanced datasets...")
    train_dataset = PascalPersonPartsDataset(
        args.dataset, split='train', image_size=args.image_size, multi_scale=True
    )
    val_dataset = PascalPersonPartsDataset(
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
    
    # Advanced loss and optimizer
    criterion = AdvancedSegmentationLoss(num_classes=7)
    
    seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
    optimizer = optim.AdamW(seg_params, lr=args.lr, weight_decay=1e-4)
    
    # Advanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    print(f"🎯 Training {sum(p.numel() for p in seg_params):,} segmentation parameters")
    
    # Training loop
    best_miou = 0
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        
        # Gradual backbone unfreezing
        unfreeze_backbone_gradually(model, optimizer, epoch)
        
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
        
        # Log comprehensive metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/miou': train_miou,
            'train/focal_loss': train_loss_components['focal'],
            'train/dice_loss': train_loss_components['dice'],
            'train/ce_loss': train_loss_components['ce'],
            'val/loss': val_loss,
            'val/miou': val_miou,
            'val/focal_loss': val_loss_components['focal'],
            'val/dice_loss': val_loss_components['dice'],
            'val/ce_loss': val_loss_components['ce'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"📊 Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.3f}")
        print(f"📊 Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.3f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'config': vars(args)
            }, f'{args.output_dir}/best_model.pth')
            print(f"💾 Saved best model with mIoU: {best_miou:.3f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'miou': val_miou,
            }, f'{args.output_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    print(f"🎉 Training completed! Best mIoU: {best_miou:.3f}")
    wandb.finish()


if __name__ == '__main__':
    main()