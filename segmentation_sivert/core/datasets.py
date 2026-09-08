#!/usr/bin/env python3
"""
Core Dataset Classes for DFINE Segmentation
Base classes for dataset loading with tier-specific augmentations
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any


class BaseSegmentationDataset(Dataset, ABC):
    """Abstract base class for segmentation datasets"""
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 image_size: int = 640,
                 num_classes: int = 7,
                 tier: str = 'standard'):
        """Initialize base dataset
        
        Args:
            root_dir: Root directory of dataset
            split: Dataset split ('train' or 'val')
            image_size: Target image size
            num_classes: Number of segmentation classes
            tier: Model tier ('lightweight', 'standard', 'advanced')
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.tier = tier
        
        # Paths
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'masks', split)
        
        # Load image list
        self.image_names = self._load_image_list()
        
        # Normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # Setup augmentations based on tier
        self.setup_augmentations()
        
        print(f"📊 Loaded {len(self.image_names)} {split} samples for {tier} tier")
    
    @abstractmethod
    def _load_image_list(self) -> List[str]:
        """Load list of image filenames"""
        pass
    
    @abstractmethod
    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask for given index"""
        pass
    
    def setup_augmentations(self):
        """Setup augmentations based on tier"""
        if self.split != 'train':
            self.augmentations = None
            return
            
        if self.tier == 'lightweight':
            self.augmentations = self._get_lightweight_augmentations()
        elif self.tier == 'standard':
            self.augmentations = self._get_standard_augmentations()
        elif self.tier == 'advanced':
            self.augmentations = self._get_advanced_augmentations()
        else:
            self.augmentations = self._get_standard_augmentations()
    
    def _get_lightweight_augmentations(self):
        """Lightweight augmentations for fast training"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        ], additional_targets={'mask': 'mask'})
    
    def _get_standard_augmentations(self):
        """Standard augmentations for balanced training"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        ], additional_targets={'mask': 'mask'})
    
    def _get_advanced_augmentations(self):
        """Advanced augmentations for robust training"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            A.Perspective(scale=(0.05, 0.15), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        ], additional_targets={'mask': 'mask'})
    
    def apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations to image and mask"""
        if self.augmentations is None:
            return image, mask
            
        transformed = self.augmentations(image=image, mask=mask)
        return transformed['image'], transformed['mask']
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor and normalize"""
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        return image
    
    def preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Convert mask to tensor"""
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask).long()
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        # Load image and mask
        image, mask = self._load_image_and_mask(idx)
        
        # Apply augmentations if training
        if self.split == 'train':
            image, mask = self.apply_augmentations(image, mask)
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        mask_tensor = self.preprocess_mask(mask)
        
        return image_tensor, mask_tensor


class PascalPersonPartsDataset(BaseSegmentationDataset):
    """Pascal Person Parts dataset implementation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _load_image_list(self) -> List[str]:
        """Load Pascal Person Parts image list"""
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
            
        image_names = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_names) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
            
        return sorted(image_names)
    
    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load Pascal Person Parts image and mask"""
        img_name = self.image_names[idx]
        
        # Determine mask name (assuming .jpg -> .png conversion)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        return image, mask


class MultiScaleDataset(torch.utils.data.Dataset):
    """Multi-scale training dataset wrapper"""
    
    def __init__(self, base_dataset: BaseSegmentationDataset, scales: List[float] = None):
        self.base_dataset = base_dataset
        self.scales = scales or [0.75, 0.85, 1.0, 1.15, 1.3]
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base item
        image, mask = self.base_dataset._load_image_and_mask(idx)
        
        # Apply multi-scale if training
        if self.base_dataset.split == 'train':
            scale = random.choice(self.scales)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            image = cv2.resize(image, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Apply augmentations
            image, mask = self.base_dataset.apply_augmentations(image, mask)
        
        # Preprocess to final size
        image_tensor = self.base_dataset.preprocess_image(image)
        mask_tensor = self.base_dataset.preprocess_mask(mask)
        
        return image_tensor, mask_tensor


def create_dataset(dataset_name: str, 
                  root_dir: str, 
                  split: str, 
                  image_size: int, 
                  tier: str,
                  multi_scale: bool = False) -> BaseSegmentationDataset:
    """Factory function to create datasets"""
    
    if dataset_name.lower() == 'pascal_person_parts':
        base_dataset = PascalPersonPartsDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            tier=tier
        )
        
        if multi_scale and split == 'train':
            return MultiScaleDataset(base_dataset)
        return base_dataset
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Export functions for convenience
def create_pascal_dataset(root_dir: str, split: str, image_size: int, tier: str, multi_scale: bool = False):
    """Create Pascal Person Parts dataset"""
    return create_dataset('pascal_person_parts', root_dir, split, image_size, tier, multi_scale)