#!/usr/bin/env python3
"""
Core TensorRT Calibrators for DFINE Segmentation
Reusable INT8 calibration classes for different datasets
"""

import os
import numpy as np
import random
from typing import List, Optional
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseCalibrator(trt.IInt8EntropyCalibrator2, ABC):
    """Base class for INT8 calibrators"""
    
    def __init__(self, 
                 batch_size: int = 1,
                 cache_file: str = "calibration.cache",
                 max_calibration_images: int = 500,
                 input_shape: tuple = (640, 640)):
        super().__init__()
        
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.max_calibration_images = max_calibration_images
        self.input_shape = input_shape
        self.current_index = 0
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image paths
        self.image_files = self._load_image_paths()
        print(f"📊 Loaded {len(self.image_files)} images for calibration")
        
        # Allocate GPU memory
        self._allocate_memory()
    
    @abstractmethod
    def _load_image_paths(self) -> List[str]:
        """Load paths to calibration images"""
        pass
    
    def _allocate_memory(self):
        """Allocate GPU memory for calibration"""
        # Input image memory
        image_size = self.batch_size * 3 * self.input_shape[0] * self.input_shape[1] * 4  # float32
        self.device_input_image = cuda.mem_alloc(image_size)
        
        # Input sizes memory (for DFINE model)
        sizes_size = self.batch_size * 2 * 8  # int64
        self.device_input_sizes = cuda.mem_alloc(sizes_size)
        
        # Host memory for sizes
        self.host_sizes = np.array([[self.input_shape[1], self.input_shape[0]] for _ in range(self.batch_size)], dtype=np.int64)
        
        # CUDA stream
        self.stream = cuda.Stream()
    
    def get_batch_size(self):
        """Return batch size"""
        return self.batch_size
    
    def get_batch(self, names):
        """Get next batch for calibration"""
        if self.current_index + self.batch_size > len(self.image_files):
            print(f"📊 Calibration complete. Processed {self.current_index} images.")
            return None
        
        batch_progress = self.current_index // self.batch_size + 1
        total_batches = len(self.image_files) // self.batch_size + 1
        print(f"📊 Calibration batch {batch_progress}/{total_batches}")
        
        # Prepare batch data
        batch_data = np.zeros((self.batch_size, 3, *self.input_shape), dtype=np.float32)
        
        valid_images = 0
        for i in range(self.batch_size):
            if self.current_index + i >= len(self.image_files):
                break
            
            image_path = self.image_files[self.current_index + i]
            
            try:
                pil_image = Image.open(image_path).convert('RGB')
                processed_image = self.transform(pil_image)
                batch_data[i] = processed_image.numpy()
                valid_images += 1
                
                if i == 0:
                    print(f"   Processing: {os.path.basename(image_path)}")
            
            except Exception as e:
                print(f"   Error processing {image_path}: {e}")
                continue
        
        if valid_images == 0:
            self.current_index += self.batch_size
            return self.get_batch(names)
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.device_input_image, batch_data.ravel(), self.stream)
        cuda.memcpy_htod_async(self.device_input_sizes, self.host_sizes.ravel(), self.stream)
        self.stream.synchronize()
        
        self.current_index += self.batch_size
        
        # Return bindings based on input names
        bindings = []
        for name in names:
            if name in ["images", "input"]:
                bindings.append(int(self.device_input_image))
            elif name in ["orig_target_sizes", "sizes"]:
                bindings.append(int(self.device_input_sizes))
            else:
                print(f"⚠️  Unknown binding name: {name}")
                bindings.append(0)
        
        return bindings
    
    def read_calibration_cache(self):
        """Read calibration cache if exists"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
                print(f"📁 Read {len(cache_data)} bytes from calibration cache: {self.cache_file}")
                return cache_data
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"💾 Wrote {len(cache)} bytes to calibration cache: {self.cache_file}")


class PascalPersonPartsCalibrator(BaseCalibrator):
    """Pascal Person Parts calibrator for human segmentation optimization"""
    
    def __init__(self, 
                 pascal_data_dir: str,
                 batch_size: int = 1,
                 cache_file: str = "pascal_calibration.cache",
                 max_calibration_images: int = 500,
                 input_shape: tuple = (640, 640)):
        
        self.pascal_data_dir = pascal_data_dir
        super().__init__(batch_size, cache_file, max_calibration_images, input_shape)
    
    def _load_image_paths(self) -> List[str]:
        """Load Pascal Person Parts image paths"""
        image_files = []
        
        # Search in multiple possible directories
        search_dirs = [
            os.path.join(self.pascal_data_dir, 'images', 'val'),
            os.path.join(self.pascal_data_dir, 'images', 'train'),
            os.path.join(self.pascal_data_dir, 'images'),
            self.pascal_data_dir
        ]
        
        for img_dir in search_dirs:
            if os.path.exists(img_dir):
                print(f"🔍 Searching Pascal images in: {img_dir}")
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(os.path.join(root, file))
                
                if len(image_files) >= self.max_calibration_images:
                    break
        
        if len(image_files) == 0:
            raise ValueError(f"No Pascal images found in {self.pascal_data_dir}")
        
        # Limit and shuffle
        if len(image_files) > self.max_calibration_images:
            random.shuffle(image_files)
            image_files = image_files[:self.max_calibration_images]
        
        return image_files


class COCOCalibrator(BaseCalibrator):
    """COCO calibrator for general detection optimization"""
    
    def __init__(self, 
                 coco_data_dir: str,
                 batch_size: int = 1,
                 cache_file: str = "coco_calibration.cache",
                 max_calibration_images: int = 500,
                 input_shape: tuple = (640, 640)):
        
        self.coco_data_dir = coco_data_dir
        super().__init__(batch_size, cache_file, max_calibration_images, input_shape)
    
    def _load_image_paths(self) -> List[str]:
        """Load COCO image paths"""
        image_files = []
        
        if os.path.exists(self.coco_data_dir):
            print(f"🔍 Searching COCO images in: {self.coco_data_dir}")
            for root, dirs, files in os.walk(self.coco_data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, file))
        
        if len(image_files) == 0:
            raise ValueError(f"No COCO images found in {self.coco_data_dir}")
        
        # Limit and shuffle
        if len(image_files) > self.max_calibration_images:
            random.shuffle(image_files)
            image_files = image_files[:self.max_calibration_images]
        
        return image_files


class MixedDatasetCalibrator(BaseCalibrator):
    """Mixed Pascal + COCO calibrator for balanced optimization"""
    
    def __init__(self, 
                 pascal_data_dir: str,
                 coco_data_dir: str,
                 pascal_ratio: float = 0.7,
                 batch_size: int = 1,
                 cache_file: str = "mixed_calibration.cache",
                 max_calibration_images: int = 500,
                 input_shape: tuple = (640, 640)):
        
        self.pascal_data_dir = pascal_data_dir
        self.coco_data_dir = coco_data_dir
        self.pascal_ratio = pascal_ratio
        super().__init__(batch_size, cache_file, max_calibration_images, input_shape)
    
    def _load_image_paths(self) -> List[str]:
        """Load mixed dataset image paths"""
        # Collect Pascal images
        pascal_images = []
        pascal_search_dirs = [
            os.path.join(self.pascal_data_dir, 'images', 'val'),
            os.path.join(self.pascal_data_dir, 'images', 'train'),
            os.path.join(self.pascal_data_dir, 'images'),
            self.pascal_data_dir
        ]
        
        for img_dir in pascal_search_dirs:
            if os.path.exists(img_dir):
                print(f"🔍 Searching Pascal images in: {img_dir}")
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            pascal_images.append(os.path.join(root, file))
                break
        
        # Collect COCO images
        coco_images = []
        if os.path.exists(self.coco_data_dir):
            print(f"🔍 Searching COCO images in: {self.coco_data_dir}")
            for root, dirs, files in os.walk(self.coco_data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        coco_images.append(os.path.join(root, file))
        
        # Mix datasets according to ratio
        num_pascal = min(len(pascal_images), int(self.max_calibration_images * self.pascal_ratio))
        num_coco = min(len(coco_images), self.max_calibration_images - num_pascal) if coco_images else 0
        
        # Sample images
        selected_pascal = []
        if pascal_images:
            random.shuffle(pascal_images)
            selected_pascal = pascal_images[:num_pascal]
        
        selected_coco = []
        if coco_images and num_coco > 0:
            random.shuffle(coco_images)
            selected_coco = coco_images[:num_coco]
        
        # Combine and shuffle
        image_files = selected_pascal + selected_coco
        random.shuffle(image_files)
        
        print(f"📊 Mixed calibration dataset:")
        print(f"   Pascal images: {len(selected_pascal)}")
        print(f"   COCO images: {len(selected_coco)}")
        print(f"   Total: {len(image_files)}")
        
        if len(image_files) == 0:
            raise ValueError("No calibration images found in either dataset")
        
        return image_files


# Factory functions for creating calibrators
def create_pascal_calibrator(pascal_data_dir: str, **kwargs) -> PascalPersonPartsCalibrator:
    """Create Pascal Person Parts calibrator"""
    return PascalPersonPartsCalibrator(pascal_data_dir, **kwargs)


def create_coco_calibrator(coco_data_dir: str, **kwargs) -> COCOCalibrator:
    """Create COCO calibrator"""
    return COCOCalibrator(coco_data_dir, **kwargs)


def create_mixed_calibrator(pascal_data_dir: str, coco_data_dir: str, **kwargs) -> MixedDatasetCalibrator:
    """Create mixed dataset calibrator"""
    return MixedDatasetCalibrator(pascal_data_dir, coco_data_dir, **kwargs)


def create_calibrator(mode: str, 
                     pascal_data_dir: Optional[str] = None,
                     coco_data_dir: Optional[str] = None,
                     **kwargs):
    """Factory function to create calibrators based on mode"""
    
    if mode == 'pascal':
        if not pascal_data_dir:
            raise ValueError("pascal_data_dir required for Pascal mode")
        return create_pascal_calibrator(pascal_data_dir, **kwargs)
    
    elif mode == 'coco':
        if not coco_data_dir:
            raise ValueError("coco_data_dir required for COCO mode")
        return create_coco_calibrator(coco_data_dir, **kwargs)
    
    elif mode == 'mixed':
        if not pascal_data_dir or not coco_data_dir:
            raise ValueError("Both pascal_data_dir and coco_data_dir required for mixed mode")
        return create_mixed_calibrator(pascal_data_dir, coco_data_dir, **kwargs)
    
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")


# Legacy aliases for backward compatibility
class COCOOnlyCalibrator(COCOCalibrator):
    """Legacy alias"""
    pass