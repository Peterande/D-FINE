import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
from PIL import Image
import torchvision.transforms as transforms
import random

class PascalPersonPartsCalibrator(trt.IInt8EntropyCalibrator2):
    """Pascal Person Parts only calibrator - optimized for human segmentation"""
    
    def __init__(self, pascal_data_dir, batch_size=1, cache_file="pascal_calibration.cache", 
                 max_calibration_images=500):
        super(PascalPersonPartsCalibrator, self).__init__()
        
        # Image preprocessing - match your training preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Find Pascal Person Parts images
        self.image_files = []
        
        # Look in datasets/pascal_person_parts/images/
        pascal_image_dirs = [
            os.path.join(pascal_data_dir, 'images', 'val'),     # Validation set (preferred)
            os.path.join(pascal_data_dir, 'images', 'train'),   # Training set if val is small
            os.path.join(pascal_data_dir, 'images'),            # Direct images folder
            pascal_data_dir  # Root directory fallback
        ]
        
        for img_dir in pascal_image_dirs:
            if os.path.exists(img_dir):
                print(f"Searching Pascal images in: {img_dir}")
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.image_files.append(os.path.join(root, file))
                
                if len(self.image_files) >= max_calibration_images:
                    break
        
        # Limit and shuffle for better representation
        if len(self.image_files) > max_calibration_images:
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_calibration_images]
            
        print(f"Pascal-only calibration: Found {len(self.image_files)} images")
        if len(self.image_files) == 0:
            raise ValueError(f"No Pascal Person Parts images found in {pascal_data_dir}")
        
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate GPU memory
        self.device_input_image = cuda.mem_alloc(self.batch_size * 3 * 640 * 640 * 4)
        self.device_input_sizes = cuda.mem_alloc(self.batch_size * 2 * 8)
        self.host_sizes = np.array([[640, 640] for _ in range(batch_size)], dtype=np.int64)
        self.stream = cuda.Stream()
        
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            print(f"Pascal calibration complete. Processed {self.current_index} images.")
            return None
        
        print(f"Pascal calibration batch {self.current_index//self.batch_size + 1}/{len(self.image_files)//self.batch_size + 1}")
        
        batch_data = np.zeros((self.batch_size, 3, 640, 640), dtype=np.float32)
        
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
                    print(f"  Processing Pascal image: {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if valid_images == 0:
            self.current_index += self.batch_size
            return self.get_batch(names)
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.device_input_image, batch_data.ravel(), self.stream)
        cuda.memcpy_htod_async(self.device_input_sizes, self.host_sizes.ravel(), self.stream)
        self.stream.synchronize()
        
        self.current_index += self.batch_size
        
        # Return bindings
        bindings = []
        for name in names:
            if name == "images":
                bindings.append(int(self.device_input_image))
            elif name == "orig_target_sizes":
                bindings.append(int(self.device_input_sizes))
            else:
                print(f"Warning: Unknown binding name: {name}")
                bindings.append(0)
        
        return bindings
        
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
                print(f"Read {len(cache_data)} bytes from Pascal calibration cache")
                return cache_data
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"Wrote {len(cache)} bytes to Pascal calibration cache")


class MixedDatasetCalibrator(trt.IInt8EntropyCalibrator2):
    """Mixed Pascal Person Parts + COCO calibrator for balanced detection + segmentation"""
    
    def __init__(self, pascal_data_dir, coco_data_dir, pascal_ratio=0.7, batch_size=1, 
                 cache_file="mixed_calibration.cache", max_calibration_images=500):
        super(MixedDatasetCalibrator, self).__init__()
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Collect Pascal Person Parts images
        pascal_images = []
        pascal_image_dirs = [
            os.path.join(pascal_data_dir, 'images', 'val'),
            os.path.join(pascal_data_dir, 'images', 'train'),
            os.path.join(pascal_data_dir, 'images'),
            pascal_data_dir
        ]
        
        for img_dir in pascal_image_dirs:
            if os.path.exists(img_dir):
                print(f"Searching Pascal images in: {img_dir}")
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            pascal_images.append(os.path.join(root, file))
                break  # Use first found directory
        
        # Collect COCO images from datasets/coco_val2017/
        coco_images = []
        if os.path.exists(coco_data_dir):
            print(f"Searching COCO images in: {coco_data_dir}")
            for root, dirs, files in os.walk(coco_data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        coco_images.append(os.path.join(root, file))
        else:
            print(f"Warning: COCO directory {coco_data_dir} not found, using Pascal-only")
        
        # Mix datasets according to ratio
        num_pascal = min(len(pascal_images), int(max_calibration_images * pascal_ratio))
        num_coco = min(len(coco_images), max_calibration_images - num_pascal) if coco_images else 0
        
        # Randomly sample images
        selected_pascal = []
        if pascal_images:
            random.shuffle(pascal_images)
            selected_pascal = pascal_images[:num_pascal]
        
        selected_coco = []
        if coco_images and num_coco > 0:
            random.shuffle(coco_images)
            selected_coco = coco_images[:num_coco]
        
        # Combine and shuffle final dataset
        self.image_files = selected_pascal + selected_coco
        random.shuffle(self.image_files)
        
        print(f"Mixed calibration dataset:")
        print(f"  Pascal Person Parts images: {len(selected_pascal)}")
        print(f"  COCO images: {len(selected_coco)}")
        print(f"  Total calibration images: {len(self.image_files)}")
        
        if len(self.image_files) == 0:
            raise ValueError("No calibration images found. Check Pascal and COCO dataset paths.")
        
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate GPU memory
        self.device_input_image = cuda.mem_alloc(self.batch_size * 3 * 640 * 640 * 4)
        self.device_input_sizes = cuda.mem_alloc(self.batch_size * 2 * 8)
        self.host_sizes = np.array([[640, 640] for _ in range(batch_size)], dtype=np.int64)
        self.stream = cuda.Stream()
        
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            print(f"Mixed calibration complete. Processed {self.current_index} images.")
            return None
        
        print(f"Mixed calibration batch {self.current_index//self.batch_size + 1}/{len(self.image_files)//self.batch_size + 1}")
        
        batch_data = np.zeros((self.batch_size, 3, 640, 640), dtype=np.float32)
        
        for i in range(self.batch_size):
            if self.current_index + i >= len(self.image_files):
                break
                
            image_path = self.image_files[self.current_index + i]
            
            try:
                pil_image = Image.open(image_path).convert('RGB')
                processed_image = self.transform(pil_image)
                batch_data[i] = processed_image.numpy()
                
                if i == 0:
                    dataset_type = "Pascal" if "pascal" in image_path.lower() else "COCO"
                    print(f"  Processing {dataset_type} image: {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.device_input_image, batch_data.ravel(), self.stream)
        cuda.memcpy_htod_async(self.device_input_sizes, self.host_sizes.ravel(), self.stream)
        self.stream.synchronize()
        
        self.current_index += self.batch_size
        
        # Return bindings
        bindings = []
        for name in names:
            if name == "images":
                bindings.append(int(self.device_input_image))
            elif name == "orig_target_sizes":
                bindings.append(int(self.device_input_sizes))
            else:
                bindings.append(0)
        
        return bindings
        
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
                print(f"Read {len(cache_data)} bytes from mixed calibration cache")
                return cache_data
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"Wrote {len(cache)} bytes to mixed calibration cache")


class COCOOnlyCalibrator(trt.IInt8EntropyCalibrator2):
    """COCO-only calibrator for detection-focused optimization"""
    
    def __init__(self, coco_data_dir, batch_size=1, cache_file="coco_calibration.cache", 
                 max_calibration_images=500):
        super(COCOOnlyCalibrator, self).__init__()
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Find COCO images in datasets/coco_val2017/
        self.image_files = []
        if os.path.exists(coco_data_dir):
            print(f"Searching COCO images in: {coco_data_dir}")
            for root, dirs, files in os.walk(coco_data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_files.append(os.path.join(root, file))
        
        # Limit and shuffle
        if len(self.image_files) > max_calibration_images:
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_calibration_images]
            
        print(f"COCO-only calibration: Found {len(self.image_files)} images")
        if len(self.image_files) == 0:
            raise ValueError(f"No COCO images found in {coco_data_dir}")
        
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate GPU memory
        self.device_input_image = cuda.mem_alloc(self.batch_size * 3 * 640 * 640 * 4)
        self.device_input_sizes = cuda.mem_alloc(self.batch_size * 2 * 8)
        self.host_sizes = np.array([[640, 640] for _ in range(batch_size)], dtype=np.int64)
        self.stream = cuda.Stream()
        
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            print(f"COCO calibration complete. Processed {self.current_index} images.")
            return None
        
        print(f"COCO calibration batch {self.current_index//self.batch_size + 1}/{len(self.image_files)//self.batch_size + 1}")
        
        batch_data = np.zeros((self.batch_size, 3, 640, 640), dtype=np.float32)
        
        for i in range(self.batch_size):
            if self.current_index + i >= len(self.image_files):
                break
                
            image_path = self.image_files[self.current_index + i]
            
            try:
                pil_image = Image.open(image_path).convert('RGB')
                processed_image = self.transform(pil_image)
                batch_data[i] = processed_image.numpy()
                
                if i == 0:
                    print(f"  Processing COCO image: {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.device_input_image, batch_data.ravel(), self.stream)
        cuda.memcpy_htod_async(self.device_input_sizes, self.host_sizes.ravel(), self.stream)
        self.stream.synchronize()
        
        self.current_index += self.batch_size
        
        # Return bindings
        bindings = []
        for name in names:
            if name == "images":
                bindings.append(int(self.device_input_image))
            elif name == "orig_target_sizes":
                bindings.append(int(self.device_input_sizes))
            else:
                bindings.append(0)
        
        return bindings
        
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
                print(f"Read {len(cache_data)} bytes from COCO calibration cache")
                return cache_data
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"Wrote {len(cache)} bytes to COCO calibration cache")