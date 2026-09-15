#!/usr/bin/env python3
"""
Core TensorRT Engine Classes for DFINE Segmentation
Reusable TensorRT engine management and inference
"""

import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging


class BaseTensorRTEngine(ABC):
    """Base class for TensorRT engines with common functionality"""
    
    def __init__(self, engine_path: str, logger_level: int = trt.Logger.WARNING):
        self.engine_path = engine_path
        self.logger = trt.Logger(logger_level)
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        
        # Memory management
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.input_shapes = {}
        self.output_shapes = {}
        self.input_names = []
        self.output_names = []
        
        # Performance tracking
        self.inference_times = []
        
        # Load and setup engine
        self._load_engine()
        self._allocate_memory()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        print(f"🔧 Loading TensorRT engine: {self.engine_path}")
        
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_bytes = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        self.stream = cuda.Stream()
        print(f"✅ Engine loaded successfully")
    
    def _allocate_memory(self):
        """Pre-allocate host and device memory"""
        print(f"🧠 Allocating memory buffers...")
        
        # Check TensorRT version for API compatibility
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            self._allocate_memory_new_api()
        else:
            self._allocate_memory_legacy_api()
    
    def _allocate_memory_new_api(self):
        """Allocate memory using TensorRT 10.x+ API"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Handle dynamic shapes
            if -1 in shape:
                actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                self.context.set_input_shape(tensor_name, actual_shape)
                shape = actual_shape
            
            # Convert dtype and calculate size
            np_dtype, element_size = self._get_numpy_dtype(dtype)
            size = trt.volume(shape)
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(size * element_size)
            
            if is_input:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes[tensor_name] = shape
                self.input_names.append(tensor_name)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes[tensor_name] = shape
                self.output_names.append(tensor_name)
            
            # Set tensor address
            self.context.set_tensor_address(tensor_name, int(device_mem))
            
            print(f"   {'📥' if is_input else '📤'} {tensor_name}: {shape} ({size * element_size / 1024 / 1024:.1f} MB)")
    
    def _allocate_memory_legacy_api(self):
        """Allocate memory using legacy TensorRT API"""
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            
            # Handle dynamic shapes
            if -1 in shape:
                actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                self.context.set_binding_shape(i, actual_shape)
                shape = actual_shape
            
            # Convert dtype and calculate size
            np_dtype, element_size = self._get_numpy_dtype(dtype)
            size = trt.volume(shape)
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(size * element_size)
            
            if is_input:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes[binding_name] = shape
                self.input_names.append(binding_name)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes[binding_name] = shape
                self.output_names.append(binding_name)
            
            self.bindings.append(int(device_mem))
            
            print(f"   {'📥' if is_input else '📤'} {binding_name}: {shape} ({size * element_size / 1024 / 1024:.1f} MB)")
    
    def _get_numpy_dtype(self, trt_dtype) -> Tuple[np.dtype, int]:
        """Convert TensorRT dtype to numpy dtype and element size"""
        if trt_dtype == trt.DataType.FLOAT:
            return np.float32, 4
        elif trt_dtype == trt.DataType.HALF:
            return np.float16, 2
        elif trt_dtype == trt.DataType.INT32:
            return np.int32, 4
        elif trt_dtype == trt.DataType.INT64:
            return np.int64, 8
        elif trt_dtype == trt.DataType.INT8:
            return np.int8, 1
        else:
            return np.float32, 4  # Default fallback
    
    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Perform inference with timing"""
        start_time = time.perf_counter()
        
        # Copy inputs to device
        for i, input_data in enumerate(inputs):
            np.copyto(self.host_inputs[i][:input_data.size], input_data.ravel())
            cuda.memcpy_htod_async(self.device_inputs[i], self.host_inputs[i], self.stream)
        
        # Execute inference
        if hasattr(self.context, 'execute_async_v3'):
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, 'execute_async_v2'):
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            success = self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs from device
        outputs = []
        for i, (host_output, device_output, shape) in enumerate(
            zip(self.host_outputs, self.device_outputs, self.output_shapes.values())
        ):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            self.stream.synchronize()
            output = host_output[:np.prod(shape)].reshape(shape).copy()
            outputs.append(output)
        
        # Track inference time
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        
        return outputs
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        info = {
            'engine_path': self.engine_path,
            'num_inputs': len(self.input_names),
            'num_outputs': len(self.output_names),
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'input_names': self.input_names,
            'output_names': self.output_names
        }
        
        if self.inference_times:
            info.update({
                'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
                'min_inference_time_ms': np.min(self.inference_times) * 1000,
                'max_inference_time_ms': np.max(self.inference_times) * 1000,
                'avg_fps': 1.0 / np.mean(self.inference_times)
            })
        
        return info
    
    def print_engine_info(self):
        """Print engine information"""
        info = self.get_engine_info()
        print(f"📊 TensorRT Engine Info:")
        print(f"   Engine: {os.path.basename(info['engine_path'])}")
        print(f"   Inputs: {info['num_inputs']}")
        print(f"   Outputs: {info['num_outputs']}")
        
        if 'avg_inference_time_ms' in info:
            print(f"   Avg inference time: {info['avg_inference_time_ms']:.2f}ms")
            print(f"   Avg FPS: {info['avg_fps']:.1f}")
    
    @abstractmethod
    def preprocess(self, *args, **kwargs) -> List[np.ndarray]:
        """Preprocess inputs for inference"""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray], *args, **kwargs) -> Any:
        """Postprocess inference outputs"""
        pass


class SegmentationTensorRTEngine(BaseTensorRTEngine):
    """TensorRT engine for segmentation models"""
    
    def __init__(self, engine_path: str, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        super().__init__(engine_path)
        
        # Validate expected outputs for segmentation
        self._validate_segmentation_outputs()
    
    def _validate_segmentation_outputs(self):
        """Validate that engine has expected segmentation outputs"""
        expected_outputs = ['labels', 'boxes', 'scores', 'seg_probs', 'seg_preds']
        
        if len(self.output_names) != 5:
            print(f"⚠️  Warning: Expected 5 outputs for segmentation model, found {len(self.output_names)}")
            print(f"   Expected: {expected_outputs}")
            print(f"   Found: {self.output_names}")
        else:
            print(f"✅ Segmentation model outputs validated")
    
    def preprocess(self, frame: np.ndarray) -> List[np.ndarray]:
        """Preprocess frame for segmentation inference"""
        # Resize frame
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB and normalize
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        # Target size tensor
        target_size_tensor = np.array([[self.target_size[0], self.target_size[1]]], dtype=np.int64)
        
        return [input_tensor, target_size_tensor]
    
    def postprocess(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Postprocess segmentation inference outputs"""
        if len(outputs) != 5:
            raise ValueError(f"Expected 5 outputs, got {len(outputs)}")
        
        # Extract outputs
        labels = outputs[0].flatten()      # Detection labels
        boxes = outputs[1][0]              # Detection boxes (N, 4)
        scores = outputs[2].flatten()      # Detection scores  
        seg_probs = outputs[3][0]          # Segmentation probabilities (7, 640, 640)
        seg_preds = outputs[4][0]          # Segmentation predictions (640, 640)
        
        return {
            'labels': labels,
            'boxes': boxes,
            'scores': scores,
            'seg_probs': seg_probs,
            'seg_preds': seg_preds
        }
    
    def infer_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Complete inference pipeline for a single frame"""
        # Preprocess
        inputs = self.preprocess(frame)
        
        # Inference
        outputs = self.infer(inputs)
        
        # Postprocess
        results = self.postprocess(outputs)
        
        return results


class LightweightSegmentationEngine(SegmentationTensorRTEngine):
    """Lightweight segmentation engine optimized for speed"""
    
    def __init__(self, engine_path: str, target_size: Tuple[int, int] = (512, 512)):
        # Use smaller input size for lightweight model
        super().__init__(engine_path, target_size)
        print(f"🚀 Lightweight segmentation engine ready")
        print(f"   Target size: {target_size}")
        print(f"   Optimized for: Speed and low memory")


class StandardSegmentationEngine(SegmentationTensorRTEngine):
    """Standard segmentation engine with balanced performance"""
    
    def __init__(self, engine_path: str, target_size: Tuple[int, int] = (640, 640)):
        super().__init__(engine_path, target_size)
        print(f"🚀 Standard segmentation engine ready")
        print(f"   Target size: {target_size}")
        print(f"   Optimized for: Balanced performance")


class AdvancedSegmentationEngine(SegmentationTensorRTEngine):
    """Advanced segmentation engine with maximum accuracy"""
    
    def __init__(self, engine_path: str, target_size: Tuple[int, int] = (768, 768)):
        # Use larger input size for advanced model
        super().__init__(engine_path, target_size)
        print(f"🚀 Advanced segmentation engine ready")
        print(f"   Target size: {target_size}")
        print(f"   Optimized for: Maximum accuracy")
    
    def infer_multiscale(self, frame: np.ndarray, scales: List[float] = [0.75, 1.0, 1.25]) -> Dict[str, np.ndarray]:
        """Multi-scale inference for better accuracy"""
        all_results = []
        
        for scale in scales:
            # Scale frame
            h, w = frame.shape[:2]
            scaled_h, scaled_w = int(h * scale), int(w * scale) 
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))
            
            # Infer on scaled frame
            result = self.infer_frame(scaled_frame)
            all_results.append(result)
        
        # Ensemble results (simple averaging for seg_probs)
        ensemble_seg_probs = np.mean([r['seg_probs'] for r in all_results], axis=0)
        ensemble_seg_preds = np.argmax(ensemble_seg_probs, axis=0)
        
        # Use results from scale 1.0 for detection
        base_result = all_results[1] if len(all_results) > 1 else all_results[0]
        
        return {
            'labels': base_result['labels'],
            'boxes': base_result['boxes'], 
            'scores': base_result['scores'],
            'seg_probs': ensemble_seg_probs,
            'seg_preds': ensemble_seg_preds
        }


# Factory functions
def create_segmentation_engine(tier: str, engine_path: str, **kwargs) -> SegmentationTensorRTEngine:
    """Factory function to create segmentation engines"""
    
    if tier == 'lightweight':
        return LightweightSegmentationEngine(engine_path, **kwargs)
    elif tier == 'standard':
        return StandardSegmentationEngine(engine_path, **kwargs)
    elif tier == 'advanced':
        return AdvancedSegmentationEngine(engine_path, **kwargs)
    else:
        raise ValueError(f"Unknown engine tier: {tier}")


# Utility functions
def benchmark_engine(engine: BaseTensorRTEngine, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark engine performance"""
    print(f"🏃 Benchmarking engine for {num_iterations} iterations...")
    
    # Create dummy input
    dummy_inputs = []
    for shape in engine.input_shapes.values():
        dummy_input = np.random.randn(*shape).astype(np.float32)
        dummy_inputs.append(dummy_input)
    
    # Warmup
    for _ in range(10):
        _ = engine.infer(dummy_inputs)
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = engine.infer(dummy_inputs)
    
    total_time = time.perf_counter() - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    print(f"✅ Benchmark completed:")
    print(f"   Average inference time: {avg_time*1000:.2f}ms")
    print(f"   Average FPS: {fps:.1f}")
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'avg_fps': fps,
        'total_time_s': total_time,
        'iterations': num_iterations
    }


# Export classes
__all__ = [
    'BaseTensorRTEngine',
    'SegmentationTensorRTEngine',
    'LightweightSegmentationEngine',
    'StandardSegmentationEngine', 
    'AdvancedSegmentationEngine',
    'create_segmentation_engine',
    'benchmark_engine'
]