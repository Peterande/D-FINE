#!/usr/bin/env python3
"""
Complete DFINE Segmentation TensorRT INT8 Engine Builder

Builds optimized INT8 TensorRT engines for DFINE + Segmentation models with multiple calibration options.

Usage Examples:
    # Pascal Person Parts only (recommended for segmentation)
    python build_complete_segmentation_int8.py -i model.onnx --pascal-data datasets/pascal_person_parts --mode pascal

    # Mixed Pascal + COCO (balanced detection + segmentation)  
    python build_complete_segmentation_int8.py -i model.onnx --pascal-data datasets/pascal_person_parts --coco-data datasets/coco_val2017 --mode mixed

    # COCO only (detection-focused)
    python build_complete_segmentation_int8.py -i model.onnx --coco-data datasets/coco_val2017 --mode coco
"""

import os
import sys
import argparse
import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Import our calibrators
from int8_calibrator import (
    PascalPersonPartsCalibrator,
    MixedDatasetCalibrator,
    COCOOnlyCalibrator
)

def parse_args():
    parser = argparse.ArgumentParser(description='Complete DFINE Segmentation INT8 Engine Builder')
    
    # Required arguments
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input ONNX segmentation model')
    
    # Dataset paths
    parser.add_argument('--pascal-data', type=str, default='datasets/pascal_person_parts',
                        help='Path to Pascal Person Parts dataset (default: datasets/pascal_person_parts)')
    parser.add_argument('--coco-data', type=str, default='datasets/coco_val2017',
                        help='Path to COCO validation dataset (default: datasets/coco_val2017)')
    
    # Calibration mode selection
    parser.add_argument('--mode', type=str, choices=['pascal', 'mixed', 'coco'], default='mixed',
                        help='Calibration mode: pascal (Pascal-only), mixed (Pascal+COCO), coco (COCO-only)')
    
    # Optional arguments
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to output TensorRT engine (default: auto-generated)')
    parser.add_argument('--workspace', type=int, default=4,
                        help='Maximum workspace size in GB (default: 4)')
    parser.add_argument('--max-calibration-images', type=int, default=5000,
                        help='Maximum number of images for calibration (default: 500)')
    parser.add_argument('--pascal-ratio', type=float, default=0.7,
                        help='Pascal ratio in mixed mode (default: 0.7 = 70% Pascal, 30% COCO)')
    
    # Build options
    parser.add_argument('--force', action='store_true',
                        help='Force engine building even if output file exists')
    parser.add_argument('--fp16-fallback', action='store_true', default=False,
                        help='Use FP16 for segmentation layers (recommended, but disable if build fails)')
    parser.add_argument('--int8-only', action='store_true',
                        help='Use pure INT8 without FP16 fallback (faster build, may reduce accuracy)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def get_engine_info():
    """Print TensorRT and system information"""
    print("="*60)
    print("TENSORRT SYSTEM INFORMATION")
    print("="*60)
    print(f"TensorRT version: {trt.__version__}")
    
    try:
        cuda_version = cuda.get_version()
        print(f"CUDA driver version: {cuda_version}")
    except Exception as e:
        print(f"Error getting CUDA version: {e}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print(f"Platform has fast INT8: {builder.platform_has_fast_int8}")
    print(f"Platform has fast FP16: {builder.platform_has_fast_fp16}")
    
    if not builder.platform_has_fast_int8:
        print("⚠️  WARNING: Your GPU may not support efficient INT8. Consider using FP16 instead.")
    if not builder.platform_has_fast_fp16:
        print("⚠️  WARNING: Your GPU may not support efficient FP16. Mixed precision may be limited.")
    
    print("="*60)

def identify_segmentation_layers(network):
    """Identify layers that should use FP16 instead of INT8 for better accuracy"""
    
    fp16_layers = []
    
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        layer_name = layer.name
        layer_type = layer.type
        
        should_use_fp16 = False
        
        # EXCLUDE layers that CANNOT use FP16 (these must stay as INT64/INT32)
        excluded_layer_types = [
            trt.LayerType.CONSTANT,        # Constant layers with INT weights
            trt.LayerType.SHAPE,           # Shape layers must be INT64
            trt.LayerType.GATHER,          # Index operations
            trt.LayerType.CAST,            # Type casting operations
            trt.LayerType.SLICE,           # Indexing operations
        ]
        
        # Skip if this is an excluded layer type
        if layer_type in excluded_layer_types:
            continue
            
        # Skip shape-related layers (these have integer outputs and can't be FP16)
        shape_keywords = [
            "constant", "shape", "gather", "cast", "slice", 
            "elementwise", "concat", "split", "unsqueeze"
        ]
        
        if any(keyword in layer_name.lower() for keyword in shape_keywords):
            continue
        
        # Only target specific segmentation-related COMPUTATION layers
        segmentation_patterns = [
            "seg_head",        # Segmentation head layers
            "/aspp/",          # ASPP module layers
            "/fpn/",           # FPN module layers  
            "/decoder/",       # Decoder layers
        ]
        
        # Check if this is a segmentation computation layer
        for pattern in segmentation_patterns:
            if pattern in layer_name.lower():
                # Additional check: only if it's a computation layer type
                computation_layer_types = [
                    trt.LayerType.CONVOLUTION,
                    trt.LayerType.DECONVOLUTION, 
                    trt.LayerType.ACTIVATION,
                    trt.LayerType.POOLING,
                    trt.LayerType.SCALE,
                    trt.LayerType.SOFTMAX,
                    trt.LayerType.RESIZE,
                ]
                
                if layer_type in computation_layer_types:
                    should_use_fp16 = True
                    break
        
        # Also target final output layers (last 5 layers only)
        if layer_idx >= network.num_layers - 5:
            # But only if they're computation layers, not shape layers
            if layer_type in [trt.LayerType.CONVOLUTION, trt.LayerType.SOFTMAX, trt.LayerType.ACTIVATION]:
                should_use_fp16 = True
        
        if should_use_fp16:
            fp16_layers.append((layer_idx, layer_name, layer_type))
    
    return fp16_layers

def create_calibrator(args):
    """Create the appropriate calibrator based on mode"""
    
    if args.mode == 'pascal':
        print("🎯 CALIBRATION MODE: Pascal Person Parts Only")
        print("   - Optimized for human segmentation")
        print("   - Best for segmentation accuracy")
        print("   - Detection works well on humans")
        print(f"   - Using dataset: {args.pascal_data}")
        
        if not os.path.exists(args.pascal_data):
            raise ValueError(f"Pascal dataset not found: {args.pascal_data}")
        
        return PascalPersonPartsCalibrator(
            pascal_data_dir=args.pascal_data,
            batch_size=1,
            cache_file="pascal_segmentation_calibration.cache",
            max_calibration_images=args.max_calibration_images
        )
    
    elif args.mode == 'mixed':
        print("🎯 CALIBRATION MODE: Mixed Pascal + COCO")
        print(f"   - Pascal ratio: {args.pascal_ratio*100:.0f}%")
        print(f"   - COCO ratio: {(1-args.pascal_ratio)*100:.0f}%")
        print("   - Balanced detection + segmentation")
        print("   - Good general performance")
        print(f"   - Pascal dataset: {args.pascal_data}")
        print(f"   - COCO dataset: {args.coco_data}")
        
        if not os.path.exists(args.pascal_data):
            raise ValueError(f"Pascal dataset not found: {args.pascal_data}")
        if not os.path.exists(args.coco_data):
            raise ValueError(f"COCO dataset not found: {args.coco_data}")
        
        return MixedDatasetCalibrator(
            pascal_data_dir=args.pascal_data,
            coco_data_dir=args.coco_data,
            pascal_ratio=args.pascal_ratio,
            batch_size=1,
            cache_file="mixed_segmentation_calibration.cache",
            max_calibration_images=args.max_calibration_images
        )
    
    elif args.mode == 'coco':
        print("🎯 CALIBRATION MODE: COCO Only")
        print("   - Optimized for general detection")
        print("   - Broad object category coverage")
        print("   - May impact segmentation accuracy")
        print(f"   - Using dataset: {args.coco_data}")
        
        if not os.path.exists(args.coco_data):
            raise ValueError(f"COCO dataset not found: {args.coco_data}")
        
        return COCOOnlyCalibrator(
            coco_data_dir=args.coco_data,
            batch_size=1,
            cache_file="coco_segmentation_calibration.cache",
            max_calibration_images=args.max_calibration_images
        )
    
    else:
        raise ValueError(f"Unknown calibration mode: {args.mode}")

def build_segmentation_int8_engine(args):
    """Build optimized INT8 TensorRT engine"""
    
    # Set logger level
    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    
    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        mode_suffix = f"_{args.mode}"
        
        if args.int8_only:
            precision_suffix = "_int8_pure"
        elif args.fp16_fallback:
            precision_suffix = "_int8_mixed"
        else:
            precision_suffix = "_int8"
            
        args.output = f"{base_name}_segmentation{mode_suffix}{precision_suffix}.engine"
    
    # Check if output exists
    if os.path.exists(args.output) and not args.force:
        print(f"❌ Engine file {args.output} already exists.")
        print("   Use --force to overwrite, or specify different --output path.")
        return None
    
    print("\n" + "="*60)
    print("BUILDING INT8 SEGMENTATION ENGINE")
    print("="*60)
    print(f"📥 Input ONNX: {args.input}")
    print(f"📤 Output Engine: {args.output}")
    print(f"💾 Workspace: {args.workspace} GB")
    print(f"🖼️  Max calibration images: {args.max_calibration_images}")
    
    if args.int8_only:
        precision_desc = "Pure INT8"
    elif use_mixed_precision:
        precision_desc = "INT8 + FP16 mixed precision"
    else:
        precision_desc = "INT8 (FP16 disabled due to platform limitations)"
        
    print(f"⚡ Precision: {precision_desc}")
    print("="*60)
    
    start_time = time.time()
    
    # Create builder and config
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 30))
    
    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    print("✅ Enabled INT8 precision")
    
    # Enable FP16 for mixed precision
    if args.fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ Enabled FP16 fallback for critical layers")
    
    # Create calibrator
    print("\n📊 SETTING UP CALIBRATION...")
    calibrator = create_calibrator(args)
    config.int8_calibrator = calibrator
    
    # Create network
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    
    # Parse ONNX model
    print(f"\n🔍 PARSING ONNX MODEL: {args.input}")
    parser = trt.OnnxParser(network, logger)
    with open(args.input, 'rb') as model:
        model_bytes = model.read()
        if not parser.parse(model_bytes):
            print("❌ Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                print(f"   Error {error}: {parser.get_error(error)}")
            return None
    
    # Print network info
    print(f"✅ Parsed ONNX model successfully")
    print(f"   Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"     {i}: {tensor.name} {tensor.shape} ({tensor.dtype})")
    
    print(f"   Outputs: {network.num_outputs}")
    expected_outputs = ['labels', 'boxes', 'scores', 'seg_probs', 'seg_preds']
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        expected = expected_outputs[i] if i < len(expected_outputs) else f"output_{i}"
        print(f"     {i}: {tensor.name} ({expected}) {tensor.shape} ({tensor.dtype})")
    
    if network.num_outputs != 5:
        print(f"⚠️  Warning: Expected 5 outputs for segmentation model, found {network.num_outputs}")
    
    # Configure mixed precision
    if args.fp16_fallback:
        print(f"\n🔧 CONFIGURING MIXED PRECISION...")
        fp16_layers = identify_segmentation_layers(network)
        print(f"   Found {len(fp16_layers)} layers to use FP16 precision")
        
        fp16_count = 0
        for layer_idx, layer_name, layer_type in fp16_layers:
            try:
                layer = network.get_layer(layer_idx)
                layer.precision = trt.float16
                
                # Set output precision
                for i in range(layer.num_outputs):
                    layer.set_output_type(i, trt.float16)
                
                fp16_count += 1
                if args.verbose:
                    print(f"     Layer {layer_idx}: '{layer_name}' -> FP16")
                    
            except Exception as e:
                if args.verbose:
                    print(f"     Warning: Could not set precision for '{layer_name}': {e}")
        
        print(f"✅ Set {fp16_count}/{len(fp16_layers)} layers to FP16 precision")
    
    # Create optimization profile
    print(f"\n⚙️  CREATING OPTIMIZATION PROFILE...")
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        if -1 in input_shape:
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for dim in input_shape:
                if dim == -1:
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(1)
                else:
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            min_shape = tuple(min_shape)
            opt_shape = tuple(opt_shape)
            max_shape = tuple(max_shape)
            
            print(f"   {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    
    # Set optimization level
    try:
        config.builder_optimization_level = 5
        print("✅ Set optimization level to 5 (maximum)")
    except:
        try:
            config.builder_optimization_level = 3
            print("✅ Set optimization level to 3")
        except:
            print("ℹ️  Using default optimization level")
    
    # Build engine
    print(f"\n🏗️  BUILDING INT8 ENGINE...")
    print("   This will take a while due to INT8 calibration...")
    print("   Calibration determines optimal quantization scales for your model.")
    
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("❌ Failed to build INT8 engine!")
        return None
    
    # Save engine
    print(f"\n💾 SAVING ENGINE...")
    with open(args.output, 'wb') as f:
        f.write(plan)
    
    build_time = time.time() - start_time
    print(f"✅ Engine built successfully in {build_time:.1f} seconds")
    print(f"📁 Saved to: {args.output}")
    
    return args.output, use_mixed_precision

def verify_engine(engine_file):
    """Verify the built engine"""
    print(f"\n🔍 VERIFYING ENGINE: {engine_file}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    try:
        with open(engine_file, 'rb') as f:
            engine_bytes = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if not engine:
            print("❌ Failed to deserialize engine!")
            return False
        
        print("✅ Engine verification successful!")
        
        # Get engine info
        input_count = 0
        output_count = 0
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_count += 1
            else:
                output_count += 1
        
        print(f"   Inputs: {input_count}")
        print(f"   Outputs: {output_count}")
        
        if output_count == 5:
            print("✅ Correct outputs for segmentation model")
        else:
            print(f"⚠️  Expected 5 outputs, found {output_count}")
        
        # Print tensor details
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            dtype = engine.get_tensor_dtype(tensor_name)
            
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                print(f"     Input: {tensor_name} {shape} ({dtype})")
            else:
                print(f"     Output: {tensor_name} {shape} ({dtype})")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine verification failed: {e}")
        return False

def print_summary(args, engine_file, build_time):
    """Print build summary"""
    print("\n" + "="*60)
    print("🎉 BUILD SUMMARY")
    print("="*60)
    print(f"✅ INT8 segmentation engine built successfully!")
    print(f"📁 Engine file: {engine_file}")
    print(f"⏱️  Build time: {build_time:.1f} seconds")
    print(f"🎯 Calibration mode: {args.mode.upper()}")
    print(f"🖼️  Calibration images: {args.max_calibration_images}")
    print(f"⚡ Precision: INT8 + {'FP16 fallback' if args.fp16_fallback else 'INT8 only'}")
    
    print(f"\n📈 EXPECTED PERFORMANCE:")
    print(f"   • Speed: ~3-4x faster than FP32")
    print(f"   • Speed: ~2x faster than FP16") 
    print(f"   • Memory: ~75% reduction")
    print(f"   • Quality: Good (with mixed precision)")
    
    print(f"\n🚀 USAGE:")
    print(f"   python tensorrt/infer_segmentation_trt.py \\")
    print(f"       -e {engine_file} \\")
    print(f"       -i input_video.mp4 \\")
    print(f"       -o output_segmented.mp4")
    
    print("="*60)

def main():
    args = parse_args()
    
    # Print system info
    get_engine_info()
    
    # Validate inputs
    if not os.path.isfile(args.input):
        print(f"❌ Input ONNX file not found: {args.input}")
        sys.exit(1)
    
    # Build engine
    start_time = time.time()
    engine_file, use_mixed_precision = build_segmentation_int8_engine(args)
    
    if not engine_file:
        print("❌ Engine building failed!")
        sys.exit(1)
    
    # Verify engine
    if not verify_engine(engine_file):
        print("❌ Engine verification failed!")
        sys.exit(1)
    
    # Print summary
    build_time = time.time() - start_time
    print_summary(args, engine_file, build_time, use_mixed_precision)

if __name__ == "__main__":
    main()