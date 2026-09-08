#!/usr/bin/env python3
"""
DFINE Segmentation TensorRT Engine Builder

Converts ONNX segmentation model to TensorRT engine for faster inference.
Usage:
    python tensorrt/build_segmentation_engine.py -i model_segmentation.onnx -o model_segmentation.engine
"""

import os
import sys
import argparse
import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DFINE Segmentation TensorRT Engine Builder')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input ONNX segmentation model')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to output TensorRT engine (default: input.onnx -> input.engine)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision (default: FP32)')
    parser.add_argument('--workspace', type=int, default=2,
                        help='Maximum workspace size in GB (default: 2GB for segmentation)')
    parser.add_argument('--max-batch-size', type=int, default=1,
                        help='Maximum batch size (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--min-timing-iterations', type=int, default=2,
                        help='Minimum number of timing iterations')
    parser.add_argument('--avg-timing-iterations', type=int, default=1,
                        help='Average number of timing iterations')
    parser.add_argument('--force', action='store_true',
                        help='Force engine building even if output file exists')
    return parser.parse_args()

def get_engine_info():
    """Print TensorRT version information"""
    print(f"TensorRT version: {trt.__version__}")
    
    # Get CUDA version
    try:
        cuda_version = cuda.get_version()
        print(f"CUDA driver version: {cuda_version}")
    except Exception as e:
        print(f"Error getting CUDA version: {e}")
    
    # Check if FP16 is supported
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    has_fp16 = builder.platform_has_fast_fp16
    print(f"Platform has fast FP16: {has_fp16}")

def build_segmentation_engine(args):
    """Build TensorRT engine from ONNX segmentation model"""
    # Set logger level
    if args.verbose:
        logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        logger = trt.Logger(trt.Logger.WARNING)
    
    # Check if output engine file exists
    output_file = args.output if args.output else args.input.replace('.onnx', '.engine')
    if os.path.exists(output_file) and not args.force:
        print(f"Engine file {output_file} already exists. Use --force to overwrite.")
        return output_file
    
    # Print build information
    print(f"Building TensorRT segmentation engine from {args.input} to {output_file}")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"Workspace size: {args.workspace} GB")
    print(f"Maximum batch size: {args.max_batch_size}")
    
    start_time = time.time()
    
    # Create builder and config
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set max workspace size (in bytes) - UPDATED for TensorRT 10.x
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 30))
    
    # Set precision flags
    if args.fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 mode")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: Platform doesn't support fast FP16, falling back to FP32")
    
    # Set optimization flags if available in this TensorRT version
    try:
        config.builder_optimization_level = 3  # Maximum optimization
    except:
        print("Note: Builder optimization level not set (may not be available in this TensorRT version)")
    
    # Create network definition with explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    
    # Read the ONNX model file
    print(f"Parsing ONNX segmentation model: {args.input}")
    with open(args.input, 'rb') as model:
        model_bytes = model.read()
        if not parser.parse(model_bytes):
            print("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                print(f"  {parser.get_error(error)}")
            return None
    
    # Print network information
    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  Input {i}: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    print(f"Network outputs: {network.num_outputs}")
    expected_outputs = ['labels', 'boxes', 'scores', 'seg_probs', 'seg_preds']
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        expected_name = expected_outputs[i] if i < len(expected_outputs) else f"output_{i}"
        print(f"  Output {i}: {tensor.name} (expected: {expected_name}), shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    if network.num_outputs != 5:
        print(f"Warning: Expected 5 outputs for segmentation model, found {network.num_outputs}")
        print("Expected outputs: labels, boxes, scores, seg_probs, seg_preds")
    
    # Create optimization profile for dynamic shapes
    print("Creating optimization profile for dynamic input shapes...")
    profile = builder.create_optimization_profile()
    
    # Set shapes for each input tensor based on its name
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        # Check if this input has dynamic dimensions
        if -1 in input_shape:
            # Create a shape with actual values (replacing -1 with concrete values)
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for dim in input_shape:
                if dim == -1:
                    # This is a dynamic dimension, typically batch size
                    min_shape.append(1)           # Minimum batch size (1)
                    opt_shape.append(1)           # Optimal batch size (1 for video processing)
                    max_shape.append(args.max_batch_size)  # Maximum batch size
                else:
                    # Fixed dimension
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            # Convert to tuple
            min_shape = tuple(min_shape)
            opt_shape = tuple(opt_shape)
            max_shape = tuple(max_shape)
            
            print(f"  Setting profile for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    # Add the optimization profile to the config
    config.add_optimization_profile(profile)
    
    # Mark output tensors if not done in ONNX
    if network.num_outputs == 0:
        print("Warning: No outputs detected in ONNX model. Marking last layer's outputs.")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                network.mark_output(layer.get_output(j))
    
    # Build engine - UPDATED for TensorRT 10.x
    print("Building TensorRT segmentation engine (this may take a while)...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("Failed to build TensorRT engine!")
        return None
    
    # Save engine to file
    with open(output_file, 'wb') as f:
        f.write(plan)
    
    build_time = time.time() - start_time
    print(f"Segmentation engine built successfully in {build_time:.2f} seconds")
    print(f"Engine saved to: {output_file}")
    
    return output_file

def verify_segmentation_engine(engine_file):
    """Verify the TensorRT segmentation engine by loading it"""
    print(f"Verifying segmentation engine: {engine_file}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    try:
        with open(engine_file, 'rb') as f:
            engine_bytes = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if not engine:
            print("Failed to deserialize engine!")
            return False
        
        print("Segmentation engine verification successful!")
        
        # Print engine information using TensorRT 10.x API
        try:
            # Try new API first (TensorRT 10.x)
            input_count = 0
            output_count = 0
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_count += 1
                else:
                    output_count += 1
                    
            print(f"Engine inputs: {input_count}")
            print(f"Engine outputs: {output_count}")
            
            # Check that we have the expected number of outputs for segmentation
            if output_count == 5:
                print("✅ Correct number of outputs for segmentation model (5)")
            else:
                print(f"⚠️  Warning: Expected 5 outputs, found {output_count}")
            
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    shape = engine.get_tensor_shape(tensor_name)
                    dtype = engine.get_tensor_dtype(tensor_name)
                    print(f"  Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
                else:
                    shape = engine.get_tensor_shape(tensor_name)
                    dtype = engine.get_tensor_dtype(tensor_name)
                    # Try to identify the output type
                    output_type = "unknown"
                    if 'label' in tensor_name.lower() or 'class' in tensor_name.lower():
                        output_type = "detection_labels"
                    elif 'box' in tensor_name.lower():
                        output_type = "detection_boxes"
                    elif 'score' in tensor_name.lower():
                        output_type = "detection_scores"
                    elif 'seg_prob' in tensor_name.lower():
                        output_type = "segmentation_probabilities"
                    elif 'seg_pred' in tensor_name.lower():
                        output_type = "segmentation_predictions"
                    
                    print(f"  Output: {tensor_name} ({output_type}), shape: {shape}, dtype: {dtype}")
        except:
            # Fall back to old API (just in case)
            print("Note: Using legacy API for engine inspection")
            input_count = sum(1 for i in range(engine.num_bindings) if engine.binding_is_input(i))
            output_count = engine.num_bindings - input_count
            print(f"Engine inputs: {input_count}")
            print(f"Engine outputs: {output_count}")
            
            if output_count == 5:
                print("✅ Correct number of outputs for segmentation model (5)")
            else:
                print(f"⚠️  Warning: Expected 5 outputs, found {output_count}")
            
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                if engine.binding_is_input(i):
                    print(f"  Input {i}: {name}, shape: {shape}, dtype: {dtype}")
                else:
                    print(f"  Output {i}: {name}, shape: {shape}, dtype: {dtype}")
        
        return True
    except Exception as e:
        print(f"Engine verification failed: {e}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    # Print TensorRT information
    get_engine_info()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} not found!")
        sys.exit(1)
    
    # Set default output path if not provided
    if not args.output:
        args.output = args.input.replace('.onnx', '.engine')
    
    # Build engine
    engine_file = build_segmentation_engine(args)
    if not engine_file:
        print("Engine building failed!")
        sys.exit(1)
    
    # Verify engine
    if not verify_segmentation_engine(engine_file):
        print("Engine verification failed!")
        sys.exit(1)
    
    print(f"TensorRT segmentation engine successfully built and saved to {engine_file}")
    print("\nTo use this engine with the DFINE segmentation model:")
    print(f"  python tensorrt/infer_segmentation_trt.py -e {engine_file} -i input_video.mp4 -o output_video.mp4")

if __name__ == "__main__":
    main()