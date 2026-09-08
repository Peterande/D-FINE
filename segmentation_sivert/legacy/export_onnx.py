#!/usr/bin/env python3
"""
Standard DFINE Segmentation ONNX Export Script
Refactored to use base classes and modular structure
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_root = os.path.dirname(os.path.dirname(current_dir))  # Go up to segmentation_sivert/
project_root = os.path.dirname(segmentation_root)  # Go up to D-FINE-NEWBRINGER/

# Add segmentation_sivert to path for core imports
if segmentation_root not in sys.path:
    sys.path.insert(0, segmentation_root)

# Add project root to path for src imports
src_path = os.path.join(project_root, 'src')
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import core modules
from core.models import load_pretrained_dfine, get_actual_backbone_channels, create_combined_model

# Import model
from models.standard import create_standard_segmentation_head


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Standard DFINE Segmentation ONNX Export')
    
    # Model configuration
    parser.add_argument('-c', '--config', type=str, default="configs/dfine_hgnetv2_x_obj2coco.yml",
                        help='Path to original DFINE configuration YAML')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to trained segmentation model weights (.pth)')
    parser.add_argument('--original-weights', default="models/dfine_x_obj2coco.pth", type=str,
                        help='Path to original DFINE model weights (.pth)')
    
    # Export configuration
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to save the ONNX model (default: auto-generated)')
    parser.add_argument('--input-shape', type=str, default='1,3,640,640',
                        help='Input shape for the model (default: 1,3,640,640)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    
    # Model configuration
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of segmentation classes (default: 7 for Pascal Person Parts)')
    parser.add_argument('--feature-dim', type=int, default=256,
                        help='Feature dimension for segmentation head')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Options
    parser.add_argument('--check', action='store_true', default=True,
                        help='Check the exported ONNX model')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify the ONNX model')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for export (default: cpu)')
    
    return parser.parse_args()


def load_hyperparameters_from_checkpoint(checkpoint_path: str) -> dict:
    """Load hyperparameters from saved checkpoint"""
    print(f"📚 Loading hyperparameters from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"✅ Found saved hyperparameters:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
        return hyperparams
    
    elif 'model_info' in checkpoint:
        # Extract from model info
        model_info = checkpoint['model_info']
        hyperparams = {
            'feature_dim': model_info.get('fpn_feature_dim', 256),
            'dropout_rate': 0.1,  # Default
            'tier': model_info.get('tier', 'standard')
        }
        print(f"✅ Extracted hyperparameters from model_info:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
        return hyperparams
    
    else:
        print("⚠️  No hyperparameters found in checkpoint, using defaults")
        return {
            'feature_dim': 256,
            'dropout_rate': 0.1,
            'tier': 'standard'
        }


def create_segmentation_model(args) -> nn.Module:
    """Create segmentation model and load trained weights"""
    
    # Load hyperparameters from checkpoint
    hyperparams = load_hyperparameters_from_checkpoint(args.resume)
    
    # Override with command line arguments if provided
    feature_dim = args.feature_dim if args.feature_dim != 256 else hyperparams.get('feature_dim', 256)
    dropout_rate = args.dropout_rate if args.dropout_rate != 0.1 else hyperparams.get('dropout_rate', 0.1)
    
    print(f"🏗️  Creating standard segmentation model:")
    print(f"   Feature dim: {feature_dim}")
    print(f"   Dropout rate: {dropout_rate}")
    print(f"   Num classes: {args.num_classes}")
    
    # Load original DFINE model
    dfine_model = load_pretrained_dfine(args.config, args.original_weights)
    backbone_channels = get_actual_backbone_channels(dfine_model)
    
    print(f"   Backbone channels: {backbone_channels}")
    
    # Create standard segmentation head with correct hyperparameters
    seg_head = create_standard_segmentation_head(
        in_channels_list=backbone_channels,
        num_classes=args.num_classes,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )
    
    # Create combined model
    model = create_combined_model(
        dfine_model=dfine_model,
        seg_head=seg_head,
        freeze_detection=False  # Don't freeze during inference
    )
    
    # Load trained weights
    print(f"📚 Loading trained segmentation weights from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded trained segmentation model successfully!")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {e}")
        # Try non-strict loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded with non-strict mode")
        if missing_keys:
            print(f"   Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   Unexpected keys: {len(unexpected_keys)}")
    
    return model, hyperparams


class StandardSegmentationModelWrapper(nn.Module):
    """Wrapper model for ONNX export with proper postprocessing"""
    
    def __init__(self, seg_model, postprocessor):
        super().__init__()
        self.seg_model = seg_model.eval()
        self.postprocessor = postprocessor
        
    def forward(self, images, orig_target_sizes):
        # Run the segmentation model
        outputs = self.seg_model(images)
        
        # Extract detection outputs for postprocessing
        det_outputs = {k: v for k, v in outputs.items() if k != 'segmentation'}
        
        # Process detection outputs
        processed_det = self.postprocessor(det_outputs, orig_target_sizes)
        
        # Get segmentation outputs
        seg_logits = outputs['segmentation']
        
        # Apply softmax to get probabilities
        seg_probs = torch.softmax(seg_logits, dim=1)
        
        # Get segmentation predictions
        seg_preds = torch.argmax(seg_logits, dim=1)
        
        # Return standard segmentation outputs
        return processed_det[0], processed_det[1], processed_det[2], seg_probs, seg_preds


def export_onnx(args):
    """Export the standard segmentation model to ONNX format"""
    
    # Set device
    device = torch.device(args.device)
    
    # Create the segmentation model
    model, hyperparams = create_segmentation_model(args)
    model = model.to(device)
    
    # Create wrapper model for ONNX export
    try:
        from src.core import YAMLConfig
        cfg = YAMLConfig(args.config)
        postprocessor = cfg.postprocessor.deploy()
    except Exception as e:
        print(f"⚠️  Could not load postprocessor: {e}")
        print("    Using identity postprocessor")
        postprocessor = lambda x, y: (x.get('pred_logits', torch.empty(0)), 
                                     x.get('pred_boxes', torch.empty(0)), 
                                     torch.empty(0))
    
    wrapper_model = StandardSegmentationModelWrapper(model, postprocessor)
    wrapper_model.eval()
    
    # Parse input shape
    input_shape = [int(x) for x in args.input_shape.split(",")]
    print(f"📐 Using input shape: {input_shape}")
    
    # Create dummy input
    data = torch.randn(*input_shape).to(device)
    size = torch.tensor([[input_shape[3], input_shape[2]]]).to(device)  # width, height format
    
    # Run test forward pass
    print("🧪 Running test forward pass...")
    with torch.no_grad():
        try:
            test_outputs = wrapper_model(data, size)
            print(f"✅ Test forward pass successful")
            print(f"   Outputs: {len(test_outputs)} tensors")
            for i, out in enumerate(test_outputs):
                print(f"   Output {i}: shape {out.shape}, dtype {out.dtype}")
        except Exception as e:
            print(f"❌ Test forward pass failed: {e}")
            raise
    
    # Define dynamic axes for ONNX export
    dynamic_axes = {
        "images": {0: "N"},  # Batch dimension
        "orig_target_sizes": {0: "N"},  # Batch dimension
        "labels": {0: "N"},
        "boxes": {0: "N"},
        "scores": {0: "N"},
        "seg_probs": {0: "N"},
        "seg_preds": {0: "N"},
    }
    
    # Determine output file name
    if args.output:
        output_file = args.output
    else:
        # Create descriptive filename
        base_name = os.path.splitext(os.path.basename(args.resume))[0]
        feature_dim = hyperparams.get('feature_dim', 256)
        output_file = f"{base_name}_standard_fd{feature_dim}.onnx"
    
    print(f"📤 Exporting standard segmentation model to ONNX: {output_file}")
    print(f"   Model: {hyperparams.get('tier', 'standard')} tier")
    print(f"   Feature dim: {hyperparams.get('feature_dim', 256)}")
    print(f"   ONNX opset: {args.opset}")
    
    # Export to ONNX
    try:
        torch.onnx.export(
            wrapper_model,
            (data, size),
            output_file,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores", "seg_probs", "seg_preds"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            verbose=False,
            do_constant_folding=True,
            export_params=True
        )
        print("✅ ONNX export successful!")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise
    
    # Check the ONNX model
    if args.check:
        try:
            import onnx
            print("🔍 Checking ONNX model...")
            onnx_model = onnx.load(output_file)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model check successful!")
            
            # Print model info
            print(f"📊 ONNX Model Info:")
            print(f"   Inputs: {len(onnx_model.graph.input)}")
            print(f"   Outputs: {len(onnx_model.graph.output)}")
            print(f"   Nodes: {len(onnx_model.graph.node)}")
            
        except ImportError:
            print("⚠️  ONNX package not found, skipping model check")
            print("   Install with: pip install onnx")
        except Exception as e:
            print(f"❌ ONNX model check failed: {e}")
    
    # Simplify the ONNX model
    if args.simplify:
        try:
            import onnx
            import onnxsim
            print("🔧 Simplifying ONNX model...")
            
            # Create input shapes dictionary
            input_shapes = {
                "images": input_shape, 
                "orig_target_sizes": [input_shape[0], 2]
            }
            
            # Simplify
            onnx_model = onnx.load(output_file)
            model_simp, check = onnxsim.simplify(
                onnx_model, 
                test_input_shapes=input_shapes,
                dynamic_input_shape=True
            )
            
            if check:
                onnx.save(model_simp, output_file)
                print(f"✅ ONNX model simplified and saved")
            else:
                print("⚠️  ONNX model simplification could not be validated")
                
        except ImportError:
            print("⚠️  onnx-simplifier package not found, skipping simplification")
            print("   Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"❌ ONNX model simplification failed: {e}")
    
    # Print summary
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n🎉 Export completed successfully!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"🎯 Tier: Standard (balanced performance)")
    print(f"\n📝 Next steps:")
    print(f"   1. Build TensorRT engine:")
    print(f"      python tensorrt/standard/build_engine.py -i {output_file}")
    print(f"   2. Or build INT8 engine:")
    print(f"      python tensorrt/standard/build_int8_engine.py -i {output_file}")
    
    return output_file


def main():
    """Main function"""
    args = parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.resume):
        print(f"❌ Model checkpoint not found: {args.resume}")
        sys.exit(1)
    
    if not os.path.isfile(args.original_weights):
        print(f"❌ Original DFINE weights not found: {args.original_weights}")
        sys.exit(1)
    
    if not os.path.isfile(args.config):
        print(f"❌ DFINE config not found: {args.config}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 STANDARD DFINE SEGMENTATION ONNX EXPORT")
    print("=" * 60)
    print(f"📁 Checkpoint: {args.resume}")
    print(f"📁 Original weights: {args.original_weights}")
    print(f"📁 Config: {args.config}")
    print(f"🎯 Target: Balanced performance (standard tier)")
    print("=" * 60)
    
    try:
        export_onnx(args)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()