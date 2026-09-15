#!/usr/bin/env python3
"""
Standard DFINE Segmentation TensorRT Inference Script
Refactored to use base classes and modular structure
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_root = os.path.dirname(os.path.dirname(current_dir))  # Go up to segmentation_sivert/

# Add segmentation_sivert to path for core imports
if segmentation_root not in sys.path:
    sys.path.insert(0, segmentation_root)
from core.engines import create_segmentation_engine, benchmark_engine


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Standard DFINE Segmentation TensorRT Inference')
    
    # Required arguments
    parser.add_argument('-e', '--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input video file')
    
    # Output configuration
    parser.add_argument('-o', '--output', type=str,
                        help='Path to output video file (required for video processing mode)')
    
    # Processing modes
    parser.add_argument('--benchmark', action='store_true',
                        help='Run pure inference benchmark mode')
    parser.add_argument('--parallel-streams', type=int, default=1,
                        help='Number of parallel streams for benchmark (default: 1)')
    
    # Processing options
    parser.add_argument('--max-frames', type=int,
                        help='Maximum number of frames to process')
    parser.add_argument('--resize-factor', type=float, default=1.0,
                        help='Resize input frames by this factor')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--seg-alpha', type=float, default=0.6,
                        help='Segmentation overlay alpha')
    
    # Video output options
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF value for video encoding (default: 23)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'],
                        help='x264 encoding preset')
    
    return parser.parse_args()


def get_body_part_colors() -> Dict[int, Tuple[int, int, int]]:
    """Get colors for each body part class"""
    return {
        0: (0, 0, 0),        # background - black (transparent)
        1: (255, 0, 0),      # head - red
        2: (0, 255, 0),      # torso - green  
        3: (0, 0, 255),      # arms - blue
        4: (255, 255, 0),    # hands - yellow
        5: (255, 0, 255),    # legs - magenta
        6: (0, 255, 255),    # feet - cyan
    }


def get_body_part_names() -> Dict[int, str]:
    """Get names for each body part class"""
    return {
        0: 'background',
        1: 'head',
        2: 'torso', 
        3: 'arms',
        4: 'hands',
        5: 'legs',
        6: 'feet'
    }


def create_segmentation_overlay(frame: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Create segmentation overlay on frame"""
    colors = get_body_part_colors()
    height, width = frame.shape[:2]
    
    # Resize segmentation mask to frame size
    seg_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        if class_id == 0:  # Skip background
            continue
        mask = (seg_resized == class_id)
        overlay[mask] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    return result


def draw_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """Draw detection boxes on frame"""
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # COCO class names (simplified)
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
    }
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)]
    
    detection_count = 0
    
    for i in range(len(scores)):
        if scores[i] < confidence_threshold:
            continue
            
        # Convert normalized coordinates to pixel coordinates
        x1 = int(boxes[i][0] * width)
        y1 = int(boxes[i][1] * height)
        x2 = int(boxes[i][2] * width)
        y2 = int(boxes[i][3] * height)
        
        label = int(labels[i])
        color = colors[label % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        class_name = class_names.get(label, f'Class {label}')
        label_text = f'{class_name}: {scores[i]:.2f}'
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        detection_count += 1
    
    return result, detection_count


def process_video_with_visualization(engine, input_video: str, output_video: str, args) -> Dict:
    """Process video with segmentation visualization"""
    print(f"\n🎬 Processing video with visualization")
    print(f"📥 Input: {input_video}")
    print(f"📤 Output: {output_video}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    
    # Calculate output dimensions
    if args.resize_factor != 1.0:
        output_width = int(width * args.resize_factor)
        output_height = int(height * args.resize_factor)
    else:
        output_width, output_height = width, height
    
    output_fps = args.fps if args.fps else video_fps
    
    print(f"📊 Video: {width}x{height} @ {video_fps:.1f} FPS")
    print(f"📊 Output: {output_width}x{output_height} @ {output_fps:.1f} FPS")
    print(f"📊 Processing {total_frames:,} frames")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir)
    
    # Performance tracking
    frame_count = 0
    total_detections = 0
    total_body_parts = 0
    start_time = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or (args.max_frames and frame_count >= args.max_frames):
                break
            
            frame_start_time = time.perf_counter()
            
            # Resize frame if needed
            if args.resize_factor != 1.0:
                frame = cv2.resize(frame, (output_width, output_height))
            
            # Run inference
            results = engine.infer_frame(frame)
            
            # Create visualization
            vis_frame = frame.copy()
            
            # Add segmentation overlay
            seg_preds = results['seg_preds'].astype(np.uint8)
            vis_frame = create_segmentation_overlay(vis_frame, seg_preds, args.seg_alpha)
            
            # Add detection boxes
            boxes = results['boxes']
            scores = results['scores']
            labels = results['labels']
            
            vis_frame, detection_count = draw_detections(
                vis_frame, boxes, scores, labels, args.confidence_threshold
            )
            
            # Count active body parts
            unique_parts = np.unique(seg_preds)
            active_parts = len([p for p in unique_parts if p != 0])
            
            # Add performance info
            frame_time = time.perf_counter() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # Performance overlay
            perf_text = [
                f"Frame: {frame_count+1}/{total_frames}",
                f"FPS: {current_fps:.1f}",
                f"Detections: {detection_count}",
                f"Body parts: {active_parts}",
                f"Standard Tier"
            ]
            
            y_offset = 30
            for text in perf_text:
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, vis_frame)
            
            # Update counters
            frame_count += 1
            total_detections += detection_count
            total_body_parts += active_parts
            
            # Progress reporting
            if frame_count % max(1, total_frames // 20) == 0:
                elapsed = time.perf_counter() - start_time
                processing_fps = frame_count / elapsed
                eta = (total_frames - frame_count) / processing_fps if processing_fps > 0 else 0
                print(f"   Processing: {frame_count:4d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                      f"Speed: {processing_fps:6.1f} FPS | ETA: {eta:4.1f}s")
    
    finally:
        cap.release()
    
    total_time = time.perf_counter() - start_time
    
    # Create video using ffmpeg
    print(f"🎬 Creating output video...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-framerate', str(output_fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', str(args.crf),
        '-preset', args.preset,
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"✅ Video saved to: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        raise
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
    
    # Calculate results
    overall_fps = frame_count / total_time
    
    return {
        'frames_processed': frame_count,
        'total_time': total_time,
        'processing_fps': overall_fps,
        'total_detections': total_detections,
        'total_body_parts': total_body_parts,
        'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
        'avg_body_parts_per_frame': total_body_parts / frame_count if frame_count > 0 else 0
    }


def run_benchmark(engine, input_video: str, args) -> Dict:
    """Run inference benchmark"""
    print(f"\n🚀 Running inference benchmark")
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    
    print(f"📊 Benchmarking {total_frames:,} frames")
    
    # Preprocess all frames
    frames = []
    frame_count = 0
    
    print("📥 Loading frames...")
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.resize_factor != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * args.resize_factor), int(w * args.resize_factor)
            frame = cv2.resize(frame, (new_w, new_h))
        
        frames.append(frame)
        frame_count += 1
        
        if frame_count % max(1, total_frames // 10) == 0:
            print(f"   Loaded {frame_count}/{total_frames} frames")
    
    cap.release()
    
    # Benchmark inference
    print("🏃 Running benchmark...")
    start_time = time.perf_counter()
    
    for i, frame in enumerate(frames):
        _ = engine.infer_frame(frame)
        
        if (i + 1) % max(1, len(frames) // 20) == 0:
            elapsed = time.perf_counter() - start_time
            current_fps = (i + 1) / elapsed
            eta = (len(frames) - i - 1) / current_fps if current_fps > 0 else 0
            print(f"   Progress: {i + 1:4d}/{len(frames)} | FPS: {current_fps:6.1f} | ETA: {eta:4.1f}s")
    
    total_time = time.perf_counter() - start_time
    
    # Get engine performance info
    engine_info = engine.get_engine_info()
    
    return {
        'frames_processed': len(frames),
        'total_time': total_time,
        'pure_inference_fps': len(frames) / total_time,
        'avg_inference_time_ms': engine_info.get('avg_inference_time_ms', 0),
        'engine_info': engine_info
    }


def main():
    """Main function"""
    args = parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.output:
        print("❌ Error: --output is required when not in benchmark mode")
        sys.exit(1)
    
    if not os.path.isfile(args.engine):
        print(f"❌ Error: Engine file not found: {args.engine}")
        sys.exit(1)
    
    if not os.path.isfile(args.input):
        print(f"❌ Error: Input video not found: {args.input}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 STANDARD DFINE SEGMENTATION TENSORRT INFERENCE")
    print("=" * 80)
    print(f"📁 Engine: {args.engine}")
    print(f"📁 Input: {args.input}")
    
    if args.benchmark:
        print(f"⚡ Mode: INFERENCE BENCHMARK")
    else:
        print(f"📁 Output: {args.output}")
        print(f"🎬 Mode: VIDEO PROCESSING")
    
    if args.max_frames:
        print(f"🔢 Max frames: {args.max_frames:,}")
    
    print("🎯 Tier: Standard (balanced performance)")
    print("=" * 80)
    
    try:
        # Create TensorRT engine
        engine = create_segmentation_engine('standard', args.engine)
        
        # Print engine info
        engine.print_engine_info()
        
        # Run processing
        if args.benchmark:
            # Benchmark mode
            results = run_benchmark(engine, args.input, args)
            
            print(f"\n🎉 BENCHMARK RESULTS:")
            print(f"   Frames processed: {results['frames_processed']:,}")
            print(f"   Total time: {results['total_time']:.2f}s")
            print(f"   Pure inference FPS: {results['pure_inference_fps']:.1f}")
            print(f"   Average inference time: {results['avg_inference_time_ms']:.2f}ms")
        
        else:
            # Video processing mode
            results = process_video_with_visualization(engine, args.input, args.output, args)
            
            print(f"\n🎉 VIDEO PROCESSING RESULTS:")
            print(f"   Frames processed: {results['frames_processed']:,}")
            print(f"   Processing FPS: {results['processing_fps']:.1f}")
            print(f"   Total detections: {results['total_detections']:,}")
            print(f"   Avg detections/frame: {results['avg_detections_per_frame']:.1f}")
            print(f"   Avg body parts/frame: {results['avg_body_parts_per_frame']:.1f}")
            print(f"   Real-time capable: {'✅ Yes' if results['processing_fps'] >= 30 else '⚠️ No'}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✅ Standard tier inference completed successfully!")


if __name__ == "__main__":
    main()