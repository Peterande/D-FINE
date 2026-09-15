#!/usr/bin/env python3
"""
Enhanced D-FINE Segmentation Video Testing Script
Complete working version that matches the robust training architecture
"""

import os
import sys
import argparse
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
import cv2

# Add src to path
sys.path.append('src')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced D-FINE Segmentation Video Testing')
    parser.add_argument('-c', '--config', type=str, default="configs/dfine_hgnetv2_x_obj2coco.yml",
                        help='Path to original DFINE configuration YAML')
    parser.add_argument('-m', '--model', type=str, default='outputs/dfine_segmentation_robust/best_model.pth',
                        help='Path to enhanced segmentation model checkpoint')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('-o', '--output', type=str, default='enhanced_segmentation_result.mp4',
                        help='Path to output video')
    parser.add_argument('-t', '--threshold', type=float, default=0.6,
                        help='Detection confidence threshold (default: 0.6)')
    parser.add_argument('--seg-alpha', type=float, default=0.6,
                        help='Segmentation overlay alpha (default: 0.6)')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--resize-factor', type=float, default=1.0,
                        help='Resize input frames by this factor')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF value for video encoding (0-51)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='x264 encoding preset')
    return parser.parse_args()


# Import core components
from src.core import YAMLConfig


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
    """Superior segmentation head matching training architecture"""
    
    def __init__(self, in_channels_list, num_classes=7, feature_dim=256, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
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
    """D-FINE with robust segmentation matching training architecture"""
    
    def __init__(self, dfine_model, seg_head):
        super().__init__()
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        
        # Set to eval mode for inference
        self.eval()
    
    def forward(self, x):
        # Get backbone features
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch
        det_outputs = self.dfine_model(x)
        outputs.update(det_outputs)
        
        # Segmentation branch
        seg_logits, boundary_map = self.seg_head(backbone_features)
        
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


def get_actual_backbone_channels(model):
    """Get actual backbone output channels"""
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)
    
    actual_channels = [feat.shape[1] for feat in backbone_features]
    return actual_channels


def load_enhanced_segmentation_model(config_file, checkpoint_path, device):
    """Load the enhanced segmentation model with correct architecture"""
    print(f"Loading enhanced D-FINE model from config: {config_file}")
    
    # Load base D-FINE model
    cfg = YAMLConfig(config_file)
    base_model = cfg.model
    
    # Detect actual backbone channels
    actual_channels = get_actual_backbone_channels(base_model)
    print(f"Detected backbone channels: {actual_channels}")
    
    # Load checkpoint
    print(f"Loading enhanced segmentation weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        hyperparams = checkpoint.get('hyperparameters', {})
    else:
        state_dict = checkpoint
        hyperparams = {}
    
    # Get hyperparameters from checkpoint or use defaults from best results
    feature_dim = hyperparams.get('feature_dim', 448)  # From your best training results
    dropout_rate = hyperparams.get('dropout_rate', 0.05)
    
    print(f"Using hyperparameters: feature_dim={feature_dim}, dropout_rate={dropout_rate}")
    
    # Create superior segmentation head with correct architecture
    seg_head = SuperiorSegmentationHead(
        in_channels_list=actual_channels,
        num_classes=7,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )
    
    # Create enhanced combined model
    model = DFineWithRobustSegmentation(
        dfine_model=base_model,
        seg_head=seg_head
    )
    
    # Load the state dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed, using filtered loading")
        # Filter state dict to only include matching keys with correct shapes
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ Model loaded with {len(filtered_state_dict)}/{len(state_dict)} weights")
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Enhanced segmentation model loaded successfully on {device}")
    return model


def get_body_part_colors():
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


def get_body_part_names():
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


def process_frame(model, frame_bgr, device, transforms_val, threshold=0.6):
    """Process a single frame with the enhanced segmentation model"""
    # Convert BGR to RGB and then to PIL
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Get original dimensions for detection postprocessing
    w, h = pil_image.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    # Transform image
    im_data = transforms_val(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(im_data)
    
    # Get segmentation results
    seg_logits = outputs['segmentation'][0]  # Remove batch dimension
    seg_pred = torch.argmax(seg_logits, dim=0).cpu().numpy()
    
    # Get detection results if available
    labels, boxes, scores = None, None, None
    if 'pred_logits' in outputs and 'pred_boxes' in outputs:
        # Process detection outputs (simplified, might need adjustment based on your postprocessor)
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        # Apply softmax and get scores
        pred_scores = torch.softmax(pred_logits, dim=-1)
        max_scores, pred_labels = torch.max(pred_scores, dim=-1)
        
        # Filter by threshold and person class (assuming class 0 is person)
        mask = (max_scores > threshold) & (pred_labels == 0)
        
        if mask.any():
            filtered_boxes = pred_boxes[mask]
            filtered_scores = max_scores[mask]
            filtered_labels = pred_labels[mask]
            
            # Convert boxes from normalized to pixel coordinates
            filtered_boxes = filtered_boxes * torch.tensor([w, h, w, h]).to(device)
            # Convert from cxcywh to xyxy
            boxes_xyxy = torch.zeros_like(filtered_boxes)
            boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2
            
            labels = [filtered_labels.cpu()]
            boxes = [boxes_xyxy.cpu()]
            scores = [filtered_scores.cpu()]
        else:
            labels, boxes, scores = [[]], [[]], [[]]
    
    return pil_image, seg_pred, labels, boxes, scores


def create_segmentation_overlay(pil_image, seg_pred, alpha=0.6):
    """Create segmentation overlay on PIL image"""
    colors = get_body_part_colors()
    
    # Create colored segmentation mask
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        if class_id == 0:  # Skip background
            continue
        mask = seg_pred == class_id
        colored_mask[mask] = color
    
    # Convert to PIL and resize to match original image
    mask_pil = Image.fromarray(colored_mask).resize(pil_image.size, Image.NEAREST)
    
    # Create overlay
    overlay = Image.blend(pil_image, mask_pil, alpha)
    
    return overlay


def draw_detections_and_segmentation(pil_image, seg_pred, labels, boxes, scores, 
                                   threshold=0.6, seg_alpha=0.6):
    """Draw both detection boxes and segmentation overlay"""
    
    # First create segmentation overlay
    result_image = create_segmentation_overlay(pil_image, seg_pred, seg_alpha)
    
    # Then draw detection boxes
    draw = ImageDraw.Draw(result_image)
    person_count = 0
    
    if labels and boxes and scores:
        # Process detection results (assuming batch size 1)
        scores_per_image = scores[0] if scores[0] is not None and len(scores[0]) > 0 else []
        boxes_per_image = boxes[0] if boxes[0] is not None and len(boxes[0]) > 0 else []
        
        for i in range(len(scores_per_image)):
            score = scores_per_image[i].item()
            if score > threshold:
                person_count += 1
                box = boxes_per_image[i].round().int().tolist()
                
                # Draw bounding box
                draw.rectangle(box, outline="red", width=3)
                
                # Draw label
                text = f"Person: {score:.2f}"
                text_y_pos = box[1] - 15 if box[1] > 20 else box[1] + 2
                draw.text((box[0] + 2, text_y_pos), text, fill="white")
    
    # Draw segmentation info
    unique_parts = np.unique(seg_pred)
    part_names = get_body_part_names()
    active_parts = [part_names[part_id] for part_id in unique_parts if part_id != 0]
    
    if active_parts:
        parts_text = f"Body parts: {', '.join(active_parts)}"
        draw.text((10, result_image.size[1] - 30), parts_text, fill="white")
    
    # Enhanced info display
    info_text = f"Robust Enhanced Model | People: {person_count} | Parts: {len(active_parts)}"
    draw.text((10, 10), info_text, fill="white")
    
    return result_image, person_count, len(active_parts)


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARNING: ffmpeg not found. Please install ffmpeg.")
        return False


def process_video(model, input_path, output_path, device, threshold, seg_alpha,
                 resize_factor, output_fps_target, crf, preset):
    """Process video with enhanced segmentation model"""
    
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for video output")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output dimensions
    if resize_factor != 1.0:
        output_width = int(original_width * resize_factor)
        output_height = int(original_height * resize_factor)
    else:
        output_width = original_width
        output_height = original_height
    
    # Model transforms
    transforms_val = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_fps = output_fps_target if output_fps_target is not None else original_fps
    if output_fps <= 0:
        output_fps = 30

    print(f"Input: {input_path} ({original_width}x{original_height} @ {original_fps:.2f} FPS)")
    print(f"Output: {output_path} ({output_width}x{output_height} @ {output_fps:.2f} FPS)")
    print(f"Device: {device}, Detection threshold: {threshold}, Seg alpha: {seg_alpha}")
    print(f"Using ROBUST ENHANCED architecture with superior segmentation head")

    # Setup ffmpeg
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{output_width}x{output_height}', '-pix_fmt', 'bgr24',
        '-r', str(output_fps), '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', preset, output_path,
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    frame_idx = 0
    total_people = 0
    total_parts_detected = 0
    processing_start_time = time.time()
    stdin_broken = False

    try:
        while True:
            ret, frame_bgr_original = cap.read()
            if not ret:
                break

            loop_start_time = time.time()

            # Resize frame if needed
            if resize_factor != 1.0:
                frame_to_process = cv2.resize(frame_bgr_original, 
                                            (output_width, output_height))
            else:
                frame_to_process = frame_bgr_original

            # Process frame with enhanced model
            pil_image, seg_pred, labels, boxes, scores = process_frame(
                model, frame_to_process, device, transforms_val, threshold
            )
            
            # Draw results with enhanced visualization
            result_pil, person_count, parts_count = draw_detections_and_segmentation(
                pil_image, seg_pred, labels, boxes, scores, threshold, seg_alpha
            )
            
            total_people += person_count
            total_parts_detected += parts_count

            # Convert back to BGR for ffmpeg
            output_frame_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            
            # Ensure correct dimensions
            if (output_frame_bgr.shape[1] != output_width or 
                output_frame_bgr.shape[0] != output_height):
                output_frame_bgr = cv2.resize(output_frame_bgr, 
                                            (output_width, output_height))

            # Write to ffmpeg
            try:
                if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                    ffmpeg_process.stdin.write(output_frame_bgr.tobytes())
                else:
                    if not stdin_broken:
                        print("\nffmpeg stdin closed")
                    stdin_broken = True
                    break
            except (BrokenPipeError, IOError) as e:
                if not stdin_broken:
                    print(f"\nffmpeg stdin error: {e}")
                stdin_broken = True
                break

            frame_idx += 1
            
            # Progress update
            if frame_idx % max(1, int(output_fps)) == 0 or frame_idx == 1:
                elapsed = time.time() - processing_start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps if fps > 0 else 0
                
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                
                sys.stdout.write(
                    f"\rFrame: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                    f"FPS: {fps:.1f} | ETA: {eta:.0f}s | "
                    f"People: {person_count} | Parts: {parts_count}    "
                )
                sys.stdout.flush()

    finally:
        cap.release()
        sys.stdout.write("\n")
        
        # Robust ffmpeg cleanup
        if ffmpeg_process.stdin:
            if not ffmpeg_process.stdin.closed and not stdin_broken:
                try:
                    ffmpeg_process.stdin.close()
                except (BrokenPipeError, IOError) as e:
                    print(f"Error closing ffmpeg stdin: {e}")
        
        stdout_data, stderr_data = None, None
        try:
            stdout_data, stderr_data = ffmpeg_process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            print("ffmpeg process timed out. Forcing termination.")
            ffmpeg_process.kill()
            stdout_data, stderr_data = ffmpeg_process.communicate()
        except ValueError as e:
            print(f"ValueError during ffmpeg.communicate(): {e}")
            if ffmpeg_process.poll() is None:
                ffmpeg_process.wait(timeout=5)
        except Exception as e:
            print(f"Error during ffmpeg.communicate(): {e}")
            if ffmpeg_process.poll() is None:
                ffmpeg_process.kill()
                ffmpeg_process.wait(timeout=5)
        
        processing_duration = time.time() - processing_start_time
        final_fps = frame_idx / processing_duration if processing_duration > 0 else 0
        
        print(f"Processed {frame_idx} frames in {processing_duration:.2f}s ({final_fps:.2f} FPS)")
        print(f"Total people detected: {total_people}")
        if frame_idx > 0:
            print(f"Average body parts per frame: {total_parts_detected/frame_idx:.1f}")
        
        if ffmpeg_process.returncode is not None and ffmpeg_process.returncode == 0:
            print("ffmpeg process completed successfully.")
        elif ffmpeg_process.returncode is not None:
            print(f"ffmpeg process exited with code {ffmpeg_process.returncode}")
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"✅ Robust enhanced segmentation output saved: {output_path} ({size_mb:.2f} MB)")
        else:
            print(f"❌ Output file not found: {output_path}")


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    try:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
        
        model = load_enhanced_segmentation_model(args.config, args.model, device)
        
        process_video(
            model=model, input_path=args.input, output_path=args.output,
            device=device, threshold=args.threshold, seg_alpha=args.seg_alpha,
            resize_factor=args.resize_factor, output_fps_target=args.fps,
            crf=args.crf, preset=args.preset
        )
        
        print("\n🎉 Robust enhanced segmentation processing complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()