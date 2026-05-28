#!/usr/bin/env python3
"""
Video inference with trained camouflage segmentation model.
Overlays segmentation mask on each frame and writes output video.

Usage:
  python infer_video.py --input "Sekvens 5.mp4" --checkpoint outputs/camo_finetune/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.append('src')

from core.models import load_pretrained_dfine, get_actual_backbone_channels, create_combined_model
from models import create_segmentation_head


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Read config from checkpoint
    config = checkpoint['config']
    tier = checkpoint['args']['tier']
    num_classes = config['dataset']['num_classes']

    seg_head_config = config.get('model', {}).get('segmentation_head', {})
    feature_dim = seg_head_config.get('feature_dim', 384)
    dropout_rate = seg_head_config.get('dropout_rate', 0.15)

    dfine_config  = config['dfine']['config_path']
    dfine_checkpoint = config['dfine']['checkpoint_path']

    dfine_model = load_pretrained_dfine(dfine_config, dfine_checkpoint)
    backbone_channels = get_actual_backbone_channels(dfine_model)

    seg_head = create_segmentation_head(
        tier=tier,
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    )

    model = create_combined_model(dfine_model=dfine_model, seg_head=seg_head, freeze_detection=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded {tier} model — {num_classes} classes, mIoU at save: {checkpoint.get('best_miou', 'N/A'):.4f}")
    return model, num_classes


def preprocess_frame(frame: np.ndarray, image_size: int = 640):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def filter_person_blobs(mask: np.ndarray, frame_h: int, frame_w: int,
                        min_area_frac: float = 0.002,
                        max_aspect: float = 6.0) -> np.ndarray:
    """
    Keep only blobs that are plausibly person-shaped:
    - Large enough (> min_area_frac of frame)
    - Not too elongated (aspect ratio < max_aspect) — removes sticks/branches
    """
    min_area = int(frame_h * frame_w * min_area_frac)
    out = np.zeros_like(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > max_aspect:
            continue

        out[labels == i] = 1

    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(0, 255, 80)):
    """Overlay binary segmentation mask on frame with colored highlight."""
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = frame.copy()
    overlay[mask_resized == 1] = (
        overlay[mask_resized == 1] * (1 - alpha) +
        np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)

    # Draw contour for crisp edge
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, num_classes = load_model(args.checkpoint, device)

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps — {total_frames} frames")

    # Output path
    input_path = Path(args.input)
    out_path = args.output or str(input_path.parent / (input_path.stem + '_segmented.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    smooth_prob = None  # running average of per-pixel camouflage probability

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Inferring"):
            ret, frame = cap.read()
            if not ret:
                break

            inp = preprocess_frame(frame, image_size=args.image_size).to(device)
            outputs = model(inp)
            seg_logits = outputs['segmentation']  # [1, C, H, W]

            # Softmax prob for class 1 (camouflaged), at model resolution
            prob = torch.softmax(seg_logits, dim=1)[0, 1].cpu().numpy()  # [H_model, W_model]

            # Temporal smoothing: EMA across frames
            if smooth_prob is None:
                smooth_prob = prob
            else:
                smooth_prob = args.temporal_alpha * prob + (1 - args.temporal_alpha) * smooth_prob

            # Threshold smoothed probability
            mask = (smooth_prob > args.threshold).astype(np.uint8)

            # Filter to person-shaped blobs only
            mask = filter_person_blobs(mask, mask.shape[0], mask.shape[1])

            result = overlay_mask(frame, mask, alpha=args.alpha)

            if args.heatmap:
                prob_resized = cv2.resize(smooth_prob, (w, h))
                heatmap = cv2.applyColorMap((prob_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                result = cv2.addWeighted(result, 0.7, heatmap, 0.3, 0)

            writer.write(result)

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--checkpoint', default='outputs/camo_finetune/best_model.pth')
    parser.add_argument('--output', default=None, help='Output video path (default: input_segmented.mp4)')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--alpha', type=float, default=0.5, help='Mask overlay transparency')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for mask')
    parser.add_argument('--temporal-alpha', type=float, default=0.4,
                        help='EMA weight for current frame (0=fully smooth, 1=no smoothing)')
    parser.add_argument('--heatmap', action='store_true', help='Also overlay confidence heatmap')
    return parser.parse_args()


if __name__ == '__main__':
    run_inference(parse_args())
