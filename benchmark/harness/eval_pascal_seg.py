#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.harness.infer_crosshair import build_model_from_merged
from segmentation_sivert.core.datasets import PascalPersonPartsDataset


LABELS = ["background", "head", "torso", "arms", "hands", "legs", "feet"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--merged-ckpt", required=True)
    parser.add_argument("--seg-head-ckpt", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    model, _ = build_model_from_merged(det_config=args.det_config, pose_config=args.pose_config,
        merged_ckpt=args.merged_ckpt, seg_num_classes=7, seg_feature_dim=384,
        seg_dropout=0.1, image_size=640)
    if args.seg_head_ckpt:
        checkpoint = torch.load(args.seg_head_ckpt, map_location="cpu", weights_only=False)
        model.seg_head.load_state_dict(checkpoint["seg_head"], strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    dataset = PascalPersonPartsDataset(root_dir=args.dataset, split="val", image_size=640,
                                       num_classes=7, tier="standard")
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    confusion = np.zeros((7, 7), dtype=np.int64)
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model.backbone(images)
            logits = model.seg_head(features)
            logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(1).cpu().numpy().reshape(-1)
            target = targets.numpy().reshape(-1)
            valid = (target >= 0) & (target < 7)
            confusion += np.bincount(7 * target[valid] + predictions[valid], minlength=49).reshape(7, 7)
            if (index + 1) % 50 == 0:
                print(f"batches={index + 1}/{len(loader)}", flush=True)
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    result = {
        "schema_version": 1,
        "samples": len(dataset),
        "labels": LABELS,
        "confusion_matrix": confusion.tolist(),
        "per_class_iou": {name: float(value) for name, value in zip(LABELS, iou)},
        "miou": float(iou.mean()),
        "pixel_accuracy": float(intersection.sum() / confusion.sum()),
        "preprocessing": "training-compatible RGB direct-resize 640, /255, ImageNet mean/std",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"miou": result["miou"], "pixel_accuracy": result["pixel_accuracy"], "per_class_iou": result["per_class_iou"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
