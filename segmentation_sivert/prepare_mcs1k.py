#!/usr/bin/env python3
"""
Prepare MCS1K dataset for the segmentation training framework.

Source:
  dataset-splitM/Training/images/ + dataset-splitM/Training/GT/
  dataset-splitM/Testing/images/  + dataset-splitM/Testing/GT/

Output:
  mcs1k/images/train/ + mcs1k/masks/train/
  mcs1k/images/val/   + mcs1k/masks/val/

Masks thresholded at 127 → binary 0/1.
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

MCS1K_ROOT = "/home/berna/.cache/kagglehub/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k/versions/1/dataset-splitM"


def prepare_split(img_dir, gt_dir, dst_img_dir, dst_mask_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Processing {len(images)} images...")

    missing = 0
    for img_name in tqdm(images):
        stem = Path(img_name).stem
        # GT has same filename as image
        mask_name = stem + '.jpg'
        src_mask = gt_dir / mask_name
        if not src_mask.exists():
            # Try png
            src_mask = gt_dir / (stem + '.png')
        if not src_mask.exists():
            missing += 1
            continue

        dst_img = dst_img_dir / img_name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink((img_dir / img_name).resolve(), dst_img)

        # Threshold to binary 0/1
        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8)
        cv2.imwrite(str(dst_mask_dir / (stem + '.png')), mask_binary)

    if missing:
        print(f"  Warning: {missing} images had no matching mask and were skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=MCS1K_ROOT)
    parser.add_argument('--dst', default='data/mcs1k')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print("Preparing train split...")
    prepare_split(src / 'Training' / 'images', src / 'Training' / 'GT',
                  dst / 'images' / 'train', dst / 'masks' / 'train')

    print("Preparing val split...")
    prepare_split(src / 'Testing' / 'images', src / 'Testing' / 'GT',
                  dst / 'images' / 'val', dst / 'masks' / 'val')

    train_count = len(list((dst / 'images' / 'train').iterdir()))
    val_count = len(list((dst / 'images' / 'val').iterdir()))
    print(f"\nDone. {train_count} train / {val_count} val images ready at: {dst}")


if __name__ == '__main__':
    main()
