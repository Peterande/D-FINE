#!/usr/bin/env python3
"""
Prepare CAMO dataset for the segmentation training framework.

Source structure:
  CAMO-V.1.0-CVIU2019/Images/Train/  (1000 .jpg)
  CAMO-V.1.0-CVIU2019/Images/Test/   (250  .jpg)
  CAMO-V.1.0-CVIU2019/GT/            (1250 .png, 0/255 binary)

Output:
  camo/images/train/ + camo/masks/train/
  camo/images/val/   + camo/masks/val/

Masks converted from 0/255 → 0/1.
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


CAMO_ROOT = "/home/berna/.cache/kagglehub/datasets/ivanomelchenkoim11/camo-dataset/versions/1/CAMO-V.1.0-CVIU2019"


def prepare_split(img_dir, gt_dir, dst_img_dir, dst_mask_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Processing {len(images)} images...")

    missing = 0
    for img_name in tqdm(images):
        stem = Path(img_name).stem
        mask_name = stem + '.png'
        src_mask = gt_dir / mask_name

        if not src_mask.exists():
            missing += 1
            continue

        # Symlink image with absolute path
        dst_img = dst_img_dir / img_name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink((img_dir / img_name).resolve(), dst_img)

        # Convert mask 0/255 → 0/1
        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8)
        cv2.imwrite(str(dst_mask_dir / mask_name), mask_binary)

    if missing:
        print(f"  Warning: {missing} images had no matching mask and were skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=CAMO_ROOT)
    parser.add_argument('--dst', default='data/camo')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    gt_dir = src / 'GT'

    print("Preparing train split...")
    prepare_split(src / 'Images' / 'Train', gt_dir, dst / 'images' / 'train', dst / 'masks' / 'train')

    print("Preparing val split...")
    prepare_split(src / 'Images' / 'Test', gt_dir, dst / 'images' / 'val', dst / 'masks' / 'val')

    train_count = len(list((dst / 'images' / 'train').iterdir()))
    val_count = len(list((dst / 'images' / 'val').iterdir()))
    print(f"\nDone. {train_count} train / {val_count} val images ready at: {dst}")


if __name__ == '__main__':
    main()
