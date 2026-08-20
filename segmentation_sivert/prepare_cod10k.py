#!/usr/bin/env python3
"""
Prepare COD10K-v3 dataset for the segmentation training framework.

Converts from:
  COD10K-v3/Train/Image/  + COD10K-v3/Train/GT_Object/
  COD10K-v3/Test/Image/   + COD10K-v3/Test/GT_Object/

To:
  cod10k/images/train/ + cod10k/masks/train/
  cod10k/images/val/   + cod10k/masks/val/

Masks are converted from 0/255 binary to 0/1 (0=background, 1=camouflaged).
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def prepare_split(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Processing {len(images)} images from {src_img_dir.name}...")

    missing_masks = 0
    for img_name in tqdm(images):
        mask_name = Path(img_name).stem + '.png'
        src_mask = src_mask_dir / mask_name
        if not src_mask.exists():
            missing_masks += 1
            continue

        # Symlink image using absolute path so symlink resolves correctly
        dst_img = dst_img_dir / img_name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink((src_img_dir / img_name).resolve(), dst_img)

        # Convert mask 0/255 → 0/1
        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8)
        cv2.imwrite(str(dst_mask_dir / mask_name), mask_binary)

    if missing_masks:
        print(f"  Warning: {missing_masks} images had no matching mask and were skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/COD10K-v3',
                        help='Path to extracted COD10K-v3 directory')
    parser.add_argument('--dst', default='data/cod10k',
                        help='Output directory for prepared dataset')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    assert (src / 'Train' / 'Image').exists(), f"Not found: {src}/Train/Image"
    assert (src / 'Test' / 'Image').exists(), f"Not found: {src}/Test/Image"

    print("Preparing train split...")
    prepare_split(
        src / 'Train' / 'Image',
        src / 'Train' / 'GT_Object',
        dst / 'images' / 'train',
        dst / 'masks' / 'train',
    )

    print("Preparing val split...")
    prepare_split(
        src / 'Test' / 'Image',
        src / 'Test' / 'GT_Object',
        dst / 'images' / 'val',
        dst / 'masks' / 'val',
    )

    train_count = len(list((dst / 'images' / 'train').iterdir()))
    val_count = len(list((dst / 'images' / 'val').iterdir()))
    print(f"\nDone. {train_count} train / {val_count} val images ready at: {dst}")


if __name__ == '__main__':
    main()
