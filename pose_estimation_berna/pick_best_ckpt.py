#!/usr/bin/env python3
"""
Pick the best checkpoint by COCO-AP(kpt) by evaluating a set of .pth files on COCO val.

Usage example:
  python3 pose_estimation_berna/pick_best_ckpt.py \
    --tier standard --config-dir pose_estimation_berna/configs \
    --ckpt-dir outputs/standard/standard \
    --out outputs/standard/standard/best_by_coco_ap.pth
"""

import argparse
import os
import sys
from pathlib import Path
import shutil

import torch

# Ensure local imports work when run as a script
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../pose_estimation_berna
repo_root_dir = os.path.dirname(current_dir)  # .../D-FINE-NEWBRINGER
src_dir = os.path.join(repo_root_dir, "src")
for p in [repo_root_dir, current_dir, src_dir]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from pose_estimation_berna.eval_coco_kpt import evaluate_one  # noqa: E402
from pose_estimation_berna.train import load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="standard", choices=["lightweight", "standard", "advanced"])
    p.add_argument("--config-dir", default="pose_estimation_berna/configs")
    p.add_argument("--ckpt-dir", required=True, help="Directory containing checkpoints (best/last/checkpoint_epoch_*.pth)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None, help="Debug speed knob; omit for full-val.")
    p.add_argument("--out", default=None, help="Where to copy the best checkpoint (default: <ckpt-dir>/best_by_coco_ap.pth)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.tier, args.config_dir)
    cfg["tier"] = args.tier
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(repo_root_dir) / ckpt_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt-dir not found: {ckpt_dir}")

    out_path = Path(args.out) if args.out else (ckpt_dir / "best_by_coco_ap.pth")
    if not out_path.is_absolute():
        out_path = Path(repo_root_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer evaluating real training checkpoints (epoch + last), but include existing best for comparison.
    patterns = ["checkpoint_epoch_*.pth", "last.pth", "best.pth"]
    ckpts = []
    for pat in patterns:
        ckpts.extend(sorted(ckpt_dir.glob(pat)))
    # de-dup
    seen = set()
    uniq = []
    for p in ckpts:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append(p)
    ckpts = uniq

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    print(f"🚀 Picking best by COCO-AP(kpt) on device: {device}")
    print(f"📦 Candidates: {len(ckpts)}")

    best_ap = None
    best_path = None
    for p in ckpts:
        ap = evaluate_one(
            cfg=cfg,
            checkpoint_path=str(p),
            device=device,
            num_workers=args.num_workers,
            max_val_samples=args.max_val_samples,
            max_val_steps=args.max_val_steps,
        )
        print(f"  - {p.name}: COCO-AP(kpt)={ap:.3f}")
        if best_ap is None or (ap == ap and float(ap) > float(best_ap)):  # ap==ap filters NaN
            best_ap = float(ap)
            best_path = p

    if best_path is None:
        raise RuntimeError("Could not pick a best checkpoint (all NaN?)")

    shutil.copy2(best_path, out_path)
    print(f"🏆 Best: {best_path} (COCO-AP(kpt)={best_ap:.3f})")
    print(f"✅ Copied to: {out_path}")


if __name__ == "__main__":
    main()


