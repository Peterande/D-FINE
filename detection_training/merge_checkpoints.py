#!/usr/bin/env python3
"""
Model Surgery: merge new soldier-finetuned backbone into original combined checkpoint.

Strategy: BACKBONE-ONLY transfer from the new soldier-finetuned model.
  - backbone   → from new fine-tune (learned to find camouflaged soldiers)
  - encoder    → from original (keeps encoder features compatible with pose decoder)
  - det_decoder → from original (80-class COCO, detects 'person')
  - pose_decoder → from original (unchanged, compatible with original encoder)
  - seg_head   → from original (unchanged)

Why backbone-only: the pose decoder was trained on the original encoder's feature
representations. Swapping encoder+decoder breaks pose (encoder features change →
pose decoder hallucinates keypoints everywhere). Backbone-only surgery preserves
all downstream compatibility while still improving low-level feature extraction
for camouflaged persons.
"""

import argparse
import torch
from pathlib import Path


def load_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            m = ckpt["ema"].get("module")
            if m is not None:
                return m
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError(f"Cannot extract state dict from {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-det", default="outputs/soldier_finetune/best_stg2.pth",
                    help="New soldier-finetuned detection checkpoint")
    ap.add_argument("--orig", default="Traning/best_modelsurgery.pth",
                    help="Original combined checkpoint (all heads)")
    ap.add_argument("--out", default="outputs/soldier_finetune/merged_surgery.pth",
                    help="Output merged checkpoint path")
    args = ap.parse_args()

    print(f"Loading new detection checkpoint: {args.new_det}")
    new_state = load_state(args.new_det)

    print(f"Loading original surgery checkpoint: {args.orig}")
    orig_state = load_state(args.orig)

    merged = {}
    counts = {"backbone_new": 0, "encoder_orig": 0, "det_decoder_orig": 0,
              "pose_decoder_orig": 0, "seg_head_orig": 0}

    # 1. Backbone: from NEW model — learns camouflage-aware low-level features
    for k, v in new_state.items():
        if k.startswith("backbone."):
            merged[k] = v
            counts["backbone_new"] += 1

    # 2-5. Everything else: from ORIGINAL — preserves encoder↔pose_decoder compatibility
    for k, v in orig_state.items():
        if k.startswith("encoder."):
            merged[k] = v
            counts["encoder_orig"] += 1
        elif k.startswith("det_decoder."):
            merged[k] = v
            counts["det_decoder_orig"] += 1
        elif k.startswith("pose_decoder."):
            merged[k] = v
            counts["pose_decoder_orig"] += 1
        elif k.startswith("seg_head."):
            merged[k] = v
            counts["seg_head_orig"] += 1

    print("\n=== Merge summary ===")
    for part, n in counts.items():
        print(f"  {part}: {n} tensors")
    print(f"  TOTAL: {sum(counts.values())} tensors")

    score_key = "det_decoder.enc_score_head.weight"
    if score_key in merged:
        print(f"\nDetection class head shape: {merged[score_key].shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged}, str(out_path))
    print(f"\nSaved merged checkpoint → {out_path}")


if __name__ == "__main__":
    main()
