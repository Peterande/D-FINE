#!/usr/bin/env python3
"""Build Option-1 single-pass merged checkpoint (shared X backbone)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


from tools.model_surgery.shared_arch import (
    SegmentationHead,
    SharedBackboneDualDecoder,
    filter_and_strip,
    infer_feature_dim_from_seg_ckpt,
    load_any_state,
    smart_load,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--det-ckpt", required=True)
    ap.add_argument("--pose-ckpt", required=True)
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=None)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--min-keep-ratio", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    for p in [repo, repo / "src", repo / "segmentation_sivert", repo / "pose_estimation_berna", repo / "tools"]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from src.core import YAMLConfig

    det_cfg = YAMLConfig(str(args.det_config))
    pose_cfg = YAMLConfig(str(args.pose_config))

    det_model = det_cfg.model
    pose_model = pose_cfg.model

    det_state_raw = load_any_state(args.det_ckpt)
    det_state = filter_and_strip("dfine_model.", det_state_raw) or det_state_raw
    smart_load(det_model, det_state, "det_model", min_keep_ratio=float(args.min_keep_ratio))

    pose_state = load_any_state(args.pose_ckpt)
    smart_load(pose_model, pose_state, "pose_model", min_keep_ratio=float(args.min_keep_ratio))

    seg_state_raw = load_any_state(args.seg_ckpt)
    seg_prefixed = {k: v for k, v in seg_state_raw.items() if k.startswith("seg_head.")}
    if not seg_prefixed:
        raise RuntimeError(
            "No seg_head.* keys found in --seg-ckpt. "
            "Use a segmentation-trained checkpoint that actually contains seg_head weights."
        )
    print(f"[seg_ckpt] found {len(seg_prefixed)} seg_head.* keys")

    det_model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, int(args.image_size), int(args.image_size))
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    feature_dim = int(args.seg_feature_dim) if args.seg_feature_dim is not None else infer_feature_dim_from_seg_ckpt(seg_prefixed, default=384)
    seg_head = SegmentationHead(
        in_channels_list=in_channels,
        num_classes=int(args.seg_num_classes),
        feature_dim=int(feature_dim),
        dropout_rate=float(args.seg_dropout),
    )
    seg_state = filter_and_strip("seg_head.", seg_state_raw)
    smart_load(seg_head, seg_state, "seg_head", min_keep_ratio=float(args.min_keep_ratio))

    model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model.decoder,
        seg_head=seg_head,
    ).eval()

    with torch.no_grad():
        out = model(dummy)
    print("[smoke]", {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "det_config": str(args.det_config),
                "pose_config": str(args.pose_config),
                "det_ckpt": str(args.det_ckpt),
                "pose_ckpt": str(args.pose_ckpt),
                "seg_ckpt": str(args.seg_ckpt),
                "seg_feature_dim": int(feature_dim),
                "seg_num_classes": int(args.seg_num_classes),
                "seg_dropout": float(args.seg_dropout),
                "image_size": int(args.image_size),
            },
        },
        str(out_path),
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()