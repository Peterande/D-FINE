#!/usr/bin/env python3
"""
Unified D-FINE model: detection (bbox) + segmentation + pose (keypoints).

Combines weights from separately-trained checkpoints that share
the same D-FINE-X backbone (HGNetv2 B5) + encoder (384-dim).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

repo_root = os.path.dirname(os.path.abspath(__file__))
for p in [repo_root, os.path.join(repo_root, "src"), os.path.join(repo_root, "segmentation_sivert")]:
    if p not in sys.path:
        sys.path.insert(0, p)


class UnifiedModel(nn.Module):
    """
    backbone  (B5)          — run once
    seg_head                — reads backbone features → masks
    encoder   (384-dim)     — run once
    det_decoder (DFINE)     — reads encoder features → bboxes + classes
    pose_decoder (DETRPose) — reads encoder features → keypoints
    """

    def __init__(self, backbone, encoder, det_decoder, seg_head, pose_decoder, pose_adapters=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.det_decoder = det_decoder
        self.seg_head = seg_head
        self.pose_decoder = pose_decoder
        self.pose_adapters = pose_adapters

    @torch.no_grad()
    def forward(self, x):
        backbone_features = self.backbone(x)

        # Segmentation (uses backbone features directly)
        seg_logits = self.seg_head(backbone_features)
        if isinstance(seg_logits, tuple):
            seg_logits = seg_logits[0]
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:],
            mode="bilinear", align_corners=False,
        )

        # Encoder (shared between det and pose)
        encoder_features = self.encoder(backbone_features)

        # Detection (bbox + class)
        det_out = self.det_decoder(encoder_features)

        # Pose (keypoints) - apply optional 1x1 adapters per level if provided
        feats_for_pose = encoder_features
        if getattr(self, "pose_adapters", None) is not None:
            feats_for_pose = [ad(f) for ad, f in zip(self.pose_adapters, encoder_features)]
        pose_out = self.pose_decoder(feats_for_pose)

        return {
            "pred_logits": det_out.get("pred_logits", det_out.get("pred", None)),
            "pred_boxes": det_out.get("pred_boxes", det_out.get("boxes", None)),
            "segmentation": seg_logits,
            "pred_keypoints": pose_out.get("pred_keypoints", pose_out.get("pose", None)),
        }


def load_unified_model(
    seg_checkpoint: str,
    pose_checkpoint: str,
    device: str = "cuda",
):
    """
    Args:
        seg_checkpoint:  Sivert's trained seg checkpoint (dfine_0.73.pth)
        pose_checkpoint: your trained pose best.pth
    """
    from src.core import YAMLConfig

    # ── 1) Build detection model (backbone + encoder + DFINE decoder) ──
    det_config = os.path.join(
        repo_root,
        "segmentation_sivert/base_dfine/dfine_hgnetv2_x_obj2coco.yml",
    )
    det_cfg = YAMLConfig(det_config)
    det_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    dfine_model = det_cfg.model

    # ── 2) Load seg checkpoint (has dfine_model.* + seg_head.*) ──
    seg_ckpt = torch.load(seg_checkpoint, map_location="cpu")
    seg_state = seg_ckpt.get("model_state_dict", seg_ckpt)
    seg_hyper = {}
    if isinstance(seg_ckpt, dict):
        seg_hyper = seg_ckpt.get("hyperparameters", seg_ckpt.get("model_hyperparameters", {})) or {}

    # Load backbone + encoder + det_decoder from seg checkpoint
    dfine_state = {}
    for k, v in seg_state.items():
        if k.startswith("dfine_model."):
            dfine_state[k.replace("dfine_model.", "")] = v
    dfine_model.load_state_dict(dfine_state, strict=False)
    print(f"Loaded backbone + encoder + det_decoder from {seg_checkpoint}")

    # ── 3) Build seg head and load weights ──
    from segmentation_sivert.models.standard import StandardSegmentationHead
    from segmentation_sivert.core.models import get_actual_backbone_channels

    backbone_channels = get_actual_backbone_channels(dfine_model)

    # Infer seg head feature_dim and dropout from checkpoint hyperparameters if available
    feature_dim = int(seg_hyper.get("seg_feature_dim", seg_hyper.get("feature_dim", 384)))
    dropout_rate = float(seg_hyper.get("seg_dropout", seg_hyper.get("dropout_rate", 0.1)))

    seg_head = StandardSegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=int(seg_hyper.get("num_classes", 7)),
        feature_dim=feature_dim,
        dropout_rate=dropout_rate,
    )
    
    seg_head_state = {}
    for k, v in seg_state.items():
        if k.startswith("seg_head."):
            seg_head_state[k.replace("seg_head.", "")] = v

    # Remap legacy flat-decoder keys to ConvBlock-wrapped keys.
    flat_to_wrapped = {
        "decoder.0.": "decoder.0.conv.",
        "decoder.1.": "decoder.0.norm.",
        "decoder.3.": "decoder.1.conv.",
        "decoder.4.": "decoder.1.norm.",
        "decoder.7.": "decoder.3.",
    }
    remapped_state = {}
    for k, v in seg_head_state.items():
        new_k = k
        for old_prefix, new_prefix in flat_to_wrapped.items():
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix, 1)
                break
        remapped_state[new_k] = v
    seg_head_state = remapped_state
    # Safe-load helper: only copy tensors whose shapes match the model's state_dict
    def _safe_load(module: nn.Module, state: dict):
        target_sd = module.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k not in target_sd:
                skipped.append((k, 'missing_in_target'))
                continue
            tgt = target_sd[k]
            if isinstance(v, torch.Tensor) and v.shape == tgt.shape:
                filtered[k] = v
            else:
                skipped.append((k, f'shape_mismatch source={tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)} target={tuple(tgt.shape)}'))
        module.load_state_dict(filtered, strict=False)
        return skipped, filtered

    skipped_keys, used = _safe_load(seg_head, seg_head_state)
    print(f"Loaded seg_head: applied {len(used)} tensors, skipped {len(skipped_keys)} tensors")
    if skipped_keys:
        print("Skipped seg_head keys (first 10):")
        for k, reason in skipped_keys[:10]:
            print(f"  - {k}: {reason}")

    # ── 4) Build pose decoder and load weights ──
    pose_config = os.path.join(
        repo_root,
        "pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml",
    )
    pose_cfg = YAMLConfig(pose_config)
    pose_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    pose_full_model = pose_cfg.model

    pose_ckpt = torch.load(pose_checkpoint, map_location="cpu")
    pose_state = pose_ckpt.get("model", pose_ckpt)
    pose_full_model.load_state_dict(pose_state, strict=False)
    pose_decoder = pose_full_model.decoder
    print(f"Loaded pose_decoder from {pose_checkpoint}")

    # Create 1x1 adapters if encoder channels != pose decoder expected dim
    # Infer encoder output channels by running a dummy input through backbone+encoder
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640)
        backbone_feats = dfine_model.backbone(dummy)
        enc_feats = dfine_model.encoder(backbone_feats)
    enc_ch = [int(f.shape[1]) for f in enc_feats]

    pose_expected_dim = None
    if hasattr(pose_decoder, "enc_output") and hasattr(pose_decoder.enc_output, "in_features"):
        pose_expected_dim = int(pose_decoder.enc_output.in_features)
    else:
        pose_expected_dim = int(getattr(pose_decoder, "hidden_dim", 256))

    pose_adapters = None
    if any(int(c) != int(pose_expected_dim) for c in enc_ch):
        print(f"🔌 Encoder channels {enc_ch} != pose decoder expected dim {pose_expected_dim} -> adding 1x1 conv adapters.")
        pose_adapters = nn.ModuleList([nn.Conv2d(int(c), int(pose_expected_dim), 1, bias=False) for c in enc_ch])

    # ── 5) Assemble unified model ──
    model = UnifiedModel(
        backbone=dfine_model.backbone,
        encoder=dfine_model.encoder,
        det_decoder=dfine_model.decoder,
        seg_head=seg_head,
        pose_decoder=pose_decoder,
    )
    model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Unified model: {total:,} parameters on {device}")
    return model


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seg-checkpoint", required=True)
    p.add_argument("--pose-checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model = load_unified_model(
        seg_checkpoint=args.seg_checkpoint,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
    )
    print("Unified model loaded OK")
