#!/usr/bin/env python3
"""Shared architecture + loading utilities for model-surgery scripts."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 6, 12)):
        super().__init__()
        dils = list(dilations)
        branch = out_channels // len(dils)

        self.convs = nn.ModuleList()
        for d in dils:
            if int(d) == 1:
                conv = nn.Conv2d(in_channels, branch, 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, branch, 3, padding=int(d), dilation=int(d), bias=False)
            self.convs.append(nn.Sequential(conv, nn.BatchNorm2d(branch), nn.ReLU(inplace=True)))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch, 1, bias=False),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
        )

        total = branch * (len(dils) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [m(x) for m in self.convs]
        g = self.global_pool(x)
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class FPN(nn.Module):
    def __init__(self, in_channels_list: Iterable[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(int(c), int(out_channels), 1, bias=False) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(out_channels), int(out_channels), 3, padding=1, bias=False),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True),
            )
            for _ in self.lateral_convs
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(self.lateral_convs)))

    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        target = outs[0].shape[-2:]
        fused = []
        for i, f in enumerate(outs):
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            fused.append(f * self.fusion_weights[i])
        return sum(fused)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, num_classes: int, feature_dim: int, dropout_rate: float):
        super().__init__()
        self.fpn = FPN(in_channels_list, feature_dim)
        self.aspp = ASPP(feature_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout_rate)),
            nn.Conv2d(feature_dim // 2, int(num_classes), 1),
        )

    def forward(self, feats):
        return self.decoder(self.aspp(self.fpn(feats)))


class SharedBackboneDualDecoder(nn.Module):
    """Single-pass combined model with explicit output naming.

    Naming convention: only prefixed model outputs to avoid ambiguity.
      - det.* from detection decoder
      - pose.* from pose decoder
      - seg.logits from segmentation head
    """

    def __init__(self, backbone, encoder, det_decoder, pose_decoder, seg_head):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.det_decoder = det_decoder
        self.pose_decoder = pose_decoder
        self.seg_head = seg_head

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        enc = self.encoder(feats)

        det_out = self.det_decoder(enc, targets)
        pose_out = self.pose_decoder(enc, targets)

        seg_logits = self.seg_head(feats)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = {"seg.logits": seg_logits}
        if isinstance(det_out, dict):
            for k, v in det_out.items():
                out[f"det.{k}"] = v
        if isinstance(pose_out, dict):
            for k, v in pose_out.items():
                out[f"pose.{k}"] = v
        return out


def load_any_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def filter_and_strip(prefix: str, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def smart_load(module: nn.Module, state: Dict[str, torch.Tensor], name: str, min_keep_ratio: float = 0.90) -> None:
    msd = module.state_dict()
    kept = {k: v for k, v in state.items() if k in msd and getattr(v, "shape", None) == getattr(msd[k], "shape", None)}
    missing, unexpected = module.load_state_dict(kept, strict=False)
    ratio = float(len(kept) / max(1, len(state)))
    print(f"[{name}] kept={len(kept)}/{len(state)} ({ratio:.1%}) missing={len(missing)} unexpected={len(unexpected)}")
    if ratio < float(min_keep_ratio):
        raise RuntimeError(
            f"[{name}] low keep ratio {ratio:.1%} (< {float(min_keep_ratio):.1%}). "
            f"Checkpoint/config mismatch likely."
        )


def infer_feature_dim_from_seg_ckpt(seg_prefixed_state: Dict[str, torch.Tensor], default: int) -> int:
    key = "seg_head.fpn.lateral_convs.0.weight"
    if key in seg_prefixed_state and seg_prefixed_state[key].ndim >= 1:
        return int(seg_prefixed_state[key].shape[0])
    return int(default)