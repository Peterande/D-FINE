#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict
import torch


@torch.no_grad()
def keypoints_to_box_xyxy(kpts_xy: torch.Tensor) -> torch.Tensor:
    """
    kpts_xy: [B,Q,K,2] in pixel coords
    returns: [B,Q,4] xyxy
    """
    x = kpts_xy[..., 0]
    y = kpts_xy[..., 1]
    x1 = x.min(dim=-1).values
    y1 = y.min(dim=-1).values
    x2 = x.max(dim=-1).values
    y2 = y.max(dim=-1).values
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def expand_xyxy(boxes: torch.Tensor, ex: float = 0.15, ey_top: float = 0.20, ey_bot: float = 0.30) -> torch.Tensor:
    """
    boxes: [B,Q,4]
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    x1 = x1 - ex * bw
    x2 = x2 + ex * bw
    y1 = y1 - ey_top * bh
    y2 = y2 + ey_bot * bh
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def union_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(a[..., 0], b[..., 0])
    y1 = torch.minimum(a[..., 1], b[..., 1])
    x2 = torch.maximum(a[..., 2], b[..., 2])
    y2 = torch.maximum(a[..., 3], b[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def clamp_xyxy(boxes: torch.Tensor, orig_wh: torch.Tensor) -> torch.Tensor:
    """
    boxes: [B,Q,4]
    orig_wh: [B,2] -> (w,h)
    """
    w = orig_wh[:, 0].view(-1, 1)
    h = orig_wh[:, 1].view(-1, 1)
    x1 = boxes[..., 0].clamp(min=0.0)
    y1 = boxes[..., 1].clamp(min=0.0)
    x2 = boxes[..., 2].clamp(min=0.0)
    y2 = boxes[..., 3].clamp(min=0.0)
    x1 = torch.minimum(x1, w)
    x2 = torch.minimum(x2, w)
    y1 = torch.minimum(y1, h)
    y2 = torch.minimum(y2, h)
    x_min = torch.minimum(x1, x2)
    y_min = torch.minimum(y1, y2)
    x_max = torch.maximum(x1, x2)
    y_max = torch.maximum(y1, y2)
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


@torch.no_grad()
def fuse_det_and_pose_boxes(
    det_boxes_xyxy: torch.Tensor,       # [B,Q,4]
    pose_keypoints_xy_norm: torch.Tensor,  # [B,Q,K,2] normalized
    orig_wh: torch.Tensor,              # [B,2] (w,h)
    ex: float = 0.15,
    ey_top: float = 0.20,
    ey_bot: float = 0.30,
) -> torch.Tensor:
    """
    Return fused full-body box = union(det_box, expanded_pose_box).
    """
    kpts_px = pose_keypoints_xy_norm * orig_wh[:, None, None, :]
    pose_box = keypoints_to_box_xyxy(kpts_px)
    pose_box = expand_xyxy(pose_box, ex=ex, ey_top=ey_top, ey_bot=ey_bot)
    fused = union_xyxy(det_boxes_xyxy, pose_box)
    fused = clamp_xyxy(fused, orig_wh)
    return fused
