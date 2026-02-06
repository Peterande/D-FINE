"""
Hungarian matcher for DETRPose-style outputs (pose-only).

Adapted from:
  DETRPose/src/models/detrpose/matcher.py

Differences vs our DFINE matcher:
- no pred_boxes needed
- matching uses class cost + keypoint L1 (visible only) + OKS cost
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def _coco_sigmas(num_keypoints: int, device, dtype):
    if num_keypoints == 17:
        sigmas = np.array(
            [
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
            ],
            dtype=np.float32,
        ) / 10.0
    else:
        raise ValueError(f"Unsupported num_keypoints={num_keypoints} for COCO sigmas")
    return torch.tensor(sigmas, device=device, dtype=dtype)


def _area_from_targets_px(targets: list[dict], device, dtype) -> torch.Tensor:
    """Area per instance in pixel^2, using target boxes (cxcywh normalized)."""
    areas = []
    for t in targets:
        boxes = t["boxes"].to(device=device, dtype=dtype)
        if "size" in t:
            h, w = t["size"].tolist()
        else:
            w, h = t["orig_size"].tolist()
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        bw = (boxes[:, 2] * w).clamp(min=1.0)
        bh = (boxes[:, 3] * h).clamp(min=1.0)
        areas.append((bw * bh).to(dtype))
    return torch.cat(areas, dim=0) if areas else torch.zeros((0,), device=device, dtype=dtype)


class HungarianMatcherDETRPose(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        focal_alpha: float = 0.25,
        cost_keypoints: float = 1.0,
        cost_oks: float = 0.01,
        num_keypoints: int = 17,
    ):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_keypoints = float(cost_keypoints)
        self.cost_oks = float(cost_oks)
        self.focal_alpha = float(focal_alpha)
        self.num_keypoints = int(num_keypoints)

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        """
        outputs:
          - pred_logits: [B,Q,C] logits
          - pred_keypoints: [B,Q,2K] normalized (x,y) in [0,1] (resized image space)
        targets:
          - labels: [N]
          - keypoints: [N,K,3] (x_px,y_px,v)
          - boxes: [N,4] cxcywh normalized (for area only)
          - size/orig_size
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*Q,C]
        out_kpt = outputs["pred_keypoints"].flatten(0, 1)  # [B*Q,2K]

        tgt_ids = torch.cat([v["labels"] for v in targets], dim=0)
        tgt_kpt = torch.cat([v["keypoints"] for v in targets], dim=0).to(out_kpt.device)  # [sumN,K,3]
        # normalize gt to [0,1] (resized image space)
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=out_kpt.device, dtype=out_kpt.dtype)
        batch_idx = torch.cat([torch.full((len(t["labels"]),), i, device=out_kpt.device, dtype=torch.long) for i, t in enumerate(targets)], dim=0)
        wh = sizes[batch_idx].clamp(min=1.0)  # [sumN,2]
        Z_gt = (tgt_kpt[..., :2] / wh[:, None, :]).reshape(tgt_kpt.shape[0], -1)  # [sumN,2K]
        V_gt = (tgt_kpt[..., 2] > 0).to(out_kpt.dtype)  # [sumN,K]

        tgt_area = _area_from_targets_px(targets, device=out_kpt.device, dtype=out_kpt.dtype)  # [sumN]

        # class cost (focal-style)
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # [B*Q, sumN]

        # keypoint L1 visible-only
        cost_keypoints = torch.abs(out_kpt[:, None, :] - Z_gt[None, :, :])  # [B*Q,sumN,2K]
        vis_rep = V_gt.repeat_interleave(2, dim=1)  # [sumN,2K]
        cost_keypoints = (cost_keypoints * vis_rep[None, :, :]).sum(-1)  # [B*Q,sumN]

        # OKS cost (1-oks)
        sigmas = _coco_sigmas(self.num_keypoints, device=out_kpt.device, dtype=out_kpt.dtype)
        variances = (sigmas * 2) ** 2  # [K]
        kpt_preds = out_kpt.reshape(-1, self.num_keypoints, 2)
        kpt_gts = Z_gt.reshape(-1, self.num_keypoints, 2)
        d2 = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + (kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]) ** 2
        # convert normalized dist to pixel dist by multiplying by max(w,h) proxy.
        # Use sqrt(area) as a scale proxy; it cancels out with area in denom to keep units stable.
        scale = torch.sqrt(tgt_area.clamp(min=1.0))  # [sumN]
        d2 = d2 * (scale[None, :, None] ** 2)
        oks = torch.exp(-d2 / (tgt_area[None, :, None] * variances[None, None, :] * 2 + 1e-6)) * V_gt[None, :, :]
        oks = oks.sum(dim=-1) / (V_gt.sum(dim=-1).clamp(min=1.0)[None, :])
        cost_oks = 1.0 - oks.clamp(min=1e-6)

        C = self.cost_class * cost_class + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
        C = C.view(bs, num_queries, -1).cpu()
        sizes_split = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_split, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return {"indices": indices}

