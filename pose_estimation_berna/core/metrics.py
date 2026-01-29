#!/usr/bin/env python3
"""
Pose metrics for COCO keypoints.

Lightweight OKS tracking on matched pairs (mainly for debugging).
Full COCO AP should be computed via COCO evaluator when integrated.
"""

from typing import Dict, List

import torch


# COCO sigmas for the 17 keypoints
COCO_SIGMAS = torch.tensor(
    [
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089,
    ],
    dtype=torch.float32,
)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class PoseMetricsTracker:
    def __init__(self, num_keypoints: int = 17):
        self.num_keypoints = int(num_keypoints)
        self.reset()

    def reset(self):
        self._oks: List[float] = []

    @torch.no_grad()
    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices):
        """
        Args:
          - outputs["pred_keypoints"]: [B,Q,K,3] (x_rel,y_rel,vis_logit) bbox-relative
          - outputs["pred_boxes"]: [B,Q,4] cxcywh normalized
          - targets[i]["keypoints"]: [N,K,3] (x_px,y_px,v)
          - targets[i]["boxes"]: [N,4] cxcywh normalized
          - indices: list of (src_idx, tgt_idx) from Hungarian matching
        """
        if "pred_keypoints" not in outputs:
            return

        pred_kpts = outputs["pred_keypoints"]
        pred_boxes = outputs["pred_boxes"]

        sigmas = COCO_SIGMAS.to(pred_kpts.device)[: self.num_keypoints]
        vars_ = (sigmas * 2) ** 2  # [K]

        for b, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue

            pb = pred_boxes[b, src]  # [M,4]
            pb_xyxy = _cxcywh_to_xyxy(pb)
            pb_wh = (pb_xyxy[:, 2:] - pb_xyxy[:, :2]).clamp(min=1e-6)  # [M,2]

            pk = pred_kpts[b, src]  # [M,K,3]
            pk_xy = pb_xyxy[:, None, :2] + pk[..., :2] * pb_wh[:, None, :]  # [M,K,2] (image-normalized)

            tk = targets[b]["keypoints"][tgt].to(pk_xy.device)  # [M,K,3]
            if "size" in targets[b]:
                h, w = targets[b]["size"].tolist()
            else:
                w, h = targets[b]["orig_size"].tolist()
            wh = torch.tensor([w, h], device=pk_xy.device, dtype=pk_xy.dtype)
            tk_xy = tk[..., :2] / wh[None, None, :]
            v = (tk[..., 2] > 0).to(pk_xy.dtype)  # [M,K]

            tb = targets[b]["boxes"][tgt].to(pk_xy.device)
            tb_xyxy = _cxcywh_to_xyxy(tb)
            area = ((tb_xyxy[:, 2] - tb_xyxy[:, 0]) * (tb_xyxy[:, 3] - tb_xyxy[:, 1])).clamp(min=1e-6)

            dx2 = (pk_xy[..., 0] - tk_xy[..., 0]) ** 2
            dy2 = (pk_xy[..., 1] - tk_xy[..., 1]) ** 2
            e = (dx2 + dy2) / (vars_[None, :] * area[:, None] * 2 + 1e-6)

            oks = torch.exp(-e) * v
            denom = v.sum(dim=1).clamp(min=1.0)
            oks = oks.sum(dim=1) / denom  # [M]
            self._oks.extend(oks.detach().cpu().tolist())

    def compute(self) -> Dict[str, float]:
        if not self._oks:
            return {"OKS": 0.0}
        return {"OKS": float(sum(self._oks) / len(self._oks))}


def create_metrics_tracker(num_keypoints: int = 17) -> PoseMetricsTracker:
    """Factory function for compatibility with the segmentation_sivert template."""
    return PoseMetricsTracker(num_keypoints=num_keypoints)


