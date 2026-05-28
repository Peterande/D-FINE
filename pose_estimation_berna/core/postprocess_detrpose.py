#!/usr/bin/env python3
"""
Post-processing for DETRPose-style outputs (pose-only).

Expected model outputs:
  - pred_logits:    [B, Q, C] (logits)
  - pred_keypoints: [B, Q, 2K] (x,y) normalized in [0,1] in resized image space

This produces a DFINEPostProcessor-like dict so existing COCOeval plumbing can run:
  - boxes:   [B, top, 4] xyxy in original pixel coords (tight box around keypoints)
  - scores:  [B, top]
  - labels:  [B, top] (MSCOCO category_id if remap enabled)
  - keypoints: [B, top, K, 3] in original pixel coords, with v=1
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _remap_labels_to_coco_category(labels: torch.Tensor) -> torch.Tensor:
    # D-FINE labels are contiguous 0..79; COCO evaluator expects category_id (person==1).
    from src.data.dataset import mscoco_label2category

    flat = labels.flatten()
    mapped = torch.tensor([mscoco_label2category[int(x.item())] for x in flat], device=labels.device)
    return mapped.view_as(labels)


class DETRPosePostProcessor(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        num_keypoints: int = 17,
        num_top_queries: int = 300,
        remap_mscoco_category: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_keypoints = int(num_keypoints)
        self.num_top_queries = int(num_top_queries)
        self.remap_mscoco_category = bool(remap_mscoco_category)

    @torch.no_grad()
    def forward(self, outputs: dict, orig_target_sizes: torch.Tensor) -> list[dict]:
        logits = outputs["pred_logits"]  # [B,Q,C]
        kpts = outputs["pred_keypoints"]  # [B,Q,2K]
        if kpts.ndim != 3:
            raise ValueError(f"DETRPosePostProcessor expects pred_keypoints [B,Q,2K], got shape={tuple(kpts.shape)}")

        B, Q, _ = kpts.shape
        K = self.num_keypoints
        if int(kpts.shape[-1]) != int(K * 2):
            raise ValueError(f"pred_keypoints last dim must be 2K={K*2}, got {int(kpts.shape[-1])}")

        # DETRPose repo selects top-k over (Q*C). That is fine when labels are stable.
        # For person-only training, it is more robust to select per-query best class first,
        # then pick top-k queries. This avoids accidentally selecting the "wrong class slot"
        # for an otherwise good pose query early in training.
        scores_all = torch.sigmoid(logits)  # [B,Q,C]
        per_q_scores, per_q_labels = scores_all.max(dim=-1)  # [B,Q]
        topk = min(int(self.num_top_queries), int(Q))
        scores, q_index = torch.topk(per_q_scores, topk, dim=-1)  # [B,top]
        labels = per_q_labels.gather(dim=1, index=q_index)  # [B,top]

        kpts_sel = kpts.gather(dim=1, index=q_index.unsqueeze(-1).repeat(1, 1, kpts.shape[-1]))  # [B,top,2K]
        kpts_xy = kpts_sel.view(B, -1, K, 2)  # normalized in resized image space

        # Convert normalized coords to original pixel coords:
        # resized pixel = norm * image_size ; original pixel = resized / scale = norm * orig_size
        orig_wh = orig_target_sizes.to(dtype=kpts_xy.dtype)  # [B,2] (w,h)
        kpts_px = kpts_xy * orig_wh[:, None, None, :]  # [B,top,K,2]

        # Build a robust box around keypoints (in original px).
        # Using strict min/max is very sensitive to a single outlier keypoint.
        x = kpts_px[..., 0]
        y = kpts_px[..., 1]
        try:
            x1 = torch.quantile(x, 0.02, dim=-1)
            y1 = torch.quantile(y, 0.02, dim=-1)
            x2 = torch.quantile(x, 0.98, dim=-1)
            y2 = torch.quantile(y, 0.98, dim=-1)
        except Exception:
            x1 = x.min(dim=-1).values
            y1 = y.min(dim=-1).values
            x2 = x.max(dim=-1).values
            y2 = y.max(dim=-1).values

        # small padding for visualization stability
        pad = 0.05
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)
        x1 = x1 - pad * bw
        x2 = x2 + pad * bw
        y1 = y1 - pad * bh
        y2 = y2 + pad * bh

        w_img = orig_wh[:, 0].view(B, 1)
        h_img = orig_wh[:, 1].view(B, 1)
        x1 = torch.clamp(x1, min=0.0)
        y1 = torch.clamp(y1, min=0.0)
        x2 = torch.clamp(x2, min=0.0)
        y2 = torch.clamp(y2, min=0.0)
        x1 = torch.minimum(x1, w_img)
        x2 = torch.minimum(x2, w_img)
        y1 = torch.minimum(y1, h_img)
        y2 = torch.minimum(y2, h_img)
        x_min = torch.minimum(x1, x2)
        y_min = torch.minimum(y1, y2)
        x_max = torch.maximum(x1, x2)
        y_max = torch.maximum(y1, y2)
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # [B,top,4]

        # Keypoints: [x,y,v] where v is a visibility flag; set to 1 for all predicted keypoints.
        v = torch.ones_like(kpts_px[..., :1])
        keypoints = torch.cat([kpts_px, v], dim=-1)  # [B,top,K,3]

        if self.remap_mscoco_category:
            # For person-only pose training we typically use 1 or 2 "classes" just to keep indices valid.
            # In that case, mapping through mscoco_label2category (0..79 -> category_id) is not meaningful
            # for label==1 (it becomes bicycle==2). That can cause almost all predictions to be filtered out
            # downstream (we keep only category_id==1 for keypoint eval).
            #
            # Make eval robust: force all predictions to COCO category_id=1 (person) when num_classes <= 2.
            if int(self.num_classes) <= 2:
                labels = torch.ones_like(labels)
            else:
                labels = _remap_labels_to_coco_category(labels)

        out: list[dict] = []
        for bi in range(B):
            out.append({"labels": labels[bi], "boxes": boxes[bi], "scores": scores[bi], "keypoints": keypoints[bi]})
        return out

