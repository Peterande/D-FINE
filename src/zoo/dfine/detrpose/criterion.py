"""
Criterion for DETRPose-style outputs (pose-only).

Adapted from:
  DETRPose/src/models/detrpose/criterion.py

This criterion is designed to work with our pose_estimation_berna dataset format:
- targets[i]["keypoints"]: [N,K,3] (x_px, y_px, v)
- targets[i]["boxes"]: [N,4] cxcywh normalized (for area proxy)
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcherDETRPose, _area_from_targets_px


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class DETRPoseCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcherDETRPose,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        gamma: float = 2.0,
        num_keypoints: int = 17,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.focal_alpha = float(focal_alpha)
        self.gamma = float(gamma)
        self.num_keypoints = int(num_keypoints)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _num_boxes(self, targets: List[dict], device):
        n = sum(len(t["labels"]) for t in targets)
        return torch.as_tensor([max(1, n)], dtype=torch.float32, device=device)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_onehot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        loss_ce = sigmoid_focal_loss(
            src_logits, target_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma
        ) * src_logits.shape[1]
        return {"loss_ce": loss_ce}

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_kpt = outputs["pred_keypoints"][idx]  # [M,2K] in [0,1]
        if src_kpt.numel() == 0:
            z = outputs["pred_logits"].sum() * 0.0
            return {"loss_keypoints": z, "loss_oks": z}

        tgt_kpt = torch.cat([t["keypoints"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(
            src_kpt.device
        )  # [M,K,3] px

        # normalize gt to [0,1]
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=src_kpt.device, dtype=src_kpt.dtype)
        batch_idx = idx[0]
        wh = sizes[batch_idx].clamp(min=1.0)
        Z_gt = (tgt_kpt[..., :2] / wh[:, None, :]).reshape(tgt_kpt.shape[0], -1)
        V_gt = (tgt_kpt[..., 2] > 0).to(src_kpt.dtype)  # [M,K]

        # area proxy in pixel^2 for OKS denom
        # (OKS is computed in matcher; here we keep a simple 1-oks-like term via weighted l1)
        pose_loss = F.l1_loss(src_kpt, Z_gt, reduction="none")
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        loss_keypoints = pose_loss.sum() / num_boxes

        # Optional extra term: encourage high OKS by penalizing large errors on visible joints (already captured in L1).
        loss_oks = (pose_loss.sum(dim=1) / (V_gt.repeat_interleave(2, dim=1).sum(dim=1).clamp(min=1.0))).mean()
        return {"loss_keypoints": loss_keypoints, "loss_oks": loss_oks}

    def loss_vfl(self, outputs, targets, indices, num_boxes):
        """
        VariFocal-like: target quality is OKS for matched pairs.
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        # compute matched OKS as quality target
        src_kpt = outputs["pred_keypoints"][idx]  # [M,2K]
        tgt_kpt = torch.cat([t["keypoints"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(
            src_kpt.device
        )
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=src_kpt.device, dtype=src_kpt.dtype)
        wh = sizes[idx[0]].clamp(min=1.0)
        Z_gt = (tgt_kpt[..., :2] / wh[:, None, :]).reshape(tgt_kpt.shape[0], -1)
        V_gt = (tgt_kpt[..., 2] > 0).to(src_kpt.dtype)  # [M,K]

        # area in px^2 (use all targets, then pick matched)
        areas_all = _area_from_targets_px(targets, device=src_kpt.device, dtype=src_kpt.dtype)
        # build mapping from concatenated targets -> matched rows
        # indices pairs already align per batch; simplest: recompute per matched by gathering target boxes
        tgt_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(src_kpt.device)
        # compute area per matched in px^2
        areas_m = []
        for b, (_, J) in enumerate(indices):
            if J.numel() == 0:
                continue
            t = targets[b]
            boxes = t["boxes"][J].to(device=src_kpt.device, dtype=src_kpt.dtype)
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            bw = (boxes[:, 2] * float(w)).clamp(min=1.0)
            bh = (boxes[:, 3] * float(h)).clamp(min=1.0)
            areas_m.append(bw * bh)
        area = torch.cat(areas_m, dim=0) if areas_m else src_kpt.new_ones((0,))

        # OKS quality target (in pixel space)
        K = self.num_keypoints
        sigmas = src_kpt.new_tensor(
            [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
        )[:K]
        vars_ = (sigmas * 2) ** 2
        pred_xy = src_kpt.reshape(-1, K, 2) * wh[:, None, :]  # px
        gt_xy = Z_gt.reshape(-1, K, 2) * wh[:, None, :]
        d2 = ((pred_xy - gt_xy) ** 2).sum(-1)
        oks = torch.exp(-d2 / (area[:, None] * vars_[None, :] * 2 + 1e-6)) * V_gt
        oks = oks.sum(-1) / V_gt.sum(-1).clamp(min=1.0)
        oks = oks.detach().clamp(0.0, 1.0)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def forward(self, outputs: dict, targets: List[dict]):
        # match
        indices = self.matcher({"pred_logits": outputs["pred_logits"], "pred_keypoints": outputs["pred_keypoints"]}, targets)[
            "indices"
        ]
        num_boxes = float(self._num_boxes(targets, device=outputs["pred_logits"].device).item())

        # base losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_vfl(outputs, targets, indices, num_boxes))
        losses.update(self.loss_keypoints(outputs, targets, indices, num_boxes))

        # aux losses (decoder layers)
        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], list):
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_idx = self.matcher(
                    {"pred_logits": aux["pred_logits"], "pred_keypoints": aux["pred_keypoints"]}, targets
                )["indices"]
                l = {}
                l.update(self.loss_labels(aux, targets, aux_idx, num_boxes))
                l.update(self.loss_vfl(aux, targets, aux_idx, num_boxes))
                l.update(self.loss_keypoints(aux, targets, aux_idx, num_boxes))
                for k, v in l.items():
                    losses[f"{k}_{i}"] = v

        # apply weights (return weighted values only; train.py sums dict values)
        weighted = {}
        for k, v in losses.items():
            # Aux losses are named like: loss_ce_0, loss_keypoints_3, ...
            # Map them back to their base key in weight_dict (loss_ce, loss_keypoints, ...).
            base_k = k
            if k.startswith("loss_"):
                parts = k.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_k = parts[0]
            # Only include losses that are explicitly weighted (matches DETRPose config style).
            if base_k not in self.weight_dict and k not in self.weight_dict:
                continue
            w = self.weight_dict.get(base_k, self.weight_dict.get(k, 0.0))
            if float(w) == 0.0:
                continue
            weighted[k] = v * float(w)
        return weighted

