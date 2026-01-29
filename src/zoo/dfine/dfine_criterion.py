"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        keypoints_box_mode: str = "target",
        vfl_target: str = "iou",
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        # Keypoint coord frame:
        # - "target": normalize GT keypoints within GT boxes (classic)
        # - "pred_detached": normalize GT keypoints within predicted boxes (no grad into boxes)
        #   This matches inference where we decode keypoints using predicted boxes.
        assert keypoints_box_mode in ("target", "pred_detached")
        self.keypoints_box_mode = keypoints_box_mode
        assert vfl_target in ("iou", "oks")
        self.vfl_target = vfl_target
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None

        # COCO keypoint sigmas (17) for OKS computation
        self.register_buffer(
            "_coco_sigmas",
            torch.tensor(
                [
                    0.26, 0.25, 0.25, 0.35, 0.35,
                    0.79, 0.79, 0.72, 0.72, 0.62,
                    0.62, 1.07, 1.07, 0.87, 0.87,
                    0.89, 0.89,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def _matched_oks(self, outputs, targets, indices):
        """
        Compute OKS for matched pairs only.
        Returns tensor [M] aligned with matched src indices.
        """
        if "pred_keypoints" not in outputs:
            return None
        # concat matched targets
        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs["pred_keypoints"][idx]  # [M,K,3]
        if src_kpts.numel() == 0:
            return src_kpts.new_zeros((0,))
        tgt_kpts = torch.cat([t["keypoints"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(src_kpts.device)
        if tgt_kpts.numel() == 0:
            return src_kpts.new_zeros((0,))

        # per-image (w,h) for pixel conversion
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=src_kpts.device, dtype=src_kpts.dtype)  # [B,2] (w,h)
        batch_idx = idx[0]  # [M]
        wh = sizes[batch_idx].clamp(min=1.0)  # [M,2]

        # predicted boxes for matched queries -> pixel xyxy
        src_boxes = outputs["pred_boxes"][idx]  # [M,4] cxcywh in [0,1]
        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.device)

        def cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        # decode predicted keypoints to pixels using predicted boxes (detached)
        box_xyxy = cxcywh_to_xyxy(src_boxes.detach())  # [M,4] in image-normalized
        x1y1 = box_xyxy[:, :2] * wh
        bwh = (box_xyxy[:, 2:] - box_xyxy[:, :2]).clamp(min=1e-6) * wh
        pred_xy_px = x1y1[:, None, :] + src_kpts[..., :2] * bwh[:, None, :]  # [M,K,2]

        gt_xy_px = tgt_kpts[..., :2]
        gt_vis = (tgt_kpts[..., 2] > 0).to(pred_xy_px.dtype)  # [M,K]

        # area from GT boxes (pixels^2)
        gt_xyxy = cxcywh_to_xyxy(tgt_boxes)  # [M,4] normalized
        gt_wh = (gt_xyxy[:, 2:] - gt_xyxy[:, :2]).clamp(min=1e-6) * wh
        area = (gt_wh[:, 0] * gt_wh[:, 1]).clamp(min=1.0)  # [M]

        d2 = ((pred_xy_px - gt_xy_px) ** 2).sum(-1)  # [M,K]
        sigmas = self._coco_sigmas.to(device=pred_xy_px.device, dtype=pred_xy_px.dtype)  # [K]
        vars_ = (sigmas * 2.0) ** 2  # [K]
        denom = (2.0 * vars_[None, :] * area[:, None]).clamp(min=1e-6)  # [M,K]
        oks = torch.exp(-d2 / denom) * gt_vis
        vis_cnt = gt_vis.sum(-1).clamp(min=1.0)
        oks = oks.sum(-1) / vis_cnt
        return oks.clamp(0.0, 1.0).detach()

    def loss_pose_lqe(self, outputs, targets, indices, num_boxes, values=None):
        """
        Pose-LQE: predict per-query pose quality (OKS) as a logit.
        """
        if "pred_pose_quality" not in outputs:
            z = outputs["pred_boxes"].sum() * 0.0
            return {"loss_pose_lqe": z}
        idx = self._get_src_permutation_idx(indices)
        pred_q = outputs["pred_pose_quality"][idx].squeeze(-1)  # [M]
        if values is None:
            oks = self._matched_oks(outputs, targets, indices)
            if oks is None:
                oks = pred_q.detach() * 0.0
        else:
            oks = values
        loss_pos = F.binary_cross_entropy_with_logits(pred_q, oks.to(pred_q.dtype), reduction="mean")

        # Optional: for DN negatives, explicitly push pose_quality -> 0 (suppresses "ghost humans" under occlusion).
        loss_neg = None
        if outputs.get("is_dn", False) and isinstance(outputs.get("dn_meta", None), dict):
            dn_meta = outputs["dn_meta"]
            if "dn_negative_idx" in dn_meta and dn_meta["dn_negative_idx"] is not None:
                neg_list = dn_meta["dn_negative_idx"]
                if isinstance(neg_list, (list, tuple)) and len(neg_list) > 0:
                    batch_idx = []
                    src_idx = []
                    for bi, nidx in enumerate(neg_list):
                        if nidx is None or int(nidx.numel()) == 0:
                            continue
                        batch_idx.append(torch.full_like(nidx, bi))
                        src_idx.append(nidx)
                    if batch_idx and src_idx:
                        b = torch.cat(batch_idx, dim=0)
                        s = torch.cat(src_idx, dim=0)
                        pred_qn = outputs["pred_pose_quality"][b, s].squeeze(-1)
                        tgt0 = torch.zeros_like(pred_qn)
                        loss_neg = F.binary_cross_entropy_with_logits(pred_qn, tgt0, reduction="mean")

        if loss_neg is None:
            return {"loss_pose_lqe": loss_pos}
        # balance pos/neg so negatives don't dominate
        return {"loss_pose_lqe": 0.5 * loss_pos + 0.5 * loss_neg}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """
        Keypoint loss for COCO-style keypoints.

        Expects:
          - outputs["pred_keypoints"]: [B, Q, K, 3] where 3 = (x_rel, y_rel, vis_logit)
              where (x_rel, y_rel) are normalized to the matched bbox (0..1 within bbox)
          - targets[i]["keypoints"]: [N_i, K, 3] where 3 = (x_px, y_px, v) and v in {0,1,2}
        Notes:
          - Target x/y are normalized using target["size"] if present, else target["orig_size"].
          - Visibility is supervised as (v > 0).
        """
        # Some output dicts (notably `pre_outputs`) don't include keypoints.
        # Skip keypoint supervision for those dicts rather than crashing.
        if "pred_keypoints" not in outputs:
            z = outputs["pred_boxes"].sum() * 0.0
            return {"loss_keypoints": z, "loss_keypoints_vis": z}
        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs["pred_keypoints"][idx]  # [M, K, 3]

        # concat targets in matched order
        tgt_kpts = torch.cat([t["keypoints"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(
            src_kpts.device
        )  # [M, K, 3]

        # normalize target coords to [0, 1] (image-normalized)
        # - target["size"] is [h, w] (see src/data/transforms/functional.py)
        # - target["orig_size"] is [w, h] (see src/data/dataset/coco_dataset.py)
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=src_kpts.device, dtype=src_kpts.dtype)  # [B,2] (w,h)

        # map each matched element back to its image to pick correct normalization
        batch_idx = idx[0]  # [M]
        wh = sizes[batch_idx]  # [M,2]
        tgt_xy = tgt_kpts[..., :2] / wh[:, None, :]  # [M,K,2]
        tgt_vis = (tgt_kpts[..., 2] > 0).to(src_kpts.dtype)  # [M,K]

        # convert image-normalized target xy -> bbox-relative xy
        # boxes are expected to be normalized cxcywh in targets/outputs
        assert "pred_boxes" in outputs, "pred_boxes required for bbox-relative keypoints"
        src_boxes = outputs["pred_boxes"][idx]  # [M,4] cxcywh in [0,1]
        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(
            src_boxes.device
        )  # [M,4] cxcywh in [0,1]

        def cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        if self.keypoints_box_mode == "pred_detached":
            # Use predicted boxes as the coordinate frame (but detach to avoid destabilizing box learning)
            box_ref = src_boxes.detach()
        else:
            box_ref = tgt_boxes

        box_xyxy = cxcywh_to_xyxy(box_ref)
        box_wh = (box_xyxy[:, 2:] - box_xyxy[:, :2]).clamp(min=1e-6)  # [M,2]
        tgt_rel_xy = (tgt_xy - box_xyxy[:, None, :2]) / box_wh[:, None, :]  # [M,K,2]
        tgt_rel_xy = tgt_rel_xy.clamp(0.0, 1.0)

        pred_xy = src_kpts[..., :2]
        pred_vis_logit = src_kpts[..., 2]

        # coordinate loss only on visible keypoints
        vis_mask = tgt_vis > 0
        if vis_mask.any():
            coord_loss = F.smooth_l1_loss(pred_xy, tgt_rel_xy, reduction="none")  # [M,K,2]
            coord_loss = coord_loss.sum(-1)  # [M,K]
            coord_loss = (coord_loss * vis_mask.to(coord_loss.dtype)).sum() / (
                vis_mask.to(coord_loss.dtype).sum() + 1e-6
            )
        else:
            coord_loss = pred_xy.sum() * 0.0

        # visibility loss (BCE with logits)
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis_logit, tgt_vis, reduction="mean")

        return {"loss_keypoints": coord_loss, "loss_keypoints_vis": vis_loss}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
            "keypoints": self.loss_keypoints,
            "pose_lqe": self.loss_pose_lqe,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            # Eval/inference outputs typically don't include aux outputs.
            # Fall back to standard last-layer matching only.
            indices_go = indices
            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["dn_meta"] = outputs["dn_meta"]
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                aux_outputs["is_dn"] = True
                aux_outputs["dn_meta"] = outputs["dn_meta"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        # KSVF-style VFL: use pose OKS as q instead of bbox IoU when enabled and available.
        if loss == "vfl" and self.vfl_target == "oks":
            oks = self._matched_oks(outputs, targets, indices)
            if oks is not None:
                return {"values": oks}

        # Pose-LQE supervised by OKS
        if loss == "pose_lqe":
            oks = self._matched_oks(outputs, targets, indices)
            if oks is not None:
                return {"values": oks}

        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl",):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
