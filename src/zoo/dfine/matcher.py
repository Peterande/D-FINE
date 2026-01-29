"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_oks = float(weight_dict.get("cost_oks", 0.0))

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 or self.cost_oks != 0
        ), "all costs cant be 0"

    @staticmethod
    def _coco_sigmas(device, dtype):
        # COCO keypoint sigmas (17)
        sigmas = torch.tensor(
            [
                0.26, 0.25, 0.25, 0.35, 0.35,
                0.79, 0.79, 0.72, 0.72, 0.62,
                0.62, 1.07, 1.07, 0.87, 0.87,
                0.89, 0.89,
            ],
            device=device,
            dtype=dtype,
        )
        return sigmas

    @torch.no_grad()
    def _oks_cost(self, pred_boxes, pred_keypoints, tgt_boxes, tgt_keypoints, img_wh):
        """
        Compute OKS-based cost matrix for one image.
        pred_boxes: [Q,4] cxcywh normalized
        pred_keypoints: [Q,K,3] (x_rel,y_rel,vis_logit)
        tgt_boxes: [N,4] cxcywh normalized
        tgt_keypoints: [N,K,3] (x_px,y_px,v)
        img_wh: (w,h) floats
        Returns: cost_oks [Q,N] where lower is better (we use -OKS)
        """
        if tgt_boxes.numel() == 0:
            return pred_boxes.new_zeros((pred_boxes.shape[0], 0))

        w, h = img_wh
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        wh = pred_boxes.new_tensor([w, h, w, h])

        # boxes to pixel xyxy
        def cxcywh_to_xyxy(b):
            cx, cy, bw, bh = b.unbind(-1)
            x1 = cx - 0.5 * bw
            y1 = cy - 0.5 * bh
            x2 = cx + 0.5 * bw
            y2 = cy + 0.5 * bh
            return torch.stack([x1, y1, x2, y2], dim=-1)

        pb = cxcywh_to_xyxy(pred_boxes) * wh
        tb = cxcywh_to_xyxy(tgt_boxes) * wh

        # pred keypoints to pixels using predicted boxes
        x1y1 = pb[:, None, :2]  # [Q,1,2]
        pwh = (pb[:, None, 2:] - pb[:, None, :2]).clamp(min=1.0)  # [Q,1,2]
        pxy = x1y1 + pred_keypoints[..., :2] * pwh  # [Q,K,2]

        gxy = tgt_keypoints[..., :2]  # [N,K,2] in pixels (already resized)
        gvis = (tgt_keypoints[..., 2] > 0).to(pxy.dtype)  # [N,K]

        # broadcast distances: [Q,N,K]
        dxy = (pxy[:, None, :, :] - gxy[None, :, :, :])  # [Q,N,K,2]
        d2 = (dxy**2).sum(-1)  # [Q,N,K]

        # area from target boxes (pixels^2)
        twh = (tb[:, 2:] - tb[:, :2]).clamp(min=1.0)  # [N,2]
        area = (twh[:, 0] * twh[:, 1]).clamp(min=1.0)  # [N]

        sigmas = self._coco_sigmas(device=pxy.device, dtype=pxy.dtype)  # [K]
        vars_ = (sigmas * 2.0) ** 2  # [K]
        denom = (2.0 * vars_[None, None, :] * area[None, :, None]).clamp(min=1e-6)  # [1,N,K]

        oks = torch.exp(-d2 / denom) * gvis[None, :, :]  # [Q,N,K]
        vis_cnt = gvis.sum(-1).clamp(min=1.0)  # [N]
        oks = oks.sum(-1) / vis_cnt[None, :]  # [Q,N]
        return -oks

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        sizes = [len(v["boxes"]) for v in targets]

        cost_oks = None
        if self.cost_oks > 0 and ("pred_keypoints" in outputs):
            # Build a [B*Q, sumN] OKS cost with per-image blocks aligned to target concatenation order.
            pred_kpts = outputs["pred_keypoints"]  # [B,Q,K,3]
            sumN = int(tgt_bbox.shape[0])
            cost_oks = out_bbox.new_zeros((bs * num_queries, sumN))

            col_start = 0
            for bi, n_t in enumerate(sizes):
                col_end = col_start + int(n_t)
                if n_t == 0:
                    col_start = col_end
                    continue
                q_boxes = outputs["pred_boxes"][bi]  # [Q,4]
                q_kpts = pred_kpts[bi]  # [Q,K,3]
                t_boxes = targets[bi]["boxes"]  # [N,4]
                t_kpts = targets[bi].get("keypoints", None)
                if t_kpts is None:
                    col_start = col_end
                    continue
                if "size" in targets[bi]:
                    hh, ww = targets[bi]["size"].tolist()
                    img_wh = (ww, hh)
                else:
                    ww, hh = targets[bi]["orig_size"].tolist()
                    img_wh = (ww, hh)
                block = self._oks_cost(q_boxes, q_kpts, t_boxes, t_kpts, img_wh)  # [Q,N]
                row_start = bi * num_queries
                row_end = row_start + num_queries
                cost_oks[row_start:row_end, col_start:col_end] = block.to(cost_oks.dtype)
                col_start = col_end

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if cost_oks is not None:
            C = C + (self.cost_oks * cost_oks)
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
