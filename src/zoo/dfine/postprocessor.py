"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        keypoints = outputs.get("pred_keypoints", None)
        pose_quality = outputs.get("pred_pose_quality", None)  # [B,Q,1] optional
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            if keypoints is not None:
                # gather query-aligned keypoints for selected boxes
                keypoints = keypoints.gather(
                    dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, keypoints.shape[-2], keypoints.shape[-1])
            )
            if pose_quality is not None:
                pose_quality = pose_quality.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, pose_quality.shape[-1]))
                # Couple person score with pose quality (Pose-LQE): higher quality => higher final score
                scores = scores * pose_quality.squeeze(-1).sigmoid()

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                if keypoints is not None:
                    keypoints = torch.gather(
                        keypoints,
                        dim=1,
                        index=index.unsqueeze(-1).unsqueeze(-1).repeat(
                            1, 1, keypoints.shape[-2], keypoints.shape[-1]
                        ),
                )

        # Clip boxes to image bounds to avoid extreme keypoint projection when boxes go out of frame.
        # orig_target_sizes convention in this repo: [B,2] = (w, h)
        if orig_target_sizes is not None:
            w_img = orig_target_sizes[:, 0].to(boxes.dtype).view(-1, 1)
            h_img = orig_target_sizes[:, 1].to(boxes.dtype).view(-1, 1)

            x1 = boxes[..., 0].clamp(min=0.0)
            y1 = boxes[..., 1].clamp(min=0.0)
            x2 = boxes[..., 2].clamp(min=0.0)
            y2 = boxes[..., 3].clamp(min=0.0)

            x1 = torch.minimum(x1, w_img)
            x2 = torch.minimum(x2, w_img)
            y1 = torch.minimum(y1, h_img)
            y2 = torch.minimum(y2, h_img)

            # ensure proper ordering after clipping
            x_min = torch.minimum(x1, x2)
            y_min = torch.minimum(y1, y2)
            x_max = torch.maximum(x1, x2)
            y_max = torch.maximum(y1, y2)
            boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for bi, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if pose_quality is not None:
                result["pose_quality"] = pose_quality[bi].squeeze(-1)
            results.append(result)

        # If pose keypoints are present, convert them to COCO format (x_px, y_px, score)
        if keypoints is not None:
            # keypoints are bbox-relative: (x_rel, y_rel) in [0,1] within each predicted bbox.
            # Convert to absolute pixels using the selected bbox predictions (already in pixels xyxy).
            # boxes is [B, Q, 4] in xyxy pixels after selection
            x1y1 = boxes[..., :2]  # [B,Q,2]
            wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1.0)  # [B,Q,2]
            kpt_xy = x1y1[:, :, None, :] + keypoints[..., :2] * wh[:, :, None, :]
            kpt_score = keypoints[..., 2].sigmoid()

            # Clip keypoints to image bounds for safer visualization/metric code downstream.
            if orig_target_sizes is not None:
                w_k = orig_target_sizes[:, 0].to(kpt_xy.dtype).view(-1, 1, 1)
                h_k = orig_target_sizes[:, 1].to(kpt_xy.dtype).view(-1, 1, 1)
                kx = kpt_xy[..., 0].clamp(min=0.0)
                ky = kpt_xy[..., 1].clamp(min=0.0)
                kx = torch.minimum(kx, w_k)
                ky = torch.minimum(ky, h_k)
                kpt_xy = torch.stack([kx, ky], dim=-1)

            kpt_out = torch.cat([kpt_xy, kpt_score[..., None]], dim=-1)  # [B, Q, K, 3]

            for i, res in enumerate(results):
                res["keypoints"] = kpt_out[i]

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
