"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


_COCO17_SIGMAS = torch.tensor(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=torch.float32,
)


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    num_keypoints: int = 0,
    keypoints_noise_scale: float = 0.15,
):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return None, None, None, dn_meta, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    input_query_kpt = None
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
            if num_keypoints and "keypoints" in targets[i]:
                if input_query_kpt is None:
                    input_query_kpt = torch.zeros([bs, max_gt_num, int(num_keypoints), 2], device=device)
                # targets keypoints are (x_px, y_px, v) in resized image pixels.
                kpts = targets[i]["keypoints"][:num_gt, :, :2].to(device)
                vis = (targets[i]["keypoints"][:num_gt, :, 2] > 0).to(kpts.dtype).to(device)
                # get (w,h) for image coords
                if "size" in targets[i]:
                    h, w = targets[i]["size"].tolist()
                else:
                    w, h = targets[i]["orig_size"].tolist()
                wh = torch.tensor([w, h], device=device, dtype=kpts.dtype).view(1, 1, 2).clamp(min=1.0)
                kpts_xy = (kpts / wh) * vis.unsqueeze(-1)
                # convert image-normalized -> bbox-relative using GT boxes (cxcywh normalized)
                boxes = targets[i]["boxes"][:num_gt].to(device)  # [N,4]
                cx, cy, bw, bh = boxes.unbind(-1)
                x1 = (cx - 0.5 * bw).view(-1, 1, 1)
                y1 = (cy - 0.5 * bh).view(-1, 1, 1)
                bw = bw.view(-1, 1, 1).clamp(min=1e-6)
                bh = bh.view(-1, 1, 1).clamp(min=1e-6)
                rel_x = ((kpts_xy[..., 0:1] - x1) / bw).clamp(0.0, 1.0)
                rel_y = ((kpts_xy[..., 1:2] - y1) / bh).clamp(0.0, 1.0)
                input_query_kpt[i, :num_gt] = torch.cat([rel_x, rel_y], dim=-1)
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    if input_query_kpt is not None:
        input_query_kpt = input_query_kpt.tile([1, 2 * num_group, 1, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # negative indices (used for optional pose-quality negative supervision)
    negative_flat_mask = negative_gt_mask.squeeze(-1).bool() & pad_gt_mask.bool()
    dn_negative_idx = torch.nonzero(negative_flat_mask)[:, 1]
    dn_negative_idx = torch.split(dn_negative_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        # shrink_mask = torch.zeros_like(rand_sign)
        # shrink_mask[:, :, :2] = (rand_sign[:, :, :2] == 1)  # rand_sign == 1 → (x1, y1) ↘ →  smaller bbox
        # shrink_mask[:, :, 2:] = (rand_sign[:, :, 2:] == -1)  # rand_sign == -1 →  (x2, y2) ↖ →  smaller bbox
        # mask = rand_part > (upper_bound / (upper_bound+1))
        # # this is to make sure the dn bbox can be reversed to the original bbox by dfine head.
        # rand_sign = torch.where((shrink_mask * (1 - negative_gt_mask) * mask).bool(), \
        #                         rand_sign * upper_bound / (upper_bound+1) / rand_part, rand_sign)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    # Full OKS-style keypoint denoising (pose):
    # - build noisy keypoints in image space with per-joint sigmas + person scale (bbox area)
    # - convert back to bbox-relative coords used by the pose head
    # - negatives receive stronger noise to act as hard negatives (suppression)
    if input_query_kpt is not None and float(keypoints_noise_scale) > 0:
        # Recover original visibility mask from the GT keypoints we encoded (0 coords for invisible).
        # input_query_kpt is bbox-relative in [0,1]; invisible keypoints are exactly 0,0.
        vis = (input_query_kpt.abs().sum(-1) > 0).to(input_query_kpt.dtype)  # [B,L,K]

        # Compute image-normalized coords from bbox-relative.
        boxes = input_query_bbox  # [B,L,4] cxcywh in [0,1]
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - 0.5 * bw).unsqueeze(-1).unsqueeze(-1)  # [B,L,1,1]
        y1 = (cy - 0.5 * bh).unsqueeze(-1).unsqueeze(-1)
        bw_ = bw.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
        bh_ = bh.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)

        kpt_img_xy = torch.zeros_like(input_query_kpt)
        kpt_img_xy[..., 0:1] = x1 + input_query_kpt[..., 0:1] * bw_
        kpt_img_xy[..., 1:2] = y1 + input_query_kpt[..., 1:2] * bh_

        # Person scale in pixels (sqrt(area_px))
        # We can only approximate using original image size per sample.
        wh_pix = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            wh_pix.append([w, h])
        wh_pix = torch.tensor(wh_pix, device=device, dtype=input_query_kpt.dtype).view(bs, 1, 1, 2).clamp(min=1.0)
        bw_pix = (bw.unsqueeze(-1).unsqueeze(-1) * wh_pix[..., 0:1]).clamp(min=1.0)
        bh_pix = (bh.unsqueeze(-1).unsqueeze(-1) * wh_pix[..., 1:2]).clamp(min=1.0)
        area_px = (bw_pix * bh_pix).clamp(min=1.0)
        scale_px = torch.sqrt(area_px)  # [B,L,1,1]

        # Per-joint sigmas
        sigmas = _COCO17_SIGMAS.to(device=device, dtype=input_query_kpt.dtype)
        k = int(num_keypoints)
        if k > sigmas.numel():
            # fall back: repeat last sigma if custom K > 17
            sigmas = torch.cat([sigmas, sigmas.new_full((k - sigmas.numel(),), float(sigmas[-1]))], dim=0)
        sigmas = sigmas[:k].view(1, 1, k, 1)  # [1,1,K,1]

        # Noise strength: positives mild, negatives stronger (hard negatives).
        neg = negative_gt_mask.to(input_query_kpt.dtype)  # [B,L,1]
        strength = float(keypoints_noise_scale) * (1.0 + 2.0 * neg).unsqueeze(-1)  # pos=1x, neg=3x

        # Sample pixel-space noise ~ N(0, sigma * sqrt(area) * strength) and convert to image-normalized.
        noise_px = torch.randn_like(kpt_img_xy) * (sigmas * scale_px) * strength
        noise_norm = noise_px / wh_pix  # normalize per axis
        kpt_noisy_img = (kpt_img_xy + noise_norm).clamp(0.0, 1.0)

        # Convert back to bbox-relative and apply visibility mask.
        rel_x = ((kpt_noisy_img[..., 0:1] - x1) / bw_).clamp(0.0, 1.0)
        rel_y = ((kpt_noisy_img[..., 1:2] - y1) / bh_).clamp(0.0, 1.0)
        input_query_kpt = torch.cat([rel_x, rel_y], dim=-1) * vis.unsqueeze(-1)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_negative_idx": dn_negative_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta, input_query_kpt
