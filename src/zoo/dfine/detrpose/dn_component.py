"""
Pose contrastive denoising component vendored from DETRPose.

Source: /DETRPose/src/models/detrpose/dn_component.py

NOTE: This file expects targets to provide:
- labels: [N]
- boxes: [N,4] (cxcywh normalized in our pipeline)
- keypoints: [N,K,3] (x_px, y_px, v) in resized image pixels
- size: [2] (H,W) resized
We adapt DETRPose logic to derive area if not provided.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import inverse_sigmoid


def get_sigmas(num_keypoints, device):
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
    elif num_keypoints == 14:
        sigmas = (
            np.array([0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.79, 0.79])
            / 10.0
        )
    elif num_keypoints == 3:
        sigmas = np.array([1.07, 1.07, 0.67], dtype=np.float32) / 10.0
    else:
        raise ValueError(f"Unsupported keypoints number {num_keypoints}")
    # prepend sigma for the center of the human
    sigmas = np.concatenate([[0.1], sigmas])
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    return sigmas[None, :, None]


def _area_from_boxes_cxcywh_norm(targets):
    # area in pixel^2, consistent with DETRPose's use in OKS denom
    areas = []
    for t in targets:
        boxes = t["boxes"]
        if "size" in t:
            h, w = t["size"].tolist()
        else:
            w, h = t["orig_size"].tolist()
        wh = boxes.new_tensor([w, h, w, h]).view(1, 4).clamp(min=1.0)
        bw = (boxes[:, 2] * wh[:, 0]).clamp(min=1.0)
        bh = (boxes[:, 3] * wh[:, 1]).clamp(min=1.0)
        areas.append((bw * bh).to(boxes.dtype))
    return torch.cat(areas, dim=0)


def prepare_for_cdn(
    dn_args,
    training,
    num_queries,
    num_classes,
    num_keypoints,
    hidden_dim,
    label_enc,
    pose_enc,
    device,
):
    """
    Return:
      input_query_label: [B, pad, (K+1), hidden_dim]
      input_query_pose:  [B, pad, (K+1), 2] (unsigmoid coords)
      attn_mask:         [pad+Q, pad+Q]
      dn_meta:           dict
    """
    if training:
        targets, dn_number, label_noise_ratio = dn_args
        dn_number = dn_number * 2  # pos+neg
        known = [torch.ones_like(t["labels"]) for t in targets]
        batch_size = len(known)
        known_num = [int(k.sum().item()) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))
        dn_number = 1 if dn_number == 0 else dn_number

        unmask_bbox = unmask_label = torch.cat(known)

        labels = torch.cat([t["labels"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_labels_expaned = known_labels.clone()

        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        # keypoint noise
        boxes = torch.cat([t["boxes"] for t in targets])  # cxcywh normalized
        xy = boxes[:, :2]  # center in [0,1]
        keypoints = torch.cat([t["keypoints"] for t in targets])  # [N,K,3] px

        # derive areas in pixel^2 (preferred for OKS denom)
        if "area" in targets[0]:
            areas = torch.cat([t["area"] for t in targets]).to(device=device)
        else:
            areas = _area_from_boxes_cxcywh_norm(targets).to(device=device)

        poses = keypoints[..., :2]  # px
        # normalize to [0,1] in resized image
        sizes = []
        for t in targets:
            if "size" in t:
                h, w = t["size"].tolist()
            else:
                w, h = t["orig_size"].tolist()
            sizes.append([w, h])
        sizes = torch.tensor(sizes, device=device, dtype=poses.dtype)
        # expand per-instance
        rep = torch.cat([torch.full((len(t["labels"]),), i, device=device, dtype=torch.long) for i, t in enumerate(targets)])
        wh = sizes[rep].clamp(min=1.0)  # [N,2]
        poses = (poses / wh[:, None, :]).clamp(0.0, 1.0)
        poses = poses.reshape(poses.shape[0], -1)  # [N, K*2]
        poses = torch.cat([xy, poses], dim=1)  # prepend center: [N, 2 + K*2]

        non_viz = (keypoints[..., 2] <= 0).reshape(keypoints.shape[0], -1)
        non_viz = torch.cat((torch.zeros_like(non_viz[:, 0:1]).bool(), non_viz), dim=1)

        vars_ = (2 * get_sigmas(num_keypoints, device)) ** 2

        known_poses = poses.repeat(2 * dn_number, 1).reshape(-1, num_keypoints + 1, 2)
        known_areas = areas.repeat(2 * dn_number)[..., None, None]
        known_non_viz = non_viz.repeat(2 * dn_number, 1)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.arange(len(poses), device=device).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.arange(dn_number, device=device).long() * len(poses) * 2).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(poses)

        eps = np.finfo("float32").eps
        rand_vector = torch.rand_like(known_poses)
        rand_vector = F.normalize(rand_vector, -1)
        rand_alpha = torch.zeros_like(known_poses[..., :1]).uniform_(-np.log(1), -np.log(0.5))
        rand_alpha[negative_idx] = rand_alpha[negative_idx].uniform_(-np.log(0.5), -np.log(0.1))
        rand_alpha *= 2 * (known_areas + eps) * vars_
        # normalize by max(H,W) in pixels (approx using resized size)
        img_dim = targets[0]["size"].tolist() if "size" in targets[0] else targets[0]["orig_size"].tolist()[::-1]
        rand_alpha = torch.sqrt(rand_alpha) / float(max(img_dim))
        rand_alpha[known_non_viz] = 0.0

        known_poses_expand = (known_poses + rand_alpha * rand_vector).clamp(0.0, 1.0)

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        input_label_pose_embed = pose_enc.weight[None].repeat(known_poses_expand.size(0), 1, 1)
        input_label_embed = torch.cat([input_label_embed.unsqueeze(1), input_label_pose_embed], dim=1).flatten(1)

        input_pose_embed = inverse_sigmoid(known_poses_expand)

        padding_label = torch.zeros(pad_size, hidden_dim * (num_keypoints + 1), device=device)
        padding_pose = torch.zeros(pad_size, num_keypoints + 1, device=device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_pose = padding_pose[..., None].repeat(batch_size, 1, 1, 2)

        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.arange(num, device=device) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_pose[(known_bid.long(), map_known_indice)] = input_pose_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

        dn_meta = {"pad_size": pad_size, "num_dn_group": dn_number}
    else:
        input_query_label = None
        input_query_pose = None
        attn_mask = None
        dn_meta = None

    return input_query_label.unflatten(-1, (-1, hidden_dim)), input_query_pose, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_keypoints, dn_meta, aux_loss, _set_aux_loss):
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_keypoints = outputs_keypoints[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_keypoints = outputs_keypoints[:, :, dn_meta["pad_size"] :, :]
        out = {"pred_logits": output_known_class[-1], "pred_keypoints": output_known_keypoints[-1]}
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_keypoints)
        dn_meta["output_known_lbs_keypoints"] = out
    return outputs_class, outputs_keypoints

