#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# -----------------------------
# Segmentation head (must match surgery architecture)
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 6, 12)):
        super().__init__()
        dils = list(dilations)
        bch = out_channels // len(dils)
        self.convs = nn.ModuleList()
        for d in dils:
            if int(d) == 1:
                conv = nn.Conv2d(in_channels, bch, 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, bch, 3, padding=int(d), dilation=int(d), bias=False)
            self.convs.append(nn.Sequential(conv, nn.BatchNorm2d(bch), nn.ReLU(inplace=True)))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, bch, 1, bias=False),
            nn.BatchNorm2d(bch),
            nn.ReLU(inplace=True),
        )
        total = bch * (len(dils) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [m(x) for m in self.convs]
        g = self.global_pool(x)
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=384):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(int(c), int(out_channels), 1, bias=False) for c in in_channels_list])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(out_channels), int(out_channels), 3, padding=1, bias=False),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        target = outs[0].shape[-2:]
        fused = []
        for i, f in enumerate(outs):
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            fused.append(f * self.fusion_weights[i])
        return sum(fused)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, num_classes=7, feature_dim=384, dropout_rate=0.1):
        super().__init__()
        self.fpn = FPN(in_channels_list, feature_dim)
        self.aspp = ASPP(feature_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(feature_dim // 2, num_classes, 1),
        )

    def forward(self, features):
        return self.decoder(self.aspp(self.fpn(features)))


class SharedBackboneDualDecoder(nn.Module):
    def __init__(self, backbone, encoder, det_decoder, pose_decoder, seg_head):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.det_decoder = det_decoder
        self.pose_decoder = pose_decoder
        self.seg_head = seg_head

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        enc = self.encoder(feats)

        det_out = self.det_decoder(enc, targets)
        pose_out = self.pose_decoder(enc, targets)  # not used for final drawing here

        seg_logits = self.seg_head(feats)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = {}
        if isinstance(det_out, dict):
            out.update({f"det_{k}": v for k, v in det_out.items()})
            if "pred_boxes" in det_out:
                out["pred_boxes"] = det_out["pred_boxes"]
            if "pred_logits" in det_out:
                out["det_pred_logits"] = det_out["pred_logits"]

        if isinstance(pose_out, dict):
            out.update({f"pose_{k}": v for k, v in pose_out.items()})

        out["segmentation"] = seg_logits
        return out


COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2),
]


def overlay_seg(frame_bgr: np.ndarray, seg_map: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    palette = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
        dtype=np.uint8,
    )
    seg_rgb = palette[np.clip(seg_map, 0, len(palette) - 1)]
    seg_bgr = seg_rgb[..., ::-1]
    fh, fw = frame_bgr.shape[:2]
    if seg_bgr.shape[:2] != (fh, fw):
        seg_bgr = cv2.resize(seg_bgr, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame_bgr, 1.0 - float(alpha), seg_bgr.astype(np.uint8), float(alpha), 0.0)


def draw_pose(img_bgr: np.ndarray, keypoints: np.ndarray, kpt_thr: float = 0.35):
    h, w = img_bgr.shape[:2]
    for i in range(keypoints.shape[0]):
        x, y, s = keypoints[i]
        if s < kpt_thr:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)

    for a, b in COCO_SKELETON:
        xa, ya, sa = keypoints[a]
        xb, yb, sb = keypoints[b]
        if sa < kpt_thr or sb < kpt_thr:
            continue
        if xa < 0 or ya < 0 or xa >= w or ya >= h:
            continue
        if xb < 0 or yb < 0 or xb >= w or yb >= h:
            continue
        cv2.line(img_bgr, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", required=True)
    ap.add_argument("--pose-ckpt", required=True, help="Standalone pose checkpoint (the good one)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--score-thr-det", type=float, default=0.35)
    ap.add_argument("--score-thr-pose", type=float, default=0.25)
    ap.add_argument("--kpt-thr", type=float, default=0.35)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-alpha", type=float, default=0.30)
    ap.add_argument("--crop-pad", type=float, default=0.12, help="extra padding ratio around det box for pose crop")
    ap.add_argument("--max-persons", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    for p in [repo, repo / "src", repo / "pose_estimation_berna", repo / "segmentation_sivert"]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from src.core import YAMLConfig
    from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor

    # merged model for det + seg
    det_cfg = YAMLConfig(str(args.det_config))
    pose_cfg_for_wrapper = YAMLConfig(str(args.pose_config))
    det_model = det_cfg.model
    pose_model_for_wrapper = pose_cfg_for_wrapper.model

    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640)
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    seg_head = SegmentationHead(
        in_channels_list=in_channels,
        num_classes=int(args.seg_num_classes),
        feature_dim=int(args.seg_feature_dim),
        dropout_rate=0.1,
    )
    merged_model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model_for_wrapper.decoder,
        seg_head=seg_head,
    )

    merged_ckpt = torch.load(args.merged_ckpt, map_location="cpu")
    merged_state = merged_ckpt["model"] if isinstance(merged_ckpt, dict) and "model" in merged_ckpt else merged_ckpt
    m_miss, m_unexp = merged_model.load_state_dict(merged_state, strict=False)
    print(f"[merged-load] missing={len(m_miss)} unexpected={len(m_unexp)}")

    # standalone pose model (good)
    pose_cfg = YAMLConfig(str(args.pose_config))
    pose_model = pose_cfg.model
    pose_ckpt = torch.load(args.pose_ckpt, map_location="cpu")
    if isinstance(pose_ckpt, dict):
        if "ema" in pose_ckpt and isinstance(pose_ckpt["ema"], dict) and "module" in pose_ckpt["ema"]:
            pose_state = pose_ckpt["ema"]["module"]
        elif "model" in pose_ckpt and isinstance(pose_ckpt["model"], dict):
            pose_state = pose_ckpt["model"]
        else:
            pose_state = pose_ckpt
    else:
        pose_state = pose_ckpt
    p_miss, p_unexp = pose_model.load_state_dict(pose_state, strict=False)
    print(f"[pose-load] missing={len(p_miss)} unexpected={len(p_unexp)}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    merged_model = merged_model.to(device).eval()
    pose_model = pose_model.deploy().to(device).eval()

    det_post = det_cfg.postprocessor.to(device).eval()
    if hasattr(det_post, "remap_mscoco_category"):
        det_post.remap_mscoco_category = True

    # pose post for crop-level predictions
    pose_post = DETRPosePostProcessor(
        num_classes=2,
        num_keypoints=17,
        num_top_queries=300,
        remap_mscoco_category=True,
    ).to(device).eval()

    tfm_full = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    tfm_crop = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame")
    h0, w0 = frame.shape[:2]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0, h0))

    frame_idx = 0
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        if frame.shape[:2] != (h0, w0):
            frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_full = tfm_full(Image.fromarray(rgb)).unsqueeze(0).to(device)
        orig_size_full = torch.tensor([[w0, h0]], device=device)

        with torch.no_grad():
            out = merged_model(x_full)
            det_in = {"pred_logits": out["det_pred_logits"], "pred_boxes": out["pred_boxes"]}
            det_res = det_post(det_in, orig_size_full)[0]
            seg_logits = out["segmentation"]
            seg_map = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        vis = overlay_seg(frame.copy(), seg_map, alpha=float(args.seg_alpha))

        det_labels = det_res["labels"].detach().cpu().numpy()
        det_scores = det_res["scores"].detach().cpu().numpy()
        det_boxes = det_res["boxes"].detach().cpu().numpy().astype(np.float32)

        keep = np.where((det_scores >= float(args.score_thr_det)) & np.isin(det_labels, [0, 1]))[0]
        if keep.size > 0:
            keep = keep[np.argsort(det_scores[keep])[::-1]][: int(args.max_persons)]

        for idx in keep:
            bx = det_boxes[idx]
            x1, y1, x2, y2 = bx.tolist()
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            px = float(args.crop_pad) * bw
            py = float(args.crop_pad) * bh
            cx1, cy1, cx2, cy2 = clamp_xyxy(x1 - px, y1 - py, x2 + px, y2 + py, w0, h0)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x_crop = tfm_crop(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
            crop_h, crop_w = crop.shape[:2]
            orig_size_crop = torch.tensor([[crop_w, crop_h]], device=device)

            with torch.no_grad():
                pose_out = pose_model(x_crop)
                pose_res = pose_post(pose_out, orig_size_crop)[0]

            p_scores = pose_res["scores"].detach().cpu().numpy()
            p_labels = pose_res["labels"].detach().cpu().numpy()
            p_kpts = pose_res["keypoints"].detach().cpu().numpy()  # [N,17,3] in crop px

            p_keep = np.where((p_scores >= float(args.score_thr_pose)) & np.isin(p_labels, [0, 1]))[0]
            if p_keep.size == 0:
                # draw det box anyway
                cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                cv2.putText(vis, f"person {det_scores[idx]:.2f}", (cx1, max(0, cy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue

            # choose best pose query in crop
            best = p_keep[np.argmax(p_scores[p_keep])]
            kpt = p_kpts[best].copy()
            # map crop -> full frame
            kpt[:, 0] += float(cx1)
            kpt[:, 1] += float(cy1)

            # draw
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(vis, f"person {det_scores[idx]:.2f}", (cx1, max(0, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            draw_pose(vis, kpt, kpt_thr=float(args.kpt_thr))

        writer.write(vis)
        frame_idx += 1
        if args.max_frames is not None and frame_idx >= int(args.max_frames):
            break

    cap.release()
    writer.release()
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
