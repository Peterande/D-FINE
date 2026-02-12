#!/usr/bin/env python3
"""Single-pass inference for Option 1 merged checkpoint (shared trunk)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# ---- bootstrap path before local imports ----
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from tools.model_surgery.shared_arch import SegmentationHead, SharedBackboneDualDecoder

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2),
]


def overlay_seg(frame_bgr: np.ndarray, seg_map: np.ndarray, alpha: float) -> np.ndarray:
    palette = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
        dtype=np.uint8,
    )
    seg_rgb = palette[np.clip(seg_map, 0, len(palette) - 1)]
    seg_bgr = seg_rgb[..., ::-1]
    h, w = frame_bgr.shape[:2]
    if seg_bgr.shape[:2] != (h, w):
        seg_bgr = cv2.resize(seg_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame_bgr, 1.0 - float(alpha), seg_bgr, float(alpha), 0.0)


def draw_pose(img: np.ndarray, box: np.ndarray, kpts: np.ndarray, kpt_thr: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cv2.rectangle(img, (int(max(0, x1)), int(max(0, y1))), (int(min(w - 1, x2)), int(min(h - 1, y2))), (0, 255, 0), 2)

    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    diag = float(np.hypot(bw, bh))
    vx1, vy1 = x1 - 0.15 * bw, y1 - 0.20 * bh
    vx2, vy2 = x2 + 0.15 * bw, y2 + 0.20 * bh

    valid = np.zeros((kpts.shape[0],), dtype=bool)
    for i in range(kpts.shape[0]):
        x, y, s = kpts[i]
        if s < kpt_thr:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        if not (vx1 <= x <= vx2 and vy1 <= y <= vy2):
            continue
        valid[i] = True
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    max_limb = 0.65 * diag
    for a, b in COCO_SKELETON:
        if not (valid[a] and valid[b]):
            continue
        xa, ya, _ = kpts[a]
        xb, yb, _ = kpts[b]
        if float(np.hypot(xa - xb, ya - yb)) > max_limb:
            continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1[None, :])
    iy1 = np.maximum(ay1, by1[None, :])
    ix2 = np.minimum(ax2, bx2[None, :])
    iy2 = np.minimum(ay2, by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / union, 0.0).astype(np.float32)


def pose_boxes_from_keypoints(kpts: np.ndarray) -> np.ndarray:
    if kpts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x = kpts[..., 0]
    y = kpts[..., 1]
    return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=-1).astype(np.float32)


def greedy_match(det_boxes: np.ndarray, pose_boxes: np.ndarray, pose_scores: np.ndarray, min_iou: float) -> List[Tuple[int, int]]:
    if det_boxes.size == 0 or pose_boxes.size == 0:
        return []
    iou = iou_matrix(det_boxes, pose_boxes)
    score = iou + 0.15 * pose_scores[None, :]
    pairs = []
    used_d, used_p = set(), set()
    for di, pi in np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1], score.shape))[0]:
        di = int(di)
        pi = int(pi)
        if di in used_d or pi in used_p:
            continue
        if float(iou[di, pi]) < float(min_iou):
            continue
        pairs.append((di, pi))
        used_d.add(di)
        used_p.add(pi)
    return pairs


def build_model_from_merged(
    det_config: str,
    pose_config: str,
    merged_ckpt: str,
    seg_num_classes: int,
    seg_feature_dim: int,
    seg_dropout: float,
    image_size: int,
):
    from src.core import YAMLConfig

    det_cfg = YAMLConfig(str(det_config))
    pose_cfg = YAMLConfig(str(pose_config))

    det_model = det_cfg.model
    pose_model = pose_cfg.model

    det_model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, int(image_size), int(image_size))
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    seg_head = SegmentationHead(in_channels, int(seg_num_classes), int(seg_feature_dim), float(seg_dropout))
    model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model.decoder,
        seg_head=seg_head,
    )

    ckpt = torch.load(merged_ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[merged-load] missing={len(missing)} unexpected={len(unexpected)}")
    return model, det_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--image-size", type=int, default=640)

    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--pose-score-thr", type=float, default=0.35)
    ap.add_argument("--kpt-thr", type=float, default=0.35)
    ap.add_argument("--match-min-iou", type=float, default=0.10)
    ap.add_argument("--max-center-dist-ratio", type=float, default=0.45)

    ap.add_argument("--tracker-oks-weight", type=float, default=0.2)
    ap.add_argument("--tracker-oks-thr", type=float, default=0.6)
    ap.add_argument("--debug-no-tracker", action="store_true",
                    help="Draw matched det+pose directly (no tracker) to isolate offset source.")

    ap.add_argument("--seg-alpha", type=float, default=0.30)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--max-persons", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    for p in [repo, repo / "src", repo / "pose_estimation_berna", repo / "segmentation_sivert", repo / "tools"]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor
    from pose_estimation_berna.core.tracking import KalmanTracker

    model, det_cfg = build_model_from_merged(
        det_config=args.det_config,
        pose_config=args.pose_config,
        merged_ckpt=args.merged_ckpt,
        seg_num_classes=int(args.seg_num_classes),
        seg_feature_dim=int(args.seg_feature_dim),
        seg_dropout=float(args.seg_dropout),
        image_size=int(args.image_size),
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()

    det_post = det_cfg.postprocessor.to(device).eval()
    if hasattr(det_post, "remap_mscoco_category"):
        det_post.remap_mscoco_category = True

    pose_post = DETRPosePostProcessor(
        num_classes=2, num_keypoints=17, num_top_queries=300, remap_mscoco_category=True
    ).to(device).eval()

    tfm = T.Compose([T.Resize((int(args.image_size), int(args.image_size))), T.ToTensor()])

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame")

    h0, w0 = frame.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0, h0))

    tracker = None
    if not args.debug_no_tracker:
        tracker = KalmanTracker(
            iou_threshold=0.3,
            oks_threshold=float(args.tracker_oks_thr),
            max_age=30,
            smooth_alpha=0.8,
            smooth_boxes=True,
            smooth_keypoints=True,
            min_area_ratio=0.6,
            score_decay=0.98,
            score_decay_grace=0,
            score_ema=0.8,
            iou_weight=1.0,
            oks_weight=float(args.tracker_oks_weight),
            kpt_thr=float(args.kpt_thr),
            kpt_age_decay=0.97,
            fps=float(fps),
            occlusion_ttl_sec=1.0,
            q_inflate_alpha=0.02,
            p_inflate_per_frame=1.02,
            size_clamp_min_scale=0.7,
            size_clamp_max_scale=1.3,
            uncertainty_rel=0.35,
            uncertainty_abs_px=80.0,
            out_of_frame_max=5,
            new_track_score_thr=float(args.score_thr),
        )

    frame_idx = 0
    t_acc = 0.0
    n_acc = 0
    matched_acc = 0
    det_acc = 0

    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        if frame.shape[:2] != (h0, w0):
            frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)

        t0 = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        orig_size = torch.tensor([[w0, h0]], device=device)

        with torch.no_grad():
            outputs = model(x)

            det_dict = {"pred_logits": outputs["det.pred_logits"], "pred_boxes": outputs["det.pred_boxes"]}
            det_res = det_post(det_dict, orig_size)[0]

            pose_dict = {"pred_logits": outputs["pose.pred_logits"], "pred_keypoints": outputs["pose.pred_keypoints"]}
            pose_res = pose_post(pose_dict, orig_size)[0]

            seg_logits = outputs["seg.logits"]
            seg_map = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        det_labels = det_res["labels"].detach().cpu().numpy()
        det_scores = det_res["scores"].detach().cpu().numpy()
        det_boxes = det_res["boxes"].detach().cpu().numpy().astype(np.float32)

        dkeep = np.where((det_scores >= float(args.score_thr)) & np.isin(det_labels, [0, 1]))[0]
        if dkeep.size > 0:
            dkeep = dkeep[np.argsort(det_scores[dkeep])[::-1]][: int(args.max_persons)]

        pose_scores = pose_res["scores"].detach().cpu().numpy()
        pose_kpts = pose_res["keypoints"].detach().cpu().numpy().astype(np.float32)
        pkeep = np.where(pose_scores >= float(args.pose_score_thr))[0]

        det_boxes_f = det_boxes[dkeep] if dkeep.size > 0 else np.zeros((0, 4), dtype=np.float32)
        det_scores_f = det_scores[dkeep] if dkeep.size > 0 else np.zeros((0,), dtype=np.float32)
        pose_kpts_f = pose_kpts[pkeep] if pkeep.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
        pose_scores_f = pose_scores[pkeep] if pkeep.size > 0 else np.zeros((0,), dtype=np.float32)
        pose_boxes_f = pose_boxes_from_keypoints(pose_kpts_f)

        pairs = greedy_match(det_boxes_f, pose_boxes_f, pose_scores_f, min_iou=float(args.match_min_iou))

        # center-distance gating
        filtered_pairs = []
        for di, pi in pairs:
            db = det_boxes_f[di]
            pb = pose_boxes_f[pi]

            dcx, dcy = 0.5 * (db[0] + db[2]), 0.5 * (db[1] + db[3])
            pcx, pcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])

            dw, dh = max(1.0, db[2] - db[0]), max(1.0, db[3] - db[1])
            ddiag = float(np.hypot(dw, dh))
            cdist = float(np.hypot(dcx - pcx, dcy - pcy))

            if cdist <= float(args.max_center_dist_ratio) * ddiag:
                filtered_pairs.append((di, pi))
        pairs = filtered_pairs

        det_acc += int(det_boxes_f.shape[0])
        matched_acc += int(len(pairs))

        vis = overlay_seg(frame.copy(), seg_map, alpha=float(args.seg_alpha))

        if args.debug_no_tracker:
            # Debug mode: draw matched results directly (no smoothing / no tracker lag)
            for di, pi in pairs:
                box = det_boxes_f[di]
                kpt = pose_kpts_f[pi]
                cv2.putText(
                    vis,
                    f"debug person {float(det_scores_f[di]):.2f}",
                    (int(box[0]), max(0, int(box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                draw_pose(vis, box, kpt, kpt_thr=float(args.kpt_thr))
        else:
            # Production mode with fallback: keep bbox tracks even if no pose match this frame
            if len(pairs) == 0:
                if det_boxes_f.shape[0] > 0:
                    track_out = tracker.update(
                        det_boxes_f,
                        det_scores_f,
                        None,
                        None,
                        img_wh=(int(w0), int(h0)),
                    )
                else:
                    track_out = tracker.update(
                        np.zeros((0, 4), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        None,
                        None,
                        img_wh=(int(w0), int(h0)),
                    )
            else:
                m_boxes = np.asarray([det_boxes_f[di] for di, _ in pairs], dtype=np.float32)
                m_scores = np.asarray([det_scores_f[di] for di, _ in pairs], dtype=np.float32)
                m_kpts = np.asarray([pose_kpts_f[pi] for _, pi in pairs], dtype=np.float32)
                track_out = tracker.update(m_boxes, m_scores, m_kpts, None, img_wh=(int(w0), int(h0)))

            for t in track_out:
                box = t.box_smooth if t.box_smooth is not None else t.box_xyxy
                kpt = t.kpt_smooth if t.kpt_smooth is not None else t.keypoints
                cv2.putText(
                    vis,
                    f"id {int(t.track_id)} person {float(t.score):.2f}",
                    (int(box[0]), max(0, int(box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if kpt is not None:
                    draw_pose(vis, box, kpt, kpt_thr=float(args.kpt_thr))

        writer.write(vis)

        if args.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if frame_idx >= 5:
                t_acc += dt
                n_acc += 1
                if n_acc % 30 == 0:
                    mr = (matched_acc / max(1, det_acc))
                    print(f"[profile] avg={t_acc / max(1, n_acc):.2f} ms/frame | match_rate={mr:.2f}")

        frame_idx += 1
        if args.max_frames is not None and frame_idx >= int(args.max_frames):
            break

    cap.release()
    writer.release()

    if args.profile and n_acc > 0:
        mr = (matched_acc / max(1, det_acc))
        print(f"[final-profile] avg={t_acc / n_acc:.2f} ms/frame | match_rate={mr:.2f}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
