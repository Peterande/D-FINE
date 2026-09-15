#!/usr/bin/env python3
"""ONNX inference for model-surgery singlepass exports (det+pose+seg raw heads)."""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.core import YAMLConfig
from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor
from tools.inference.trt_inf import (
    COCO_SKELETON,
    fallback_assign_by_center,
    greedy_match,
    iou_matrix,
    overlay_seg,
    pose_boxes_from_keypoints_conf,
)


def draw_pose(img: np.ndarray, box: np.ndarray, kpts: np.ndarray, kpt_thr: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cv2.rectangle(
        img,
        (int(max(0, x1)), int(max(0, y1))),
        (int(min(w - 1, x2)), int(min(h - 1, y2))),
        (0, 255, 0),
        2,
    )

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


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--pose-score-thr", type=float, default=0.35)
    ap.add_argument("--kpt-thr", type=float, default=0.35)
    ap.add_argument("--pose-box-kpt-thr", type=float, default=0.45)
    ap.add_argument("--pose-box-min-kpts", type=int, default=6)
    ap.add_argument("--match-min-iou", type=float, default=0.25)
    ap.add_argument("--max-center-dist-ratio", type=float, default=0.30)
    ap.add_argument("--fallback-min-match-rate", type=float, default=1.0)
    ap.add_argument("--max-persons", type=int, default=10)
    ap.add_argument("--seg-alpha", type=float, default=0.3)
    args = ap.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    out_names = [o.name for o in sess.get_outputs()]
    in_names = [i.name for i in sess.get_inputs()]
    print(f"ONNX providers: {sess.get_providers()}")
    print(f"ONNX inputs: {in_names}")
    print(f"ONNX outputs: {out_names}")

    det_cfg = YAMLConfig(args.det_config)
    det_post = det_cfg.postprocessor.eval()
    if hasattr(det_post, "remap_mscoco_category"):
        det_post.remap_mscoco_category = True
    pose_post = DETRPosePostProcessor(
        num_classes=2, num_keypoints=17, num_top_queries=300, remap_mscoco_category=True
    ).eval()

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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0, h0))

    tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    frame_idx = 0
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tfm(frame_pil).unsqueeze(0).numpy()
        inp = {"images": x}
        ort_out = sess.run(None, inp)
        out_map = {name: val for name, val in zip(out_names, ort_out)}

        orig_size_wh = torch.tensor([[w0, h0]], dtype=torch.float32)
        det_dict = {
            "pred_logits": torch.from_numpy(out_map["det_pred_logits"]),
            "pred_boxes": torch.from_numpy(out_map["det_pred_boxes"]),
        }
        det_res = det_post(det_dict, orig_size_wh)[0]
        det_labels = as_numpy(det_res["labels"])
        det_scores = as_numpy(det_res["scores"]).astype(np.float32)
        det_boxes = as_numpy(det_res["boxes"]).astype(np.float32)
        person_keep = np.where((det_scores >= float(args.score_thr)) & (det_labels == 1))[0]
        if person_keep.size > 0:
            dkeep = person_keep
        else:
            dkeep = np.where((det_scores >= float(args.score_thr)) & np.isin(det_labels, [0, 1]))[0]
        if dkeep.size > 0:
            dkeep = dkeep[np.argsort(det_scores[dkeep])[::-1]][: int(args.max_persons)]
        det_boxes_f = det_boxes[dkeep] if dkeep.size > 0 else np.zeros((0, 4), dtype=np.float32)
        det_scores_f = det_scores[dkeep] if dkeep.size > 0 else np.zeros((0,), dtype=np.float32)

        seg_map = np.argmax(out_map["seg_logits"], axis=1)[0].astype(np.uint8)
        vis = overlay_seg(frame.copy(), seg_map, alpha=float(args.seg_alpha))

        pose_dict = {
            "pred_logits": torch.from_numpy(out_map["pose_pred_logits"]),
            "pred_keypoints": torch.from_numpy(out_map["pose_pred_keypoints"]),
        }
        pose_res = pose_post(pose_dict, orig_size_wh)[0]
        pose_scores = as_numpy(pose_res["scores"]).astype(np.float32)
        pose_kpts = as_numpy(pose_res["keypoints"]).astype(np.float32)
        pkeep = np.where(pose_scores >= float(args.pose_score_thr))[0]
        pose_scores_f = pose_scores[pkeep] if pkeep.size > 0 else np.zeros((0,), dtype=np.float32)
        pose_kpts_f = pose_kpts[pkeep] if pkeep.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
        pose_boxes_f, pose_valid = pose_boxes_from_keypoints_conf(
            pose_kpts_f, conf_thr=float(args.pose_box_kpt_thr), min_kpts=int(args.pose_box_min_kpts)
        )
        if pose_valid.size > 0:
            pose_kpts_f = pose_kpts_f[pose_valid]
            pose_scores_f = pose_scores_f[pose_valid]

        pairs = greedy_match(
            det_boxes_f,
            pose_boxes_f,
            pose_scores_f,
            min_iou=float(args.match_min_iou),
            max_center_dist_ratio=float(args.max_center_dist_ratio),
        )
        denom = max(1, min(det_boxes_f.shape[0], pose_boxes_f.shape[0]))
        match_rate = float(len(pairs)) / float(denom)
        if match_rate < float(args.fallback_min_match_rate):
            fallback_pairs = fallback_assign_by_center(
                det_boxes_f,
                pose_boxes_f,
                pose_scores_f,
                max_center_dist_ratio=float(args.max_center_dist_ratio),
            )
            fallback_rate = float(len(fallback_pairs)) / float(denom)
            if fallback_rate > match_rate:
                pairs = fallback_pairs

        for di, pi in pairs:
            box = det_boxes_f[di]
            cv2.putText(
                vis,
                f"id {int(di)} person {float(det_scores_f[di]):.2f}",
                (int(box[0]), max(0, int(box[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            draw_pose(vis, box, pose_kpts_f[pi], kpt_thr=float(args.kpt_thr))

        out.write(vis)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"processed {frame_idx} frames")

    cap.release()
    out.release()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

