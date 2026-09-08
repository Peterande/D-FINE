#!/usr/bin/env python3
"""
Experimental dynamic-batch inference pipeline for TensorRT.

This keeps the stable single-frame pipeline untouched and provides a separate
runner for batching experiments.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import cv2
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in os.sys.path:
    os.sys.path.insert(0, REPO)

from tools.inference.trt_inf import (  # noqa: E402
    TRTInference,
    _as_numpy,
    _candidate_pose_quality,
    _find_output_key,
    _get_detection_outputs,
    draw_pose,
    fallback_assign_by_center,
    greedy_match,
    overlay_seg,
    pose_boxes_from_keypoints_conf,
)


def _preprocess_frame_bgr_to_tensor(frame_bgr: np.ndarray, device: str):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_640 = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb_640).to(device=device, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().div_(255.0)
    orig_size_wh = torch.tensor([w, h], device=device, dtype=torch.float32).unsqueeze(0)
    orig_size_hw = torch.tensor([h, w], device=device, dtype=torch.float32).unsqueeze(0)
    return x, orig_size_wh, orig_size_hw


def _slice_output_map(output_b: Dict[str, torch.Tensor], bi: int, bsz: int):
    out = {}
    for k, v in output_b.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and int(v.shape[0]) == int(bsz):
            out[k] = v[bi : bi + 1]
        else:
            out[k] = v
    return out


def process_video_batched(
    m: TRTInference,
    file_path: str,
    out_path: str,
    device: str,
    det_post,
    pose_post,
    dynamic_batch_size: int,
    score_thr: float,
    pose_score_thr: float,
    kpt_thr: float,
    pose_box_kpt_thr: float,
    pose_box_min_kpts: int,
    match_min_iou: float,
    max_center_dist_ratio: float,
    fallback_min_match_rate: float,
    pose_orig_size_order: str,
    max_persons: int,
    seg_alpha: float,
    pose_topk: int,
):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError(f"Invalid input size from {file_path}: {orig_w}x{orig_h}")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (orig_w, orig_h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output writer: {out_path}")

    frame_count = 0
    t0 = time.perf_counter()
    print(f"[batch] processing with dynamic_batch_size={dynamic_batch_size}")

    try:
        while True:
            frames: List[np.ndarray] = []
            for _ in range(max(1, int(dynamic_batch_size))):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            if not frames:
                break

            batch_imgs = []
            batch_wh = []
            batch_hw = []
            for fr in frames:
                x, swh, shw = _preprocess_frame_bgr_to_tensor(fr, device=device)
                batch_imgs.append(x)
                batch_wh.append(swh)
                batch_hw.append(shw)

            images_b = torch.cat(batch_imgs, dim=0)
            orig_size_wh_b = torch.cat(batch_wh, dim=0)
            blob = {"images": images_b, "orig_target_sizes": orig_size_wh_b}

            output_b = m(blob)
            bsz = len(frames)

            for bi, frame in enumerate(frames):
                output = _slice_output_map(output_b, bi, bsz)
                orig_size_wh = batch_wh[bi]
                orig_size_hw = batch_hw[bi]
                frame = frame.copy()

                labels, boxes, scores = _get_detection_outputs(output, orig_size_wh, det_post=det_post)
                labels = _as_numpy(labels)
                boxes = _as_numpy(boxes).astype(np.float32)
                scores = _as_numpy(scores).astype(np.float32)

                person_keep = np.where((scores >= float(score_thr)) & (labels == 1))[0]
                if person_keep.size > 0:
                    keep = person_keep
                else:
                    keep = np.where((scores >= float(score_thr)) & ((labels == 1) | (labels == 0)))[0]
                if keep.size > 0:
                    keep = keep[np.argsort(scores[keep])[::-1][: int(max_persons)]]
                det_boxes_f = boxes[keep] if keep.size > 0 else np.zeros((0, 4), dtype=np.float32)
                det_scores_f = scores[keep] if keep.size > 0 else np.zeros((0,), dtype=np.float32)

                seg_k = _find_output_key(output, ["seg_logits"])
                if seg_k is not None:
                    seg_logits = output[seg_k]
                    if isinstance(seg_logits, torch.Tensor):
                        seg_map = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                        frame = overlay_seg(frame, seg_map, alpha=float(seg_alpha))

                pose_logit_k = _find_output_key(output, ["pose_pred_logits"])
                pose_kpt_k = _find_output_key(output, ["pose_pred_keypoints"])
                if pose_logit_k is not None and pose_kpt_k is not None and pose_post is not None:
                    if hasattr(pose_post, "num_top_queries"):
                        try:
                            q_dim = int(_as_numpy(output[pose_logit_k]).shape[1])
                            pose_post.num_top_queries = int(min(max(1, int(pose_topk)), q_dim)) if int(pose_topk) > 0 else int(q_dim)
                        except Exception:
                            pass
                    pose_dict = {"pred_logits": output[pose_logit_k], "pred_keypoints": output[pose_kpt_k]}
                    pose_res_wh = pose_post(pose_dict, orig_size_wh)[0]
                    pose_scores_wh = _as_numpy(pose_res_wh["scores"]).astype(np.float32)
                    pose_kpts_wh = _as_numpy(pose_res_wh["keypoints"]).astype(np.float32)
                    pkeep_wh = np.where(pose_scores_wh >= float(pose_score_thr))[0]
                    pose_scores_f = pose_scores_wh[pkeep_wh] if pkeep_wh.size > 0 else np.zeros((0,), dtype=np.float32)
                    pose_kpts_f = pose_kpts_wh[pkeep_wh] if pkeep_wh.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)

                    if pose_orig_size_order in ["hw", "auto"]:
                        pose_res_hw = pose_post(pose_dict, orig_size_hw)[0]
                        pose_scores_hw = _as_numpy(pose_res_hw["scores"]).astype(np.float32)
                        pose_kpts_hw = _as_numpy(pose_res_hw["keypoints"]).astype(np.float32)
                        pkeep_hw = np.where(pose_scores_hw >= float(pose_score_thr))[0]
                        pose_scores_hw_f = pose_scores_hw[pkeep_hw] if pkeep_hw.size > 0 else np.zeros((0,), dtype=np.float32)
                        pose_kpts_hw_f = pose_kpts_hw[pkeep_hw] if pkeep_hw.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
                        if pose_orig_size_order == "hw":
                            pose_scores_f = pose_scores_hw_f
                            pose_kpts_f = pose_kpts_hw_f
                        else:
                            q_wh = _candidate_pose_quality(
                                det_boxes_f, pose_kpts_f, kpt_thr=float(pose_box_kpt_thr), min_visible_kpts=int(pose_box_min_kpts)
                            )
                            q_hw = _candidate_pose_quality(
                                det_boxes_f, pose_kpts_hw_f, kpt_thr=float(pose_box_kpt_thr), min_visible_kpts=int(pose_box_min_kpts)
                            )
                            if q_hw > q_wh:
                                pose_scores_f = pose_scores_hw_f
                                pose_kpts_f = pose_kpts_hw_f

                    pose_boxes_all, pose_valid_mask = pose_boxes_from_keypoints_conf(
                        pose_kpts_f, conf_thr=float(pose_box_kpt_thr), min_kpts=int(pose_box_min_kpts)
                    )
                    pose_kpts_f = pose_kpts_f[pose_valid_mask] if pose_valid_mask.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
                    pose_scores_f = pose_scores_f[pose_valid_mask] if pose_valid_mask.size > 0 else np.zeros((0,), dtype=np.float32)
                    pose_boxes_f = pose_boxes_all

                    for i, b in enumerate(det_boxes_f):
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"person {float(det_scores_f[i]):.2f}",
                            (int(b[0]), max(0, int(b[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    if det_boxes_f.shape[0] > 0 and pose_boxes_f.shape[0] > 0:
                        pairs = greedy_match(
                            det_boxes_f,
                            pose_boxes_f,
                            pose_scores_f,
                            min_iou=float(match_min_iou),
                            max_center_dist_ratio=float(max_center_dist_ratio),
                        )
                        denom = max(1, min(det_boxes_f.shape[0], pose_boxes_f.shape[0]))
                        if float(len(pairs)) / float(denom) < float(fallback_min_match_rate):
                            fallback_pairs = fallback_assign_by_center(
                                det_boxes_f, pose_boxes_f, pose_scores_f, max_center_dist_ratio=float(max_center_dist_ratio)
                            )
                            if len(fallback_pairs) > len(pairs):
                                pairs = fallback_pairs
                        for di, pi in pairs:
                            draw_pose(frame, det_boxes_f[di], pose_kpts_f[pi], kpt_thr=float(kpt_thr))
                else:
                    for i, b in enumerate(det_boxes_f):
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"person {float(det_scores_f[i]):.2f}",
                            (int(b[0]), max(0, int(b[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                out.write(frame)
                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = max(1e-9, time.perf_counter() - t0)
                    fps_avg = frame_count / elapsed
                    ms_frame = elapsed * 1000.0 / frame_count
                    print(f"[batch] frames={frame_count} avg_fps={fps_avg:.2f} avg_ms={ms_frame:.2f}")

    finally:
        cap.release()
        out.release()

    elapsed = max(1e-9, time.perf_counter() - t0)
    fps_avg = frame_count / elapsed if frame_count > 0 else 0.0
    ms_frame = elapsed * 1000.0 / frame_count if frame_count > 0 else 0.0
    size = os.path.getsize(out_path) if os.path.isfile(out_path) else -1
    print(
        f"[done] frames={frame_count} out='{out_path}' size_bytes={size} "
        f"elapsed_s={elapsed:.3f} fps={fps_avg:.2f} ms_per_frame={ms_frame:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-trt", "--trt", required=True)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="trt_result_batch.mp4")
    ap.add_argument("-d", "--device", default="cuda:0")
    ap.add_argument("--plugin-lib", action="append", default=[])
    ap.add_argument("--det-config", type=str, default=None)
    ap.add_argument("--dynamic-batch-size", type=int, default=2)
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--pose-score-thr", type=float, default=0.4)
    ap.add_argument("--kpt-thr", type=float, default=0.4)
    ap.add_argument("--pose-box-kpt-thr", type=float, default=0.45)
    ap.add_argument("--pose-box-min-kpts", type=int, default=6)
    ap.add_argument("--match-min-iou", type=float, default=0.25)
    ap.add_argument("--max-center-dist-ratio", type=float, default=0.30)
    ap.add_argument("--fallback-min-match-rate", type=float, default=1.0)
    ap.add_argument("--pose-orig-size-order", choices=["wh", "hw", "auto"], default="auto")
    ap.add_argument("--max-persons", type=int, default=10)
    ap.add_argument("--seg-alpha", type=float, default=0.3)
    ap.add_argument("--pose-topk", type=int, default=60)
    args = ap.parse_args()

    m = TRTInference(args.trt, device=args.device, plugin_libs=args.plugin_lib)
    print(f"Engine outputs: {m.output_names}")

    det_post = None
    pose_post = None
    if args.det_config:
        from src.core import YAMLConfig
        from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor

        det_cfg = YAMLConfig(args.det_config)
        det_post = det_cfg.postprocessor.to(args.device).eval()
        if hasattr(det_post, "remap_mscoco_category"):
            det_post.remap_mscoco_category = True
        pose_post = DETRPosePostProcessor(
            num_classes=2,
            num_keypoints=17,
            num_top_queries=300,
            remap_mscoco_category=True,
        ).to(args.device).eval()

    process_video_batched(
        m=m,
        file_path=args.input,
        out_path=args.out,
        device=args.device,
        det_post=det_post,
        pose_post=pose_post,
        dynamic_batch_size=max(1, int(args.dynamic_batch_size)),
        score_thr=float(args.score_thr),
        pose_score_thr=float(args.pose_score_thr),
        kpt_thr=float(args.kpt_thr),
        pose_box_kpt_thr=float(args.pose_box_kpt_thr),
        pose_box_min_kpts=int(args.pose_box_min_kpts),
        match_min_iou=float(args.match_min_iou),
        max_center_dist_ratio=float(args.max_center_dist_ratio),
        fallback_min_match_rate=float(args.fallback_min_match_rate),
        pose_orig_size_order=str(args.pose_orig_size_order),
        max_persons=int(args.max_persons),
        seg_alpha=float(args.seg_alpha),
        pose_topk=int(args.pose_topk),
    )


if __name__ == "__main__":
    main()

