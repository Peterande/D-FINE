#!/usr/bin/env python3
"""Render official HumanQueryNet detection, pose, instance masks and attributes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector

from models.builder import build_detector

SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth source frame and divide output FPS by N.")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.bbox_head.with_smpl = False
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.cfg = cfg
    model.eval()

    cap = cv2.VideoCapture(args.input)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride,
                             (width, height))
    frame_count, source_index, model_seconds = 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames is not None and frame_count >= args.max_frames):
            break
        if source_index % args.stride:
            source_index += 1
            continue
        source_index += 1
        started = time.perf_counter()
        result = inference_detector(model, frame)
        model_seconds += time.perf_counter() - started

        mask_boxes, masks = result["ins_results"][0]
        if mask_boxes:
            mask_boxes, masks = mask_boxes[0], masks[0]
            tint = np.zeros_like(frame)
            union = np.zeros((height, width), dtype=bool)
            for box, mask in zip(mask_boxes, masks):
                if float(box[4]) >= args.score_threshold:
                    union |= np.asarray(mask, dtype=bool)
            tint[union] = (40, 190, 60)
            frame = cv2.addWeighted(frame, 1.0, tint, 0.38, 0)

        detections = result["det"][0][0]
        for detection in detections:
            score = float(detection[4])
            if score < args.score_threshold:
                continue
            x1, y1, x2, y2 = detection[:4].astype(int)
            gender, age = float(detection[5]), float(detection[6])
            keypoints = detection[-34:].reshape(17, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 255), 2)
            label = f"person {score:.2f} age~{age:.0f} {'F' if gender > 0.5 else 'M'}"
            cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (40, 220, 255), 1, cv2.LINE_AA)
            for a, b in SKELETON:
                pa, pb = tuple(keypoints[a].astype(int)), tuple(keypoints[b].astype(int))
                cv2.line(frame, pa, pb, (255, 80, 180), 2, cv2.LINE_AA)
            for point in keypoints:
                cv2.circle(frame, tuple(point.astype(int)), 2, (255, 255, 255), -1)
        cv2.putText(frame, "HumanQueryNet: person mask (not body parts)", (8, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[hqnet] {frame_count} frames", flush=True)
    cap.release(); writer.release()
    print({"frames": frame_count, "model_ms_per_frame": 1000 * model_seconds / max(1, frame_count),
           "out": args.out})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
