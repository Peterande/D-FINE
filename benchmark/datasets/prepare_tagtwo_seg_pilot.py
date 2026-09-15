#!/usr/bin/env python3
"""Build a review-first TagTwo body-part segmentation pilot from videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "src", REPO / "tools", REPO / "segmentation_sivert",
             REPO / "pose_estimation_berna"):
    sys.path.insert(0, str(path))

from benchmark.harness.infer_crosshair import build_model_from_merged  # noqa: E402

CLASSES = ["background", "head", "torso", "arms", "hands", "legs", "feet"]
COLORS_BGR = np.asarray([
    [0, 0, 0], [40, 40, 230], [60, 190, 60], [230, 150, 30],
    [210, 60, 210], [230, 220, 40], [40, 220, 220],
], dtype=np.uint8)


def scan_video(path: Path) -> tuple[list[dict], float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows, previous = [], None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        motion = 0.0 if previous is None else float(cv2.absdiff(small, previous).mean())
        rows.append({"frame": index, "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
                     "motion": motion, "brightness": float(small.mean())})
        previous = small
        index += 1
    cap.release()
    return rows, fps


def choose_frames(rows: list[dict], count: int, video_name: str) -> dict[int, set[str]]:
    chosen: dict[int, set[str]] = {}
    per_group = max(1, count // 4)

    def add(candidates, reason):
        for row in candidates:
            frame = int(row["frame"])
            if any(abs(frame - existing) < 4 for existing in chosen):
                continue
            chosen.setdefault(frame, set()).add(reason)
            if sum(reason in reasons for reasons in chosen.values()) >= per_group:
                break

    add(sorted(rows, key=lambda row: row["motion"], reverse=True), "high_motion")
    add(sorted(rows, key=lambda row: row["sharpness"]), "low_sharpness")
    add(sorted(rows, key=lambda row: row["brightness"]), "low_light")
    temporal = [rows[round(i * (len(rows) - 1) / max(1, per_group - 1))] for i in range(per_group)]
    add(temporal, "temporal_coverage")

    # Known chair-occlusion control in David. This is selection metadata, not a label.
    if video_name.lower() == "david":
        for frame in np.linspace(340, min(470, len(rows) - 1), 10).round().astype(int):
            chosen.setdefault(int(frame), set()).add("chair_occlusion_control")

    if len(chosen) > count:
        mandatory = {f for f, reasons in chosen.items() if "chair_occlusion_control" in reasons}
        rest = [f for f in sorted(chosen) if f not in mandatory]
        keep = mandatory | set(rest[: max(0, count - len(mandatory))])
        chosen = {f: chosen[f] for f in sorted(keep)}
    elif len(chosen) < count:
        for row in rows:
            frame = int(row["frame"])
            if frame not in chosen and all(abs(frame - old) >= 4 for old in chosen):
                chosen[frame] = {"coverage_fill"}
                if len(chosen) == count:
                    break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--frames-per-video", type=int, default=40)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    for directory in ("images", "preannotations", "overlays"):
        (args.out / directory).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model_from_merged(args.det_config, args.pose_config, args.checkpoint,
                                       7, 384, 0.1, 640)
    model = model.to(device).eval()
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    manifest = []

    for video in args.video:
        rows, fps = scan_video(video)
        metrics = {int(row["frame"]): row for row in rows}
        selected = choose_frames(rows, args.frames_per_video, video.stem)
        cap = cv2.VideoCapture(str(video))
        for frame_index, reasons in sorted(selected.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read {video}:{frame_index}")
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(tensor)["seg.logits"].argmax(1)[0].byte().cpu().numpy()
            mask = cv2.resize(output, (width, height), interpolation=cv2.INTER_NEAREST)
            stem = f"{video.stem.lower()}_{frame_index:06d}"
            cv2.imwrite(str(args.out / "images" / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(args.out / "preannotations" / f"{stem}.png"), mask)
            color = COLORS_BGR[mask]
            overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0)
            cv2.putText(overlay, "PREANNOTATION - NEEDS MANUAL REVIEW", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.out / "overlays" / f"{stem}.jpg"), overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            metric = metrics[frame_index]
            manifest.append({"image": f"images/{stem}.jpg",
                             "preannotation": f"preannotations/{stem}.png",
                             "source_video": str(video), "frame": frame_index,
                             "timestamp_seconds": round(frame_index / fps, 3),
                             "selection_reason": ";".join(sorted(reasons)),
                             "motion": round(metric["motion"], 3),
                             "sharpness": round(metric["sharpness"], 3),
                             "brightness": round(metric["brightness"], 3),
                             "review_status": "needs_manual_review"})
        cap.release()

    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    (args.out / "labels.json").write_text(json.dumps(
        {"classes": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]}, indent=2))
    (args.out / "README.md").write_text(
        "# TagTwo segmentation pilot\n\n"
        "The `images/` folder contains the annotation inputs. `preannotations/` are model-generated "
        "class-ID PNG masks and are **not ground truth**. Use `overlays/` only as a review aid. "
        "Correct every mask manually, mark the row reviewed in `manifest.csv`, and keep reviewed "
        "masks in a separate `ground_truth/` folder. Do not train on this pilot if it is used for "
        "final evaluation. Classes: background=0, head=1, torso=2, arms=3, hands=4, legs=5, feet=6.\n")
    print(json.dumps({"frames": len(manifest), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
