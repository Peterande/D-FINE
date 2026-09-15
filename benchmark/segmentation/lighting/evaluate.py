#!/usr/bin/env python3
"""Evaluate ModelSurgery segmentation consistency under controlled lighting changes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
COLORS = np.asarray([
    [0, 0, 0], [40, 40, 230], [60, 190, 60], [230, 150, 30],
    [210, 60, 210], [230, 220, 40], [40, 220, 220],
], dtype=np.uint8)


def clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def variants(image: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    value = image.astype(np.float32)
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]

    dark = clip(255.0 * np.power(value / 255.0, 1.45) * 0.55)
    dark_3x = clip(value / 3.0)
    dark_10x = clip(value / 10.0)
    overexposed = clip(value * 1.45 + 42.0)
    warm = clip(value * np.asarray([0.72, 0.96, 1.25], dtype=np.float32))
    cold = clip(value * np.asarray([1.28, 1.02, 0.75], dtype=np.float32))

    diagonal = 0.25 + 0.75 * (xx + yy) / max(1, width + height - 2)
    strong_shadow = clip(value * diagonal[..., None])

    distance = ((xx - 0.52 * width) / max(1, 0.55 * width)) ** 2
    distance += ((yy - 0.38 * height) / max(1, 0.55 * height)) ** 2
    glow = np.exp(-3.2 * distance)[..., None]
    backlight = clip(value * (0.55 + 0.25 * (1.0 - glow)) + 150.0 * glow)

    led_mix = np.empty_like(value)
    blend = (xx / max(1, width - 1))[..., None]
    cyan = np.asarray([55, 18, 0], dtype=np.float32)
    magenta = np.asarray([15, 0, 60], dtype=np.float32)
    cast = cyan * (1.0 - blend) + magenta * blend
    led_mix = clip(value * 0.78 + cast)

    rng = np.random.default_rng(seed)
    noisy = clip(value * 0.72 + rng.normal(0.0, 18.0, value.shape))
    return {
        "dark": dark,
        "dark_3x": dark_3x,
        "dark_10x": dark_10x,
        "overexposed": overexposed,
        "warm": warm,
        "cold": cold,
        "strong_shadow": strong_shadow,
        "backlight": backlight,
        "led_color_cast": led_mix,
        "low_light_noise": noisy,
    }


def predict(model, transform, device, image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        mask = model(tensor)["seg.logits"].argmax(1)[0].byte().cpu().numpy()
    return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


def class_metrics(reference: np.ndarray, candidate: np.ndarray, class_id: int) -> tuple[float, float, float] | None:
    ref = reference == class_id
    pred = candidate == class_id
    union = int(np.logical_or(ref, pred).sum())
    if union == 0:
        return None
    intersection = int(np.logical_and(ref, pred).sum())
    ref_area, pred_area = int(ref.sum()), int(pred.sum())
    iou = intersection / union
    retention = intersection / ref_area if ref_area else float("nan")
    area_ratio = pred_area / ref_area if ref_area else float("nan")
    return iou, retention, area_ratio


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image, 0.65, COLORS[mask], 0.35, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seg-head-ckpt", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    comparisons = args.out / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model_from_merged(args.det_config, args.pose_config, args.checkpoint,
                                       7, 384, 0.1, 640)
    if args.seg_head_ckpt:
        checkpoint = torch.load(args.seg_head_ckpt, map_location="cpu", weights_only=False)
        model.seg_head.load_state_dict(checkpoint["seg_head"], strict=True)
    model = model.to(device).eval()
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    rows = []
    images = sorted((args.pilot / "images").glob("*.jpg"))
    for image_number, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        reference = predict(model, transform, device, image)
        for condition, changed in variants(image, image_number).items():
            candidate = predict(model, transform, device, changed)
            condition_dir = comparisons / condition
            condition_dir.mkdir(exist_ok=True)
            panels = [overlay(image, reference), changed, overlay(changed, candidate)]
            comparison = np.hstack(panels)
            labels = ["NORMAL + MASK", condition.upper(), f"{condition.upper()} + MASK"]
            panel_width = image.shape[1]
            for index, label in enumerate(labels):
                cv2.putText(comparison, label, (index * panel_width + 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(condition_dir / image_path.name), comparison,
                        [cv2.IMWRITE_JPEG_QUALITY, 91])
            for class_id, class_name in enumerate(CLASSES):
                metrics = class_metrics(reference, candidate, class_id)
                if metrics is None:
                    continue
                iou, retention, area_ratio = metrics
                rows.append({"image": image_path.name, "condition": condition,
                             "class_id": class_id, "class": class_name,
                             "consistency_iou": iou, "reference_retention": retention,
                             "predicted_area_ratio": area_ratio})
        if (image_number + 1) % 10 == 0:
            print(f"[lighting] {image_number + 1}/{len(images)}", flush=True)

    with (args.out / "per_image_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["class"])].append(row)
    summary = []
    for (condition, class_name), items in sorted(grouped.items()):
        ious = np.asarray([item["consistency_iou"] for item in items], dtype=float)
        retention = np.asarray([item["reference_retention"] for item in items], dtype=float)
        ratios = np.asarray([item["predicted_area_ratio"] for item in items], dtype=float)
        summary.append({"condition": condition, "class": class_name, "samples": len(items),
                        "mean_iou": float(np.nanmean(ious)),
                        "mean_retention": float(np.nanmean(retention)),
                        "mean_area_ratio": float(np.nanmean(ratios))})
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    foreground = defaultdict(list)
    for row in summary:
        if row["class"] != "background":
            foreground[row["condition"]].append(row["mean_iou"])
    ranked = sorted(((condition, float(np.mean(values))) for condition, values in foreground.items()),
                    key=lambda item: item[1])
    report = ["# ModelSurgery segmentation lighting robustness", "",
              "These are prediction-consistency results, not ground-truth accuracy. A lower score",
              "means the unchanged model altered its mask more when only lighting changed.", "",
              "| Lighting condition | Mean foreground consistency IoU |", "|---|---:|"]
    report.extend(f"| {condition} | {score * 100:.2f}% |" for condition, score in ranked)
    report += ["", "Detailed hand/foot and per-image values are in `summary.csv` and",
               "`per_image_class.csv`. Visual comparisons are under `comparisons/`."]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"images": len(images), "conditions": len(variants(np.zeros((2, 2, 3), np.uint8), 0)),
                      "worst_condition": ranked[0][0], "worst_foreground_iou": ranked[0][1],
                      "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
