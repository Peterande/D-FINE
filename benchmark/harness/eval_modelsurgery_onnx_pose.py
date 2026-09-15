#!/usr/bin/env python3
"""Evaluate ModelSurgery ONNX pose output with the standard COCO protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    coco = COCO(str(args.coco_root / "annotations/person_keypoints_val2017.json"))
    image_ids = sorted(coco.getImgIds(catIds=[1]))
    if args.limit:
        image_ids = image_ids[: args.limit]
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    results = []
    for number, image_id in enumerate(image_ids, 1):
        info = coco.loadImgs(image_id)[0]
        image = Image.open(args.coco_root / "val2017" / info["file_name"]).convert("RGB")
        width, height = image.size
        resized = image.resize((640, 640), Image.Resampling.BILINEAR)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        logits, keypoints = session.run(
            ["pose_pred_logits", "pose_pred_keypoints"], {"images": np.ascontiguousarray(tensor)})
        scores = (1.0 / (1.0 + np.exp(-logits[0]))).max(-1)
        poses = keypoints[0].reshape(-1, 17, 2)
        for index in scores.argsort()[::-1][:20]:
            xy = poses[index] * np.asarray([width, height], dtype=np.float32)
            flat = [[float(x), float(y), 1.0] for x, y in xy]
            results.append({"image_id": int(image_id), "category_id": 1,
                            "keypoints": np.asarray(flat).reshape(-1).tolist(),
                            "score": float(scores[index])})
        if number % 50 == 0:
            print(f"[onnx-pose] {number}/{len(image_ids)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(results))
    detections = coco.loadRes(str(args.out)); evaluator = COCOeval(coco, detections, "keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    print(json.dumps({"ap": float(evaluator.stats[0]), "images": len(image_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
