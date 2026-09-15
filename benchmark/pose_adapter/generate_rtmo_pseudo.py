#!/usr/bin/env python3
"""Generate RTMO pseudo keypoints for an isolated training subset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from rtmlib import RTMO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--engine", help="Validated RTMO TensorRT engine (faster than CPU ONNX).")
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--score-thr", type=float, default=0.20)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.model) == bool(args.engine):
        parser.error("supply exactly one of --model or --engine")

    ann = args.coco_root / "annotations" / f"person_keypoints_{args.split}.json"
    coco = COCO(str(ann))
    ids = sorted(coco.getImgIds(catIds=[1]))
    if args.limit > 0:
        ids = ids[: args.limit]
    model = None
    runner = None
    if args.engine:
        from benchmark.harness.eval_rtmo_trt import Runner
        from rtmlib.tools.object_detection.post_processings import multiclass_nms

        runner = Runner(args.engine)
    else:
        model = RTMO(args.model, model_input_size=(640, 640), score_thr=0.01,
                     nms_thr=0.65, device="cpu")
    results = []
    completed = set()
    if args.out.exists():
        saved = json.loads(args.out.read_text())
        if isinstance(saved, dict):
            results = saved.get("predictions", [])
            completed = set(saved.get("completed_image_ids", []))
        else:  # Backward-compatible with the original proof-of-concept artifact.
            results = saved
            completed = {int(item["image_id"]) for item in saved}
        print(f"[rtmo-pseudo] resume: {len(completed)} images, {len(results)} poses")

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps({"completed_image_ids": sorted(completed),
                                         "predictions": results}))
        temporary.replace(args.out)

    for number, image_id in enumerate(ids, 1):
        if image_id in completed:
            continue
        info = coco.loadImgs(image_id)[0]
        image = cv2.imread(str(args.coco_root / args.split / info["file_name"]))
        if runner is not None:
            (detections, keypoints), ratio = runner(image)
            detections, keypoints = detections[0], keypoints[0]
            _, keep = multiclass_nms(detections[:, :4] / ratio, detections[:, 4, None],
                                     nms_thr=0.65, score_thr=0.001)
            keypoints = keypoints[keep] if keep is not None else keypoints[:0]
            keypoints[..., :2] /= ratio
            scores = keypoints[..., 2]
            keypoints = keypoints[..., :2]
        else:
            keypoints, scores = model(image)
            keypoints = np.asarray(keypoints)
            scores = np.asarray(scores)
        if keypoints.ndim == 2:
            keypoints = keypoints[None]
            scores = scores[None]
        for pose, joint_scores in zip(keypoints, scores):
            confidence = float(np.mean(joint_scores))
            if confidence < args.score_thr:
                continue
            flat = []
            for (x, y), score in zip(pose, joint_scores):
                flat.extend([float(x), float(y), float(score)])
            results.append({"image_id": int(image_id), "keypoints": flat,
                            "score": confidence})
        completed.add(int(image_id))
        if number % 50 == 0:
            print(f"[rtmo-pseudo] {number}/{len(ids)}", flush=True)
        if len(completed) % args.save_every == 0:
            save()
    save()
    print(json.dumps({"images": len(completed), "poses": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
