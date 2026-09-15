#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


LABELS = ["background", "head", "torso", "arms", "hands", "legs", "feet"]


def metrics(confusion: np.ndarray) -> dict:
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    return {"miou": float(iou.mean()), "pixel_accuracy": float(intersection.sum()/confusion.sum()),
            "per_class_iou": {name: float(value) for name, value in zip(LABELS, iou)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model.resolve()))
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = None
    session = None
    if args.onnx:
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).cuda().eval()
    images = sorted((args.dataset / "images" / "val").glob("*.jpg"))
    direct = np.zeros((7,7), dtype=np.int64); semantic = np.zeros((7,7), dtype=np.int64)
    remap = np.array([0,1,2,3,3,5,5], dtype=np.uint8)
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            paths = images[start:start+args.batch_size]
            pil = [Image.open(path).convert("RGB") for path in paths]
            if session:
                batch = processor(images=pil, return_tensors="np")["pixel_values"]
                output = torch.from_numpy(session.run(["logits"], {"pixel_values": batch})[0])
            else:
                batch = processor(images=pil, return_tensors="pt")["pixel_values"].cuda()
                output = model(pixel_values=batch).logits
            for index, path in enumerate(paths):
                target = np.asarray(Image.open(args.dataset/"masks"/"val"/f"{path.stem}.png"), dtype=np.uint8)
                logits = torch.nn.functional.interpolate(output[index:index+1], target.shape, mode="bilinear", align_corners=False)
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                for matrix, values in ((direct,pred),(semantic,remap[pred])):
                    matrix += np.bincount(7*target.reshape(-1)+values.reshape(-1), minlength=49).reshape(7,7)
            if start and start % 400 == 0: print(f"images={start}/{len(images)}", flush=True)
    result={"samples":len(images),"warning":"direct IDs are semantically invalid for classes 3..6",
            "direct_id_metrics":metrics(direct),"project_semantic_remap":metrics(semantic)}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
