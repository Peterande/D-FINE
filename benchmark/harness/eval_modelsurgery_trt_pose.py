#!/usr/bin/env python3
"""Evaluate a fixed-shape ModelSurgery TensorRT engine on COCO keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Runner:
    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.buffers = {}
        self.shapes = {}
        self.dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.shapes[name], self.dtypes[name] = shape, dtype
            self.buffers[name] = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            self.context.set_tensor_address(name, int(self.buffers[name]))

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        cuda.memcpy_htod_async(self.buffers["images"], image, self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execution failed")
        outputs = {}
        for name, shape in self.shapes.items():
            if name == "images":
                continue
            value = np.empty(shape, dtype=self.dtypes[name])
            cuda.memcpy_dtoh_async(value, self.buffers[name], self.stream)
            outputs[name] = value
        self.stream.synchronize()
        return outputs


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    annotation = args.coco_root / "annotations/person_keypoints_val2017.json"
    coco = COCO(str(annotation))
    image_ids = sorted(coco.getImgIds(catIds=[1]))
    if args.limit:
        image_ids = image_ids[: args.limit]
    runner = Runner(args.engine)
    results = []
    for number, image_id in enumerate(image_ids, 1):
        info = coco.loadImgs(image_id)[0]
        image = Image.open(args.coco_root / "val2017" / info["file_name"]).convert("RGB")
        width, height = image.size
        resized = image.resize((640, 640), Image.Resampling.BILINEAR)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        output = runner(np.ascontiguousarray(tensor))
        logits = output["pose_pred_logits"][0]
        keypoints = output["pose_pred_keypoints"][0].reshape(-1, 17, 2)
        scores = sigmoid(logits).max(-1)
        for index in scores.argsort()[::-1][:20]:
            xy = keypoints[index] * np.asarray([width, height], dtype=np.float32)
            flat = [[float(x), float(y), 1.0] for x, y in xy]
            results.append({"image_id": int(image_id), "category_id": 1,
                            "keypoints": np.asarray(flat).reshape(-1).tolist(),
                            "score": float(scores[index])})
        if number % 500 == 0:
            print(f"[trt-pose] {number}/{len(image_ids)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    detections = coco.loadRes(str(args.out))
    evaluator = COCOeval(coco, detections, "keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    print(json.dumps({"ap": float(evaluator.stats[0]), "images": len(image_ids),
                      "predictions": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
