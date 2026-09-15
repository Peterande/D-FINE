#!/usr/bin/env python3
"""Raw tensor parity check between ONNXRuntime and TensorRT engines.

Supports both final-head parity and wide parity across all shared outputs
to localize the first strongly diverging subgraph.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.inference.trt_inf import TRTInference  # noqa: E402


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-8)
    rel = d / denom
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    an = float(np.linalg.norm(af))
    bn = float(np.linalg.norm(bf))
    cos = 0.0
    if an > 0.0 and bn > 0.0:
        cos = float(np.dot(af, bf) / (an * bn))
    return {
        "mae": float(d.mean()),
        "max_abs": float(d.max()),
        "mre": float(rel.mean()),
        "cos": cos,
    }


def _print_line(name: str, s: Dict[str, float], shape_a, shape_b):
    print(
        f"{name:18s} shape_onnx={tuple(shape_a)} shape_trt={tuple(shape_b)} "
        f"mae={s['mae']:.6e} max_abs={s['max_abs']:.6e} "
        f"mre={s['mre']:.6e} cos={s['cos']:.6f}"
    )


def _collect_ranked_parity(
    onnx_map: Dict[str, np.ndarray],
    trt_map: Dict[str, np.ndarray],
    name_filter: str,
) -> Tuple[
    List[Tuple[str, Dict[str, float], Tuple[int, ...], Tuple[int, ...]]],
    List[str],
    List[str],
    int,
]:
    def _canon(name: str) -> str:
        s = str(name).strip()
        if s.endswith(":0"):
            s = s[:-2]
        if s.startswith("onnx::"):
            s = s[len("onnx::") :]
        return s

    rows = []
    shape_mismatch = 0
    shared = sorted(set(onnx_map.keys()) & set(trt_map.keys()))
    for k in shared:
        if name_filter and name_filter not in k:
            continue
        a = _to_numpy(onnx_map[k]).astype(np.float32)
        b = _to_numpy(trt_map[k]).astype(np.float32)
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        rows.append((k, _stats(a, b), tuple(a.shape), tuple(b.shape)))

    # Fallback: canonical-name matching for cases like "linear_29:0" vs "linear_29".
    if not rows:
        onnx_canon = {}
        trt_canon = {}
        for k in onnx_map.keys():
            ck = _canon(k)
            if ck not in onnx_canon:
                onnx_canon[ck] = k
        for k in trt_map.keys():
            ck = _canon(k)
            if ck not in trt_canon:
                trt_canon[ck] = k
        shared_canon = sorted(set(onnx_canon.keys()) & set(trt_canon.keys()))
        for ck in shared_canon:
            if name_filter and name_filter not in ck:
                continue
            ko = onnx_canon[ck]
            kt = trt_canon[ck]
            a = _to_numpy(onnx_map[ko]).astype(np.float32)
            b = _to_numpy(trt_map[kt]).astype(np.float32)
            if a.shape != b.shape:
                shape_mismatch += 1
                continue
            rows.append((ck, _stats(a, b), tuple(a.shape), tuple(b.shape)))

    if name_filter:
        onnx_only = sorted([k for k in onnx_map if name_filter in k and k not in trt_map])
        trt_only = sorted([k for k in trt_map if name_filter in k and k not in onnx_map])
    else:
        onnx_only = sorted([k for k in onnx_map if k not in trt_map])
        trt_only = sorted([k for k in trt_map if k not in onnx_map])
    return rows, onnx_only, trt_only, shape_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--trt", required=True)
    ap.add_argument("--input", required=True, help="Video path")
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--plugin-lib",
        action="append",
        default=[],
        help="Path to custom TensorRT plugin .so/.dll. May be repeated.",
    )
    ap.add_argument(
        "--mode",
        choices=["heads", "all"],
        default="heads",
        help="heads: compare primary task heads only. all: compare all shared outputs.",
    )
    ap.add_argument(
        "--name-filter",
        default="",
        help="Optional substring filter for --mode all (e.g. 'pose' or 'linear_').",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=30,
        help="When --mode all, print top-k tensors sorted by MAE.",
    )
    ap.add_argument(
        "--sort-by",
        choices=["mae", "max_abs", "mre", "cos", "name"],
        default="mae",
        help="Sort order for --mode all table.",
    )
    ap.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="How many ONNX-only / TRT-only names to print in --mode all.",
    )
    args = ap.parse_args()

    # Read one frame
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    idx = 0
    frame = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx == int(args.frame_idx):
            frame = fr
            break
        idx += 1
    cap.release()
    if frame is None:
        raise RuntimeError(f"Could not read frame {args.frame_idx} from {args.input}")

    # Common preprocessing
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    x = tfm(frame_pil).unsqueeze(0)
    x_np = x.numpy()

    # ONNX
    sess = ort.InferenceSession(
        args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    onnx_input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_input_name: x_np})
    onnx_names = [o.name for o in sess.get_outputs()]
    onnx_map = {k: v for k, v in zip(onnx_names, onnx_out)}

    # TRT
    trt = TRTInference(
        args.trt, device=args.device, max_batch_size=1, plugin_libs=args.plugin_lib
    )
    blob = {"images": x.to(args.device)}
    trt_map_t = trt(blob)
    trt_map = {k: _to_numpy(v) for k, v in trt_map_t.items()}

    print("=== ONNX vs TRT raw parity ===")
    print(f"frame_idx={args.frame_idx} input={args.input} mode={args.mode}")
    if args.mode == "heads":
        keys = [
            "det_pred_logits",
            "det_pred_boxes",
            "pose_pred_logits",
            "pose_pred_keypoints",
            "seg_logits",
        ]
        for k in keys:
            if k not in onnx_map:
                print(f"{k:18s} missing in ONNX outputs")
                continue
            if k not in trt_map:
                print(f"{k:18s} missing in TRT outputs")
                continue
            a = _to_numpy(onnx_map[k]).astype(np.float32)
            b = _to_numpy(trt_map[k]).astype(np.float32)
            if a.shape != b.shape:
                print(f"{k:18s} shape mismatch onnx={a.shape} trt={b.shape}")
                continue
            s = _stats(a, b)
            _print_line(k, s, a.shape, b.shape)
        return

    rows, onnx_only, trt_only, shape_mismatch = _collect_ranked_parity(
        onnx_map, trt_map, args.name_filter
    )
    if not rows:
        print("No comparable tensors found in --mode all with current filter.")
        if args.name_filter:
            onnx_f = sorted([k for k in onnx_map.keys() if args.name_filter in str(k)])
            trt_f = sorted([k for k in trt_map.keys() if args.name_filter in str(k)])
            print(f"filtered_onnx_names={len(onnx_f)} filtered_trt_names={len(trt_f)}")
            if onnx_f:
                print("--- ONNX filtered names (first 30) ---")
                for n in onnx_f[:30]:
                    print(n)
            if trt_f:
                print("--- TRT filtered names (first 30) ---")
                for n in trt_f[:30]:
                    print(n)
        return

    if args.sort_by == "name":
        rows_sorted = sorted(rows, key=lambda x: x[0])
    elif args.sort_by == "cos":
        rows_sorted = sorted(rows, key=lambda x: x[1]["cos"])
    else:
        rows_sorted = sorted(rows, key=lambda x: x[1][args.sort_by], reverse=True)

    print(
        f"onnx_outputs={len(onnx_map)} trt_outputs={len(trt_map)} "
        f"shared_compared={len(rows_sorted)} "
        f"shape_mismatch={shape_mismatch} filter='{args.name_filter or '*'}' "
        f"onnx_only={len(onnx_only)} trt_only={len(trt_only)}"
    )
    print(f"--- Top {min(len(rows_sorted), int(args.topk))} by {args.sort_by} ---")
    for name, s, ash, bsh in rows_sorted[: int(args.topk)]:
        _print_line(name, s, ash, bsh)
    if int(args.show_missing) > 0:
        if onnx_only:
            print(f"--- ONNX-only (showing {min(len(onnx_only), int(args.show_missing))}) ---")
            for n in onnx_only[: int(args.show_missing)]:
                print(n)
        if trt_only:
            print(f"--- TRT-only (showing {min(len(trt_only), int(args.show_missing))}) ---")
            for n in trt_only[: int(args.show_missing)]:
                print(n)


if __name__ == "__main__":
    main()
